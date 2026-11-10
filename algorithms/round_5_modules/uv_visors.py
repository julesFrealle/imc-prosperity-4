from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Any
import json
import math


AMBER = "UV_VISOR_AMBER"
MAGENTA = "UV_VISOR_MAGENTA"
ORANGE = "UV_VISOR_ORANGE"
RED = "UV_VISOR_RED"
YELLOW = "UV_VISOR_YELLOW"

UV_PRODUCTS = [AMBER, MAGENTA, ORANGE, RED, YELLOW]


LIMITS = {
    AMBER: 200,
    MAGENTA: 200,
    ORANGE: 200,
    RED: 200,
    YELLOW: 200,
}


CFG = {
    # AMBER was the big loser, so make it very conservative.
    AMBER: {
        "min_spread": 12,
        "edge": 5.0,
        "base_qty": 2,
        "inv_skew": 28.0,
        "micro_strength": 1.5,
        "signal_threshold": 3.0,
        "mode": "mr",
    },

    MAGENTA: {
        "min_spread": 15,
        "edge": 5.0,
        "base_qty": 4,
        "inv_skew": 24.0,
        "micro_strength": 1.6,
        "signal_threshold": 2.5,
        "mode": "mr",
    },

    ORANGE: {
        "min_spread": 14,
        "edge": 5.0,
        "base_qty": 4,
        "inv_skew": 24.0,
        "micro_strength": 1.6,
        "signal_threshold": 2.5,
        "mode": "neutral",
    },

    RED: {
        "min_spread": 15,
        "edge": 5.0,
        "base_qty": 4,
        "inv_skew": 24.0,
        "micro_strength": 1.6,
        "signal_threshold": 2.5,
        "mode": "trend",
    },

    YELLOW: {
        "min_spread": 15,
        "edge": 5.0,
        "base_qty": 3,
        "inv_skew": 28.0,
        "micro_strength": 1.6,
        "signal_threshold": 2.75,
        "mode": "mr",
    },
}


# EMA parameters
EMA_FAST_ALPHA = 2.0 / (30.0 + 1.0)
EMA_SLOW_ALPHA = 2.0 / (200.0 + 1.0)

# Residual stats
RES_ALPHA = 2.0 / (1000.0 + 1.0)
RES_MIN_COUNT = 200
RES_MIN_SIGMA = 30.0
RES_Z_CLIP = 3.0

# Residual skews are now thresholded, not always-on.
RES_Z_ON = 1.75
RES_STRENGTH_OMA = 3.5
RES_STRENGTH_MRY = 3.5

# Stronger residual inventory cleaner
RES_INV_STRENGTH = 20.0

# Emergency inventory cleaner
EMERGENCY_POS_FRAC = 0.65


class Trader:
    def run(self, state: TradingState):
        data = self._decode(state.traderData)
        self._ensure(data)

        orders: Dict[str, List[Order]] = {}

        books = {}
        mids = {}
        micro_deltas = {}

        # ---------------------------------------------------------------------
        # 1. Read books and update EMAs
        # ---------------------------------------------------------------------
        for product in UV_PRODUCTS:
            od = state.order_depths.get(product)
            book = self._book(od)

            if book is None:
                continue

            books[product] = book
            mids[product] = book["mid"]
            micro_deltas[product] = book["micro"] - book["mid"]

            self._update_ema(data, product, book["mid"])

        # ---------------------------------------------------------------------
        # 2. Build directional scores
        #
        # score > 0 => prefer buying / quote bid
        # score < 0 => prefer selling / quote ask
        # ---------------------------------------------------------------------
        scores = {p: 0.0 for p in books}

        for product in books:
            book = books[product]
            cfg = CFG[product]

            mid = book["mid"]
            fast = data["ema_fast"].get(product, mid)
            slow = data["ema_slow"].get(product, mid)

            # Microprice signal.
            scores[product] += cfg["micro_strength"] * micro_deltas[product]

            # Product-specific momentum behaviour.
            #
            # For MAGENTA/YELLOW/AMBER: recent extension tends to mean-revert.
            # If fast > slow, product is locally rich => sell bias.
            #
            # For RED: momentum behaved more trend-following.
            #
            # For ORANGE: weak/unstable, so mostly residual + micro.
            mom = fast - slow

            if cfg["mode"] == "mr":
                scores[product] += -0.025 * mom
            elif cfg["mode"] == "trend":
                scores[product] += 0.020 * mom

            # Inventory signal.
            # If long, score goes down => prefer ask.
            # If short, score goes up => prefer bid.
            pos = state.position.get(product, 0)
            limit = LIMITS[product]
            scores[product] += -cfg["inv_skew"] * (pos / limit)

        # ---------------------------------------------------------------------
        # 3. Residual overlay 1:
        #       r_OMA = 2O - M - A
        #
        # If high: ORANGE rich, MAGENTA/AMBER cheap.
        # ---------------------------------------------------------------------
        if all(p in mids for p in [ORANGE, MAGENTA, AMBER]):
            r_oma = 2.0 * mids[ORANGE] - mids[MAGENTA] - mids[AMBER]
            z_oma = self._update_residual(data, "OMA", r_oma)

            if abs(z_oma) >= RES_Z_ON:
                s = RES_STRENGTH_OMA * z_oma

                scores[ORANGE] -= s
                scores[MAGENTA] += 0.5 * s
                scores[AMBER] += 0.5 * s

            # Residual inventory cleaner.
            q_oma = (
                2.0 * state.position.get(ORANGE, 0)
                - state.position.get(MAGENTA, 0)
                - state.position.get(AMBER, 0)
            )

            inv = RES_INV_STRENGTH * q_oma / (4.0 * LIMITS[ORANGE])

            scores[ORANGE] -= inv
            scores[MAGENTA] += 0.5 * inv
            scores[AMBER] += 0.5 * inv

        # ---------------------------------------------------------------------
        # 4. Residual overlay 2:
        #       r_MRY = 2M - R - Y
        #
        # If high: MAGENTA rich, RED/YELLOW cheap.
        # ---------------------------------------------------------------------
        if all(p in mids for p in [MAGENTA, RED, YELLOW]):
            r_mry = 2.0 * mids[MAGENTA] - mids[RED] - mids[YELLOW]
            z_mry = self._update_residual(data, "MRY", r_mry)

            if abs(z_mry) >= RES_Z_ON:
                s = RES_STRENGTH_MRY * z_mry

                scores[MAGENTA] -= s
                scores[RED] += 0.5 * s
                scores[YELLOW] += 0.5 * s

            # Residual inventory cleaner.
            q_mry = (
                2.0 * state.position.get(MAGENTA, 0)
                - state.position.get(RED, 0)
                - state.position.get(YELLOW, 0)
            )

            inv = RES_INV_STRENGTH * q_mry / (4.0 * LIMITS[MAGENTA])

            scores[MAGENTA] -= inv
            scores[RED] += 0.5 * inv
            scores[YELLOW] += 0.5 * inv

        # ---------------------------------------------------------------------
        # 5. Make selective quotes
        # ---------------------------------------------------------------------
        for product, book in books.items():
            os = self._quote_product(
                product=product,
                book=book,
                score=scores[product],
                position=state.position.get(product, 0),
            )

            if os:
                orders[product] = os

        trader_data = json.dumps(data, separators=(",", ":"))
        conversions = 0

        return orders, conversions, trader_data

    # =========================================================================
    # Quoting
    # =========================================================================

    def _quote_product(
        self,
        product: str,
        book: Dict[str, float],
        score: float,
        position: int,
    ) -> List[Order]:

        cfg = CFG[product]
        limit = LIMITS[product]

        orders: List[Order] = []

        best_bid = int(book["best_bid"])
        best_ask = int(book["best_ask"])
        spread = int(book["spread"])
        mid = float(book["mid"])

        if spread < cfg["min_spread"]:
            return orders

        buy_room = limit - position
        sell_room = limit + position

        if buy_room <= 0 and sell_room <= 0:
            return orders

        threshold = cfg["signal_threshold"]
        edge = cfg["edge"]
        base_qty = cfg["base_qty"]

        # Directional fair.
        fair = mid + score

        bid_px = best_bid + 1
        ask_px = best_ask - 1

        # Safety: never cross/lock.
        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)

        pos_frac = position / limit

        emergency_sell = pos_frac > EMERGENCY_POS_FRAC
        emergency_buy = pos_frac < -EMERGENCY_POS_FRAC

        want_buy = score >= threshold
        want_sell = score <= -threshold

        # Emergency inventory cleaning overrides normal selectivity.
        if emergency_buy:
            want_buy = True

        if emergency_sell:
            want_sell = True

        # Do not quote both sides unless inventory is extreme.
        # This avoids the old adverse-selection problem.
        if want_buy and want_sell:
            if position > 0:
                want_buy = False
            elif position < 0:
                want_sell = False
            else:
                want_buy = False
                want_sell = False

        # Size shrinks as inventory gets large in that direction.
        buy_scale = max(0.25, min(1.00, buy_room / limit))
        sell_scale = max(0.25, min(1.00, sell_room / limit))

        buy_qty = min(buy_room, max(1, int(round(base_qty * buy_scale))))
        sell_qty = min(sell_room, max(1, int(round(base_qty * sell_scale))))

        # Buy only if our bid is still cheap versus fair.
        if want_buy and buy_room > 0:
            if fair - bid_px >= edge or emergency_buy:
                orders.append(Order(product, int(bid_px), int(buy_qty)))

        # Sell only if our ask is still rich versus fair.
        if want_sell and sell_room > 0:
            if ask_px - fair >= edge or emergency_sell:
                orders.append(Order(product, int(ask_px), -int(sell_qty)))

        return orders

    # =========================================================================
    # State helpers
    # =========================================================================

    def _decode(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return {}

        try:
            data = json.loads(trader_data)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        return {}

    def _ensure(self, data: Dict[str, Any]) -> None:
        if "ema_fast" not in data or not isinstance(data["ema_fast"], dict):
            data["ema_fast"] = {}

        if "ema_slow" not in data or not isinstance(data["ema_slow"], dict):
            data["ema_slow"] = {}

        if "residuals" not in data or not isinstance(data["residuals"], dict):
            data["residuals"] = {}

    def _update_ema(self, data: Dict[str, Any], product: str, mid: float) -> None:
        if product not in data["ema_fast"]:
            data["ema_fast"][product] = mid
            data["ema_slow"][product] = mid
            return

        old_fast = float(data["ema_fast"][product])
        old_slow = float(data["ema_slow"][product])

        data["ema_fast"][product] = old_fast + EMA_FAST_ALPHA * (mid - old_fast)
        data["ema_slow"][product] = old_slow + EMA_SLOW_ALPHA * (mid - old_slow)

    def _update_residual(self, data: Dict[str, Any], name: str, value: float) -> float:
        stats = data["residuals"].get(name)

        if stats is None:
            data["residuals"][name] = {
                "mean": value,
                "var": RES_MIN_SIGMA * RES_MIN_SIGMA,
                "count": 1,
            }
            return 0.0

        mean = float(stats.get("mean", value))
        var = float(stats.get("var", RES_MIN_SIGMA * RES_MIN_SIGMA))
        count = int(stats.get("count", 1))

        sigma = max(RES_MIN_SIGMA, math.sqrt(max(var, 1.0)))

        if count >= RES_MIN_COUNT:
            z = (value - mean) / sigma
        else:
            z = 0.0

        z = max(-RES_Z_CLIP, min(RES_Z_CLIP, z))

        delta = value - mean
        new_mean = mean + RES_ALPHA * delta
        new_var = (1.0 - RES_ALPHA) * (var + RES_ALPHA * delta * delta)

        stats["mean"] = new_mean
        stats["var"] = max(new_var, 1.0)
        stats["count"] = count + 1

        data["residuals"][name] = stats

        return z

    def _book(self, od: OrderDepth):
        if od is None or not od.buy_orders or not od.sell_orders:
            return None

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        bid_qty = od.buy_orders[best_bid]
        ask_qty = -od.sell_orders[best_ask]

        # Some backtesters store ask volumes as positive in CSV but negative in state.
        if ask_qty <= 0:
            ask_qty = abs(od.sell_orders[best_ask])

        if bid_qty <= 0 or ask_qty <= 0:
            return None

        if best_ask <= best_bid:
            return None

        mid = 0.5 * (best_bid + best_ask)
        spread = best_ask - best_bid

        micro = (best_ask * bid_qty + best_bid * ask_qty) / (bid_qty + ask_qty)

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "mid": mid,
            "spread": spread,
            "micro": micro,
        }