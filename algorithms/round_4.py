"""
IMC PROSPERITY 4 — R3 core with fixed references + EMA-blended refs
=====================================================================

Bot overlay:
  - Uses Mark 14 / Mark 38 globally.
      Mark 14 = good/informed/passive side -> follow.
      Mark 38 = bad/toxic/aggressive side -> fade.

  - Adds VELVETFRUIT_EXTRACT-specific Mark 01/14 vs Mark 55 dynamic.
      Mark 01 / Mark 14 = good side.
      Mark 55 = bad aggressive side.

  - Bot signal is instantaneous only.
  - No persistent bot_signal in traderData.
  - No bot target-position system.
  - Bot layer posts passive orders instead of crossing the spread.
  - Bot layer does not override or flatten the main mean-reversion logic.
"""

from datamodel import Order, OrderDepth, TradingState, ProsperityEncoder
from typing import Dict, List, Tuple, Any
import json


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZER LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[str, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = max(0, (self.max_log_length - base_length) // 3)

        print(
            self.to_json(
                [
                    self.compress_state(
                        state,
                        self.truncate(state.traderData, max_item_length)
                        if getattr(state, "traderData", "")
                        else "",
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict) -> list:
        compressed = []

        for listing in listings.values():
            try:
                compressed.append(
                    [
                        listing.symbol,
                        listing.product,
                        listing.denomination,
                    ]
                )
            except AttributeError:
                compressed.append(
                    [
                        listing["symbol"],
                        listing["product"],
                        listing["denomination"],
                    ]
                )

        return compressed

    def compress_order_depths(self, order_depths: Dict[str, OrderDepth]) -> dict:
        compressed = {}

        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders,
                order_depth.sell_orders,
            ]

        return compressed

    def compress_trades(self, trades: dict) -> list:
        compressed = []

        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations) -> list:
        if observations is None:
            return [{}, {}]

        if not hasattr(observations, "plainValueObservations"):
            return observations

        conversion_observations = {}

        if hasattr(observations, "conversionObservations"):
            for product, observation in observations.conversionObservations.items():
                conversion_observations[product] = [
                    getattr(observation, "bidPrice", None),
                    getattr(observation, "askPrice", None),
                    getattr(observation, "transportFees", None),
                    getattr(observation, "exportTariff", None),
                    getattr(observation, "importTariff", None),
                    getattr(observation, "sugarPrice", None),
                    getattr(observation, "sunlightIndex", None),
                ]

        return [
            observations.plainValueObservations,
            conversion_observations,
        ]

    def compress_orders(self, orders: Dict[str, List[Order]]) -> list:
        compressed = []

        for arr in orders.values():
            for order in arr:
                compressed.append(
                    [
                        order.symbol,
                        order.price,
                        order.quantity,
                    ]
                )

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if value is None:
            return ""

        value = str(value)

        if len(value) <= max_length:
            return value

        if max_length <= 3:
            return value[:max_length]

        return value[: max_length - 3] + "..."


logger = Logger()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TOGGLES
# ─────────────────────────────────────────────────────────────────────────────

MID_MR = True

PERSONALISED_Z = True

ENTRY_Z = 0.6
CLOSE_Z = 0.0

USE_L2 = False


# ─────────────────────────────────────────────────────────────────────────────
# EMA-BLENDED REFERENCE TOGGLES
# ─────────────────────────────────────────────────────────────────────────────

EMA_BLEND = True

EMA_WEIGHT_UNIVERSAL = False

EMA_WEIGHT = 0.2
EMA_WINDOW = 500

CAP = True
SHIFT_CAP_STD = 1.5

EMA_WEIGHT_BY_PRODUCT = {
    "VELVETFRUIT_EXTRACT": 0.1,

    # Deep ITM: should adapt with the underlying
    "VEV_4000": 0.1,
    "VEV_4500": 0.1,

    # ATM-ish: keep mostly fixed-anchor
    "VEV_5000": 0.20,
    "VEV_5100": 0.30,
    "VEV_5200": 0.30,

    # OTM: more maturity/day-level drift, so more EMA
    "VEV_5300": 0.5,
    "VEV_5400": 0.5,
    "VEV_5500": 0.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# HYDROGEL_PACK
# ─────────────────────────────────────────────────────────────────────────────
HYD = "HYDROGEL_PACK"
HYD_POS_LIMIT = 200
HYD_FAIR = 10006
HYD_SIGMA = 30
HYD_LADDER = [(19.0 / 30.0, 1.0)]
HYD_ENTRY_Z = 19.0 / 30.0
HYD_EXIT_Z = -19.0 / 30.0
HYD_EMA_ALPHA = 0.005
HYD_EMA_W = 0.15



# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT CONFIG (d1+d2+d3)/3 , std from day 3
# ─────────────────────────────────────────────────────────────────────────────


CFG = {
    "VELVETFRUIT_EXTRACT": {
        "fair": 5245.53, "mid_std": 18.60,
        "bid_ref": 5243.04, "bid_std": 18.59,
        "ask_ref": 5248.02, "ask_std": 18.61,
        "limit": 200, "cap": 300, "enabled": True,
    },

    "VEV_4000": {
        "fair": 1245.53, "mid_std": 18.63,
        "bid_ref": 1235.18, "bid_std": 18.50,
        "ask_ref": 1255.89, "ask_std": 18.80,
        "limit": 300, "cap": 300, "enabled": True,
    },
    "VEV_4500": {
        "fair": 745.53, "mid_std": 18.62,
        "bid_ref": 737.66, "bid_std": 18.44,
        "ask_ref": 753.41, "ask_std": 18.82,
        "limit": 300, "cap": 300, "enabled": True,
    },

    "VEV_5000": {
        "fair": 248.76, "mid_std": 17.86,
        "bid_ref": 245.81, "bid_std": 17.68,
        "ask_ref": 251.72, "ask_std": 18.04,
        "limit": 300, "cap": 300, "enabled": True,
    },
    "VEV_5100": {
        "fair": 158.22, "mid_std": 15.74,
        "bid_ref": 156.16, "bid_std": 15.59,
        "ask_ref": 160.28, "ask_std": 15.90,
        "limit": 300, "cap": 300, "enabled": True,
    },
    "VEV_5200": {
        "fair": 86.20, "mid_std": 11.78,
        "bid_ref": 84.85, "bid_std": 11.66,
        "ask_ref": 87.55, "ask_std": 11.90,
        "limit": 300, "cap": 300, "enabled": True,
    },
    "VEV_5300": {
        "fair": 38.92, "mid_std": 6.75,
        "bid_ref": 37.96, "bid_std": 6.68,
        "ask_ref": 39.88, "ask_std": 6.84,
        "limit": 300, "cap": 300, "enabled": True,
    },
    "VEV_5400": {
        "fair": 11.60, "mid_std": 2.83,
        "bid_ref": 10.96, "bid_std": 2.80,
        "ask_ref": 12.24, "ask_std": 2.88,
        "limit": 300, "cap": 300, "enabled": True,
    },
    "VEV_5500": {
        "fair": 4.10, "mid_std": 1.03,
        "bid_ref": 3.55, "bid_std": 1.02,
        "ask_ref": 4.64, "ask_std": 1.06,
        "limit": 300, "cap": 300, "enabled": True,
    },

    "VEV_6000": {
        "fair": 0.50, "mid_std": 0.00,
        "bid_ref": 0.00, "bid_std": 0.00,
        "ask_ref": 1.00, "ask_std": 0.00,
        "limit": 300, "cap": 300, "enabled": False,
    },
    "VEV_6500": {
        "fair": 0.50, "mid_std": 0.00,
        "bid_ref": 0.00, "bid_std": 0.00,
        "ask_ref": 1.00, "ask_std": 0.00,
        "limit": 300, "cap": 300, "enabled": False,
    },
}



PERSONALISED_Z_VALUES = {
    "VELVETFRUIT_EXTRACT": 0.8,

    "VEV_4000": 0.8,
    "VEV_4500": 0.8,

    "VEV_5000": 0.6,
    "VEV_5100": 0.8,
    "VEV_5200": 0.6,

    "VEV_5300": 0.8,
    "VEV_5400": 0.8,

    "VEV_5500": 0.8,
}

TRADABLE_PRODUCTS = list(CFG.keys())


# ─────────────────────────────────────────────────────────────────────────────
# BOT SIGNAL — passive implementation
# ─────────────────────────────────────────────────────────────────────────────

BOT_SIGNAL_ENABLED = True

# Generic global bot signal.
GOOD_BOT = "Mark 14"
BAD_BOT = "Mark 38"

# Extra pair-specific VELVET signal.
VELVET_PAIR_PRODUCT = "VELVETFRUIT_EXTRACT"
VELVET_GOOD_BOTS = {"Mark 01", "Mark 14"}
VELVET_BAD_BOT = "Mark 55"

# Edge is how far we are willing to quote away from the observed bot trade price.
BOT_EDGE_BY_PRODUCT = {
    "HYDROGEL_PACK": 5.0,
    "VELVETFRUIT_EXTRACT": 1.5,
    "VEV_4000": 6.0,

    "VEV_5200": 0.5,
    "VEV_5300": 0.4,
    "VEV_5500": 0.25,
}

BOT_MAX_QTY_BY_PRODUCT = {
    "HYDROGEL_PACK": 25,
    "VELVETFRUIT_EXTRACT": 30,
    "VEV_4000": 20,

    "VEV_5200": 10,
    "VEV_5300": 10,
    "VEV_5500": 8,
}

BOT_Z_GUARD_BY_PRODUCT = {
    "HYDROGEL_PACK": 0.45,
    "VELVETFRUIT_EXTRACT": 0.35,
    "VEV_4000": 0.45,

    "VEV_5200": 0.30,
    "VEV_5300": 0.30,
    "VEV_5500": 0.25,
}

BOT_QTY_MULT = 1.0
BOT_IMPROVE_TICKS = 1


class Trader:

    # ─────────────────────────────────────────────────────────────────────
    # TraderData / EMA state
    # ─────────────────────────────────────────────────────────────────────
    def load_data(self, trader_data: str) -> Dict[str, Any]:
        if trader_data:
            try:
                data = json.loads(trader_data)
                if isinstance(data, dict):
                    for key in ["ema_bid", "ema_ask", "ema_mid", "n"]:
                        if key not in data or not isinstance(data[key], dict):
                            data[key] = {}
                    return data
            except Exception:
                pass

        return {
            "ema_bid": {},
            "ema_ask": {},
            "ema_mid": {},
            "n": {},
        }

    def encode_data(self, data: Dict[str, Any]) -> str:
        try:
            return json.dumps(data)
        except Exception:
            return ""

    # ─────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK helpers
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _update_ema(prev, value, alpha):
        if prev is None:
            return float(value)
        return alpha * float(value) + (1.0 - alpha) * float(prev)

    def _ladder_target(self, abs_z, sign):
        frac = 0.0
        for threshold, f in HYD_LADDER:
            if abs_z >= threshold:
                frac = f
        return int(sign * frac * HYD_POS_LIMIT)

    def _hyd_trade(
        self,
        state: TradingState,
        data: Dict[str, Any],
        bot_signal: Dict[str, float],
    ) -> List[Order]:
        od = state.order_depths[HYD]
        pos = state.position.get(HYD, 0)
        buys = od.buy_orders
        sells = od.sell_orders
        orders: List[Order] = []

        if not buys or not sells:
            return orders

        best_bid = max(buys)
        best_ask = min(sells)
        bid_vol = buys[best_bid]
        ask_vol = abs(sells[best_ask])
        mid = (best_bid + best_ask) / 2.0

        ema_prev = data.get("hyd_ema")
        ema_now = self._update_ema(ema_prev, mid, HYD_EMA_ALPHA)
        data["hyd_ema"] = ema_now

        fair = HYD_EMA_W * ema_now + (1.0 - HYD_EMA_W) * HYD_FAIR

        z_mid = (mid - fair) / HYD_SIGMA

        max_buy = HYD_POS_LIMIT - pos
        max_sell = HYD_POS_LIMIT + pos

        target = pos

        if pos == 0:
            if z_mid <= -HYD_ENTRY_Z:
                target = self._ladder_target(abs(z_mid), +1)
            elif z_mid >= HYD_ENTRY_Z:
                target = self._ladder_target(abs(z_mid), -1)

        elif pos > 0:
            if z_mid >= -HYD_EXIT_Z:
                target = 0
            elif z_mid <= -HYD_ENTRY_Z:
                lt = self._ladder_target(abs(z_mid), +1)
                target = max(pos, lt)

        else:
            if z_mid <= HYD_EXIT_Z:
                target = 0
            elif z_mid >= HYD_ENTRY_Z:
                lt = self._ladder_target(abs(z_mid), -1)
                target = min(pos, lt)

        delta = target - pos

        if delta > 0:
            qty = min(delta, max_buy, ask_vol)
            if qty > 0:
                orders.append(Order(HYD, best_ask, qty))

        elif delta < 0:
            qty = min(-delta, max_sell, bid_vol)
            if qty > 0:
                orders.append(Order(HYD, best_bid, -qty))

        # Passive bot overlay for HYDROGEL_PACK.
        if BOT_SIGNAL_ENABLED and bot_signal and HYD in BOT_EDGE_BY_PRODUCT:
            direction = int(bot_signal["dir"])
            bot_price = float(bot_signal["price"])
            signal_qty = int(bot_signal["qty"])
            spread = best_ask - best_bid

            if direction != 0 and signal_qty > 0 and spread > 1:
                max_qty = BOT_MAX_QTY_BY_PRODUCT.get(HYD, 0)
                qty_wanted = min(int(signal_qty * BOT_QTY_MULT), max_qty)

                edge = BOT_EDGE_BY_PRODUCT.get(HYD, 0.0)
                z_guard = BOT_Z_GUARD_BY_PRODUCT.get(HYD, 0.45)
                position_after = pos + sum(order.quantity for order in orders)

                improve = BOT_IMPROVE_TICKS

                if direction > 0 and position_after < HYD_POS_LIMIT:
                    px = min(best_bid + improve, best_ask - 1)
                    z_px = (px - fair) / HYD_SIGMA

                    if px <= bot_price + edge and z_px <= z_guard:
                        capacity = HYD_POS_LIMIT - position_after

                        # If currently short after MR orders, only allow bot to reduce/flatten.
                        if position_after < 0:
                            capacity = min(capacity, -position_after)

                        qty = min(qty_wanted, capacity)

                        if qty > 0:
                            orders.append(Order(HYD, px, qty))

                elif direction < 0 and position_after > -HYD_POS_LIMIT:
                    px = max(best_ask - improve, best_bid + 1)
                    z_px = (px - fair) / HYD_SIGMA

                    if px >= bot_price - edge and z_px >= -z_guard:
                        capacity = HYD_POS_LIMIT + position_after

                        # If currently long after MR orders, only allow bot to reduce/flatten.
                        if position_after > 0:
                            capacity = min(capacity, position_after)

                        qty = min(qty_wanted, capacity)

                        if qty > 0:
                            orders.append(Order(HYD, px, -qty))

        return orders

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────
    def entry_z(self, product: str) -> float:
        if PERSONALISED_Z:
            return PERSONALISED_Z_VALUES.get(product, ENTRY_Z)
        return ENTRY_Z

    def ema_weight(self, product: str) -> float:
        if EMA_WEIGHT_UNIVERSAL:
            return EMA_WEIGHT
        return EMA_WEIGHT_BY_PRODUCT.get(product, EMA_WEIGHT)

    def capped_shift(self, shift: float, std: float) -> float:
        if not CAP:
            return shift

        cap = SHIFT_CAP_STD * std
        return max(-cap, min(cap, shift))

    def blended_ref(
        self,
        product: str,
        fixed_ref: float,
        fixed_std: float,
        ema_value: float,
    ) -> float:
        if not EMA_BLEND:
            return fixed_ref

        if ema_value is None:
            return fixed_ref

        shift = ema_value - fixed_ref
        shift = self.capped_shift(shift, fixed_std)

        return fixed_ref + self.ema_weight(product) * shift

    def best_bid_ask_mid(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None, None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        return best_bid, best_ask, mid

    def update_ema_after_signal(
        self,
        product: str,
        best_bid: float,
        best_ask: float,
        mid: float,
        data: Dict[str, Any],
    ):
        alpha = 2.0 / (EMA_WINDOW + 1.0)

        n_old = int(data["n"].get(product, 0))

        if n_old <= 0 or product not in data["ema_mid"]:
            data["ema_bid"][product] = float(best_bid)
            data["ema_ask"][product] = float(best_ask)
            data["ema_mid"][product] = float(mid)
            data["n"][product] = 1
            return

        old_bid = float(data["ema_bid"].get(product, best_bid))
        old_ask = float(data["ema_ask"].get(product, best_ask))
        old_mid = float(data["ema_mid"].get(product, mid))

        data["ema_bid"][product] = alpha * float(best_bid) + (1.0 - alpha) * old_bid
        data["ema_ask"][product] = alpha * float(best_ask) + (1.0 - alpha) * old_ask
        data["ema_mid"][product] = alpha * float(mid) + (1.0 - alpha) * old_mid
        data["n"][product] = n_old + 1

    def refs_and_stds(self, product: str, data: Dict[str, Any]) -> Tuple[float, float, float, float]:
        cfg = CFG[product]

        ema_bid = data["ema_bid"].get(product)
        ema_ask = data["ema_ask"].get(product)
        ema_mid = data["ema_mid"].get(product)

        fair_ref = self.blended_ref(product, cfg["fair"], cfg["mid_std"], ema_mid)
        ask_ref = self.blended_ref(product, cfg["ask_ref"], cfg["ask_std"], ema_ask)
        bid_ref = self.blended_ref(product, cfg["bid_ref"], cfg["bid_std"], ema_bid)

        if MID_MR:
            buy_ref = fair_ref
            sell_ref = fair_ref
            buy_std = cfg["mid_std"]
            sell_std = cfg["mid_std"]
        else:
            buy_ref = ask_ref
            sell_ref = bid_ref
            buy_std = cfg["ask_std"]
            sell_std = cfg["bid_std"]

        return buy_ref, buy_std, sell_ref, sell_std

    def z_buy(self, product: str, price: float, data: Dict[str, Any]) -> float:
        buy_ref, buy_std, _, _ = self.refs_and_stds(product, data)

        if buy_std <= 1e-9:
            return 999.0

        return (price - buy_ref) / buy_std

    def z_sell(self, product: str, price: float, data: Dict[str, Any]) -> float:
        _, _, sell_ref, sell_std = self.refs_and_stds(product, data)

        if sell_std <= 1e-9:
            return -999.0

        return (price - sell_ref) / sell_std

    # ─────────────────────────────────────────────────────────────────────
    # Bot helpers — instantaneous signal, passive execution
    # ─────────────────────────────────────────────────────────────────────
    def current_bot_signals(self, state: TradingState) -> Dict[str, Dict[str, float]]:
        """
        Instantaneous bot signal.

        Direction:
            +1 = bullish / buy
            -1 = bearish / sell

        Generic:
            Mark 14 = follow.
            Mark 38 = fade.

        VELVET-specific:
            Mark 01 / Mark 14 = good.
            Mark 55 = bad.
        """
        signals: Dict[str, Dict[str, float]] = {}

        if not BOT_SIGNAL_ENABLED:
            return signals

        raw: Dict[str, Dict[str, float]] = {}

        for product, trades in state.market_trades.items():
            if product not in BOT_EDGE_BY_PRODUCT:
                continue

            for trade in trades:
                buyer = getattr(trade, "buyer", "")
                seller = getattr(trade, "seller", "")
                price = float(getattr(trade, "price", 0))
                qty = abs(int(getattr(trade, "quantity", 0)))

                if qty <= 0:
                    continue

                direction = 0

                # ------------------------------------------------------------
                # Pair-specific VELVET signal:
                # Mark 01 / Mark 14 vs Mark 55.
                #
                # Count this first so Mark 14 vs Mark 55 is not double-counted
                # by the generic Mark 14 logic below.
                # ------------------------------------------------------------
                if product == VELVET_PAIR_PRODUCT:
                    if buyer in VELVET_GOOD_BOTS and seller == VELVET_BAD_BOT:
                        # Good bot buys from bad bot selling.
                        direction = 1

                    elif seller in VELVET_GOOD_BOTS and buyer == VELVET_BAD_BOT:
                        # Bad bot buys from good bot selling.
                        direction = -1

                # ------------------------------------------------------------
                # Generic Mark 14 / Mark 38 signal.
                # ------------------------------------------------------------
                if direction == 0:
                    if buyer == GOOD_BOT:
                        direction = 1
                    elif seller == GOOD_BOT:
                        direction = -1
                    elif buyer == BAD_BOT:
                        direction = -1
                    elif seller == BAD_BOT:
                        direction = 1

                if direction == 0:
                    continue

                if product not in raw:
                    raw[product] = {
                        "signed_qty": 0.0,
                        "buy_qty": 0.0,
                        "buy_px_qty": 0.0,
                        "sell_qty": 0.0,
                        "sell_px_qty": 0.0,
                    }

                raw[product]["signed_qty"] += direction * qty

                if direction > 0:
                    raw[product]["buy_qty"] += qty
                    raw[product]["buy_px_qty"] += price * qty
                else:
                    raw[product]["sell_qty"] += qty
                    raw[product]["sell_px_qty"] += price * qty

        for product, r in raw.items():
            signed_qty = r["signed_qty"]

            if signed_qty > 0 and r["buy_qty"] > 0:
                signals[product] = {
                    "dir": 1,
                    "qty": abs(signed_qty),
                    "price": r["buy_px_qty"] / r["buy_qty"],
                }

            elif signed_qty < 0 and r["sell_qty"] > 0:
                signals[product] = {
                    "dir": -1,
                    "qty": abs(signed_qty),
                    "price": r["sell_px_qty"] / r["sell_qty"],
                }

        return signals

    def add_bot_passive_order(
        self,
        product: str,
        order_depth: OrderDepth,
        position_after: int,
        orders: List[Order],
        data: Dict[str, Any],
        bot_signal: Dict[str, float],
    ) -> List[Order]:
        """
        Passive bot implementation.

        The bot edge is measured from the bot trade price, usually a passive
        bid/ask trade price. Therefore we do not copy the bot by crossing the
        spread. We post passively instead.

        Bullish signal:
            post/join bid.

        Bearish signal:
            post/join ask.
        """
        if not BOT_SIGNAL_ENABLED or not bot_signal:
            return orders

        if product not in BOT_EDGE_BY_PRODUCT:
            return orders

        cfg = CFG[product]
        limit = cfg["limit"]

        direction = int(bot_signal["dir"])
        bot_price = float(bot_signal["price"])
        signal_qty = int(bot_signal["qty"])

        if direction == 0 or signal_qty <= 0:
            return orders

        best_bid, best_ask, _ = self.best_bid_ask_mid(order_depth)

        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid

        if spread <= 1:
            return orders

        max_qty = BOT_MAX_QTY_BY_PRODUCT.get(product, 0)
        qty_wanted = min(int(signal_qty * BOT_QTY_MULT), max_qty)

        if qty_wanted <= 0:
            return orders

        edge = BOT_EDGE_BY_PRODUCT.get(product, 0.0)
        z_guard = BOT_Z_GUARD_BY_PRODUCT.get(product, 0.35)
        improve = BOT_IMPROVE_TICKS

        # Bullish signal: post passive buy.
        if direction > 0:
            if position_after >= limit:
                return orders

            px = min(best_bid + improve, best_ask - 1)

            # Do not chase far above the bot trade price.
            if px > bot_price + edge:
                return orders

            # Do not buy if model says price is too high.
            if self.z_buy(product, px, data) > z_guard:
                return orders

            capacity = limit - position_after

            # If currently short after MR orders, only allow bot to reduce/flatten.
            if position_after < 0:
                capacity = min(capacity, -position_after)

            qty = min(qty_wanted, capacity)

            if qty > 0:
                orders.append(Order(product, px, qty))

        # Bearish signal: post passive sell.
        else:
            if position_after <= -limit:
                return orders

            px = max(best_ask - improve, best_bid + 1)

            # Do not chase far below the bot trade price.
            if px < bot_price - edge:
                return orders

            # Do not sell if model says price is too low.
            if self.z_sell(product, px, data) < -z_guard:
                return orders

            capacity = limit + position_after

            # If currently long after MR orders, only allow bot to reduce/flatten.
            if position_after > 0:
                capacity = min(capacity, position_after)

            qty = min(qty_wanted, capacity)

            if qty > 0:
                orders.append(Order(product, px, -qty))

        return orders

    # ─────────────────────────────────────────────────────────────────────
    # Core z-score MR logic
    # ─────────────────────────────────────────────────────────────────────
    def zscore_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        data: Dict[str, Any],
        bot_signal: Dict[str, float],
    ) -> List[Order]:
        cfg = CFG[product]

        best_bid, best_ask, mid = self.best_bid_ask_mid(order_depth)

        if best_bid is None or best_ask is None or mid is None:
            return []

        # Disabled products still update EMA state, but never trade.
        if not cfg.get("enabled", True):
            self.update_ema_after_signal(product, best_bid, best_ask, mid, data)
            return []

        entry = self.entry_z(product)
        limit = cfg["limit"]

        best_ask_z = self.z_buy(product, best_ask, data)
        best_bid_z = self.z_sell(product, best_bid, data)

        target = position

        if best_ask_z <= -entry:
            target = limit
        elif best_bid_z >= entry:
            target = -limit
        elif position > 0 and best_bid_z >= CLOSE_Z:
            target = 0
        elif position < 0 and best_ask_z <= CLOSE_Z:
            target = 0

        delta = target - position

        orders: List[Order] = []

        if delta != 0:
            orders = self.execute_delta(
                product=product,
                order_depth=order_depth,
                position=position,
                delta=delta,
                entry_z=entry,
                opening_target=(target != 0),
                data=data,
            )

        position_after = position + sum(order.quantity for order in orders)

        # Bot layer only adds a small passive order.
        # It does not change the MR target and does not flatten MR positions.
        orders = self.add_bot_passive_order(
            product=product,
            order_depth=order_depth,
            position_after=position_after,
            orders=orders,
            data=data,
            bot_signal=bot_signal,
        )

        self.update_ema_after_signal(product, best_bid, best_ask, mid, data)

        return orders

    # ─────────────────────────────────────────────────────────────────────
    # R3-style taker execution
    # ─────────────────────────────────────────────────────────────────────
    def execute_delta(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        delta: int,
        entry_z: float,
        opening_target: bool,
        data: Dict[str, Any],
    ) -> List[Order]:
        orders: List[Order] = []
        cfg = CFG[product]

        limit = cfg["limit"]
        cap = cfg["cap"]
        pos = position
        traded = 0

        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        asks = sorted(order_depth.sell_orders.keys())

        if opening_target:
            buy_threshold = -entry_z
            sell_threshold = entry_z
        else:
            buy_threshold = CLOSE_Z
            sell_threshold = CLOSE_Z

        if delta > 0:
            qty_needed = min(delta, limit - pos, cap)

            if qty_needed <= 0:
                return orders

            max_levels = 2 if USE_L2 else 1

            for ask in asks[:max_levels]:
                ask_z = self.z_buy(product, ask, data)

                if ask_z > buy_threshold:
                    continue

                available = -order_depth.sell_orders[ask]
                qty = min(available, qty_needed, cap - traded)

                if qty <= 0:
                    continue

                orders.append(Order(product, ask, qty))

                pos += qty
                qty_needed -= qty
                traded += qty

                if qty_needed <= 0 or traded >= cap:
                    break

        elif delta < 0:
            qty_needed = min(-delta, limit + pos, cap)

            if qty_needed <= 0:
                return orders

            max_levels = 2 if USE_L2 else 1

            for bid in bids[:max_levels]:
                bid_z = self.z_sell(product, bid, data)

                if bid_z < sell_threshold:
                    continue

                available = order_depth.buy_orders[bid]
                qty = min(available, qty_needed, cap - traded)

                if qty <= 0:
                    continue

                orders.append(Order(product, bid, -qty))

                pos -= qty
                qty_needed -= qty
                traded += qty

                if qty_needed <= 0 or traded >= cap:
                    break

        return orders

    # ─────────────────────────────────────────────────────────────────────
    # Main
    # ─────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        data = self.load_data(state.traderData)

        # Instantaneous bot signals from current visible trades.
        bot_signals = self.current_bot_signals(state)

        # HYDROGEL_PACK.
        if HYD in state.order_depths:
            hyd_orders = self._hyd_trade(
                state=state,
                data=data,
                bot_signal=bot_signals.get(HYD, {}),
            )
            if hyd_orders:
                result[HYD] = hyd_orders

        for product in TRADABLE_PRODUCTS:
            if product not in state.order_depths:
                continue

            orders = self.zscore_orders(
                product=product,
                order_depth=state.order_depths[product],
                position=state.position.get(product, 0),
                data=data,
                bot_signal=bot_signals.get(product, {}),
            )

            if orders:
                result[product] = orders

        traderData = self.encode_data(data)

        logger.print("pos", state.position, "bots", bot_signals)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData