from datamodel import Order, OrderDepth, TradingState, ProsperityEncoder
from typing import Dict, List, Tuple, Any
import json
import math

try:
    import numpy as np
except Exception:
    np = None


FAST_MODE = True


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer logger
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        if FAST_MODE:
            return
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
# Global caches and histories
# ─────────────────────────────────────────────────────────────────────────────

NP_COEF_CACHE: Dict[str, Any] = {}
NP_MATRIX_CACHE: Dict[Tuple[int, ...], Any] = {}
NP_W_CACHE: Dict[Tuple[int, float], Any] = {}

ROBOT_S1_HISTORY: List[float] = []
ROBOT_S2_HISTORY: List[float] = []
ROBOT_LAST_TIMESTAMP = None

MICROCHIP_EFFECTIVE_K = 14
GALAXY_EFFECTIVE_K = 14
PEBBLES_EFFECTIVE_K = 14
SLEEP_POD_EFFECTIVE_K = 14


# ─────────────────────────────────────────────────────────────────────────────
# Snackpack config
# ─────────────────────────────────────────────────────────────────────────────

SNPK_RB_LIMIT = 10

SNPK_FAIR = {
    "SNACKPACK_VANILLA": 10097,
    "SNACKPACK_RASPBERRY": 10080,
}

SNPK_ENTRY_THRESH_VANILLA = 130
SNPK_ENTRY_THRESH_RASPBERRY = 150
SNPK_EXIT_THRESH_VANILLA = 20
SNPK_EXIT_THRESH_RASPBERRY = 30
SNPK_MM_MAX_QTY = 4

CLUSTER_SNACKPACK_VANILLA = ["SNACKPACK_CHOCOLATE"]
CLUSTER_SNACKPACK_RASPBERRY = ["SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY"]


# ─────────────────────────────────────────────────────────────────────────────
# Fourier model configs
# ─────────────────────────────────────────────────────────────────────────────

SNACKPACK_FOURIER_PRODUCTS = [
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_VANILLA",
]


ROBOT_POSITION_LIMIT = 10
ROBOT_MAX_HISTORY = 1000
ROBOT_MAX_TAKE_SPREAD = 10
ROBOT_ENABLE_PASSIVE_OVERLAY = True

ROBOT_VACUUMING = "ROBOT_VACUUMING"
ROBOT_MOPPING = "ROBOT_MOPPING"
ROBOT_DISHES = "ROBOT_DISHES"
ROBOT_LAUNDRY = "ROBOT_LAUNDRY"
ROBOT_IRONING = "ROBOT_IRONING"


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_trader_data(raw):
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def product_sector(product: str) -> str:
    if product.startswith("GALAXY_SOUNDS"):
        return "GALAXY_SOUNDS"
    if product.startswith("MICROCHIP"):
        return "MICROCHIP"
    if product.startswith("OXYGEN_SHAKE"):
        return "OXYGEN_SHAKE"
    if product.startswith("PANEL"):
        return "PANEL"
    if product.startswith("PEBBLES"):
        return "PEBBLES"
    if product.startswith("ROBOT"):
        return "ROBOT"
    if product.startswith("SLEEP_POD"):
        return "SLEEP_POD"
    if product.startswith("SNACKPACK"):
        return "SNACKPACK"
    if product.startswith("TRANSLATOR"):
        return "TRANSLATOR"
    if product.startswith("UV_VISOR"):
        return "UV_VISOR"
    return product


# ─────────────────────────────────────────────────────────────────────────────
# Snackpack mean reversion
# ─────────────────────────────────────────────────────────────────────────────

def _lev_trade(
    leverage_products: List[str],
    direction: int,
    lev_limit: int,
    state: TradingState,
    result: Dict[str, List[Order]],
) -> None:
    for product in leverage_products:
        od = state.order_depths.get(product)
        if not od:
            continue

        pos = state.position.get(product, 0)

        if direction > 0:
            if not od.sell_orders or pos >= lev_limit:
                continue
            best_ask = min(od.sell_orders)
            qty = min(-od.sell_orders[best_ask], lev_limit - pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_ask, qty))
                logger.print(f"LEV BUY {product} {qty}@{best_ask} pos={pos}")
        else:
            if not od.buy_orders or pos <= -lev_limit:
                continue
            best_bid = max(od.buy_orders)
            qty = min(od.buy_orders[best_bid], lev_limit + pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_bid, -qty))
                logger.print(f"LEV SELL {product} {qty}@{best_bid} pos={pos}")


def snackpack_strategy(
    product: str,
    leverage_products: List[str],
    limit: int,
    lev_limit: int,
    entry_threshold: int,
    exit_threshold: int,
    state: TradingState,
    result: Dict[str, List[Order]],
    td: dict,
    key: str,
) -> None:
    od = state.order_depths.get(product)
    if not (od and od.buy_orders and od.sell_orders):
        return

    pos = state.position.get(product, 0)
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2
    fair = SNPK_FAIR[product]
    dev = mid - fair
    sig = int(td.get(f"{key}_sig", 0))

    if sig == 0 and dev < -entry_threshold:
        sig = 1
    elif sig == 0 and dev > entry_threshold:
        sig = -1
    elif sig == 1 and dev >= -exit_threshold:
        sig = 2
    elif sig == -1 and dev <= exit_threshold:
        sig = -2
    elif sig == 2 and pos <= 0:
        sig = 0
    elif sig == -2 and pos >= 0:
        sig = 0

    td[f"{key}_sig"] = sig
    logger.print(f"{product} dev={dev:.1f} fair={fair} pos={pos} sig={sig}")

    orders: List[Order] = []
    net_qty = 0

    if sig == 1 and pos < limit:
        qty = min(-od.sell_orders.get(best_ask, 0), limit - pos)
        if qty > 0:
            orders.append(Order(product, best_ask, qty))
            net_qty += qty
            logger.print(f"ENTRY BUY {qty}@{best_ask}")
        _lev_trade(leverage_products, -1, lev_limit, state, result)

    elif sig == -1 and pos > -limit:
        qty = min(od.buy_orders.get(best_bid, 0), limit + pos)
        if qty > 0:
            orders.append(Order(product, best_bid, -qty))
            net_qty -= qty
            logger.print(f"ENTRY SELL {qty}@{best_bid}")
        _lev_trade(leverage_products, 1, lev_limit, state, result)

    elif sig == 2 and pos > 0:
        qty = min(od.buy_orders.get(best_bid, 0), pos)
        if qty > 0:
            orders.append(Order(product, best_bid, -qty))
            net_qty -= qty
            logger.print(f"UNWIND SELL {qty}@{best_bid} pos={pos}")
        _lev_trade(leverage_products, 1, lev_limit, state, result)

    elif sig == -2 and pos < 0:
        qty = min(-od.sell_orders.get(best_ask, 0), -pos)
        if qty > 0:
            orders.append(Order(product, best_ask, qty))
            net_qty += qty
            logger.print(f"UNWIND BUY {qty}@{best_ask} pos={pos}")
        _lev_trade(leverage_products, -1, lev_limit, state, result)

    if sig == 0 and spread >= 2 and abs(dev) < entry_threshold:
        buy_cap = limit - pos - max(0, net_qty)
        sell_cap = limit + pos + max(0, -net_qty)
        bid_px = best_bid + 1
        ask_px = best_ask - 1

        if bid_px < ask_px:
            bid_qty = min(buy_cap, SNPK_MM_MAX_QTY)
            ask_qty = min(sell_cap, SNPK_MM_MAX_QTY)

            if bid_qty > 0:
                orders.append(Order(product, bid_px, bid_qty))
                logger.print(f"MM BID {bid_qty}@{bid_px} dev={dev:.0f}")
            if ask_qty > 0:
                orders.append(Order(product, ask_px, -ask_qty))
                logger.print(f"MM ASK {ask_qty}@{ask_px} dev={dev:.0f}")

    for order in orders:
        result.setdefault(product, []).append(order)


# ─────────────────────────────────────────────────────────────────────────────
# Pebbles market making
# ─────────────────────────────────────────────────────────────────────────────

def pebbles_market_make(state: TradingState, result: Dict[str, List[Order]], td: dict) -> None:
    mids = {}

    for product in PEBBLES_PRODUCTS:
        od = state.order_depths.get(product)
        if od and od.buy_orders and od.sell_orders:
            mids[product] = (max(od.buy_orders) + min(od.sell_orders)) / 2

    basket_dev = sum(mids.values()) - PEBBLES_BASKET_FAIR if len(mids) == len(PEBBLES_PRODUCTS) else 0
    basket_skew = 0

    if basket_dev > 2:
        basket_skew = 1
    elif basket_dev < -2:
        basket_skew = -1

    prev_mids = td.get("pebbles_prev_mids", {})
    td["pebbles_prev_mids"] = {p: round(m, 1) for p, m in mids.items()}

    for product, (lim, min_spread, max_qty) in PEBBLES_MM_CFG.items():
        od = state.order_depths.get(product)
        if not (od and od.buy_orders and od.sell_orders):
            continue

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        spread = best_ask - best_bid

        if spread < min_spread:
            continue

        pos = state.position.get(product, 0)

        product_fair = (
            PEBBLES_BASKET_FAIR
            - sum(mids.get(other, 0) for other in PEBBLES_PRODUCTS if other != product)
            if len(mids) == len(PEBBLES_PRODUCTS)
            else (best_bid + best_ask) / 2
        )

        fair_dev = ((best_bid + best_ask) / 2) - product_fair
        fair_skew = basket_skew

        if pos > PEBBLES_SOFT_CAP:
            inv_skew = 1
        elif pos < -PEBBLES_SOFT_CAP:
            inv_skew = -1
        else:
            inv_skew = 0

        mom_skew = 0
        prev_mid = prev_mids.get(product)

        if prev_mid is not None and product in PEBBLES_MOM_STYLE:
            momentum = ((best_bid + best_ask) / 2) - prev_mid
            if abs(momentum) >= PEBBLES_MOM_THRESH:
                mom_skew = -PEBBLES_MOM_STYLE[product] * (1 if momentum > 0 else -1)

        skew = max(-2, min(2, fair_skew + inv_skew + mom_skew))

        max_improve = max(0, (spread - 1) // 2)
        bid_improve = max(0, min(max_improve, PEBBLES_BASE_IMPROVE - skew))
        ask_improve = max(0, min(max_improve, PEBBLES_BASE_IMPROVE + skew))

        bid_px = best_bid + bid_improve
        ask_px = best_ask - ask_improve

        if bid_px >= ask_px:
            continue

        bid_qty = max(0, min(max_qty, lim - pos))
        ask_qty = max(0, min(max_qty, lim + pos))

        if bid_px >= product_fair:
            bid_qty = 0
        if ask_px <= product_fair:
            ask_qty = 0

        if bid_qty > 0:
            result.setdefault(product, []).append(Order(product, bid_px, bid_qty))
            logger.print(
                f"{product} MM BID {bid_qty}@{bid_px} pos={pos} "
                f"basket_dev={basket_dev:.1f} fair_dev={fair_dev:.1f} skew={skew}"
            )

        if ask_qty > 0:
            result.setdefault(product, []).append(Order(product, ask_px, -ask_qty))
            logger.print(
                f"{product} MM ASK {ask_qty}@{ask_px} pos={pos} "
                f"basket_dev={basket_dev:.1f} fair_dev={fair_dev:.1f} skew={skew}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Microchip strategies
# ─────────────────────────────────────────────────────────────────────────────

def microchip_curve_make(state: TradingState, result: Dict[str, List[Order]], td: dict) -> None:
    for product in MICROCHIP_PRODUCTS:
        od = state.order_depths.get(product)
        if not (od and od.buy_orders and od.sell_orders):
            continue

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        spread = best_ask - best_bid

        if spread < 3:
            continue

        mid = (best_bid + best_ask) / 2
        pos = state.position.get(product, 0)

        fast_key = f"mc_fast_{product}"
        slow_key = f"mc_slow_{product}"

        prev_fast = td.get(fast_key, mid)
        prev_slow = td.get(slow_key, mid)

        fast = prev_fast + MICROCHIP_FAST_ALPHA * (mid - prev_fast)
        slow = prev_slow + MICROCHIP_SLOW_ALPHA * (mid - prev_slow)

        td[fast_key] = round(fast, 3)
        td[slow_key] = round(slow, 3)

        trend = fast - slow
        thresh = MICROCHIP_TREND_THRESH[product]

        direction = 1 if trend > thresh else -1 if trend < -thresh else 0

        max_improve = max(0, (spread - 1) // 2)

        if direction > 0:
            bid_improve = min(max_improve, 2)
            ask_improve = 0
        elif direction < 0:
            bid_improve = 0
            ask_improve = min(max_improve, 2)
        else:
            bid_improve = min(max_improve, 1)
            ask_improve = min(max_improve, 1)

        bid_px = best_bid + bid_improve
        ask_px = best_ask - ask_improve

        if bid_px >= ask_px:
            continue

        buy_cap = MICROCHIP_LIMIT - pos
        sell_cap = MICROCHIP_LIMIT + pos
        base_qty = MICROCHIP_TREND_QTY if direction != 0 else MICROCHIP_MM_QTY

        bid_qty = max(0, min(base_qty, buy_cap))
        ask_qty = max(0, min(base_qty, sell_cap))

        if direction > 0:
            ask_qty = min(ask_qty, max(1, MICROCHIP_MM_QTY // 3))
        elif direction < 0:
            bid_qty = min(bid_qty, max(1, MICROCHIP_MM_QTY // 3))

        if bid_qty > 0:
            result.setdefault(product, []).append(Order(product, bid_px, bid_qty))
            logger.print(f"{product} MC BID {bid_qty}@{bid_px} pos={pos} trend={trend:.1f}")

        if ask_qty > 0:
            result.setdefault(product, []).append(Order(product, ask_px, -ask_qty))
            logger.print(f"{product} MC ASK {ask_qty}@{ask_px} pos={pos} trend={trend:.1f}")


def update_microchip_flow_signals(state: TradingState, td: dict) -> None:
    pending = td.get(MICROCHIP_FLOW_KEY, [])

    if not isinstance(pending, list):
        pending = []

    score = 0
    seen_sectors = set()

    for product, trades in state.market_trades.items():
        sector = product_sector(product)

        if sector not in MICROCHIP_FLOW_SOURCE_SECTORS:
            continue

        od = state.order_depths.get(product)
        if not (od and od.buy_orders and od.sell_orders):
            continue

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        product_score = 0
        has_aggressor_trade = False

        for trade in trades:
            if trade.price <= best_bid:
                product_score += 1
                has_aggressor_trade = True
            elif trade.price >= best_ask:
                product_score -= 1
                has_aggressor_trade = True

        if has_aggressor_trade:
            score += product_score
            seen_sectors.add(sector)

    if len(seen_sectors) >= MICROCHIP_FLOW_MIN_SECTORS and abs(score) > MICROCHIP_FLOW_THRESHOLD:
        direction = 1 if score > 0 else -1
        pending.append(
            [
                state.timestamp + MICROCHIP_FLOW_LAG_MS,
                state.timestamp + MICROCHIP_FLOW_LAG_MS + MICROCHIP_FLOW_HOLD_MS,
                direction,
                score,
            ]
        )

    live = []

    for value in pending:
        try:
            start, expiry, direction, score = int(value[0]), int(value[1]), int(value[2]), float(value[3])
        except Exception:
            continue

        if state.timestamp <= expiry:
            live.append([start, expiry, direction, score])

    if live:
        td[MICROCHIP_FLOW_KEY] = live
    else:
        td.pop(MICROCHIP_FLOW_KEY, None)


def microchip_flow_strategy(state: TradingState, result: Dict[str, List[Order]], td: dict) -> None:
    pending = td.get(MICROCHIP_FLOW_KEY, [])

    if not isinstance(pending, list):
        return

    active_score = 0.0

    for value in pending:
        try:
            start, expiry, direction, score = int(value[0]), int(value[1]), int(value[2]), float(value[3])
        except Exception:
            continue

        if start <= state.timestamp <= expiry:
            active_score += direction * abs(score)

    if active_score == 0:
        return

    direction = 1 if active_score > 0 else -1

    for product in MICROCHIP_PRODUCTS:
        od = state.order_depths.get(product)
        if not (od and od.buy_orders and od.sell_orders):
            continue

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        pos = state.position.get(product, 0)

        if direction > 0:
            qty = min(5, MICROCHIP_LIMIT - pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_ask, qty))
                logger.print(f"{product} MC_FLOW BID {qty}@{best_ask} score={active_score:.0f} pos={pos}")
        else:
            qty = min(5, MICROCHIP_LIMIT + pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_bid, -qty))
                logger.print(f"{product} MC_FLOW ASK {qty}@{best_bid} score={active_score:.0f} pos={pos}")


def _microchip_offset(state: TradingState, td: dict) -> float:
    if "mc_offset" in td:
        return float(td["mc_offset"])

    observed = {}

    for product in MICROCHIP_PRODUCTS:
        od = state.order_depths.get(product)
        if od and od.buy_orders and od.sell_orders:
            observed[product] = (max(od.buy_orders) + min(od.sell_orders)) / 2

    if len(observed) < 3:
        return 0.0

    local_tick = state.timestamp // 100

    if np is not None:
        products = list(observed)
        coef_matrix = _np_coef_matrix(products, MICROCHIP_FOURIER_COEF, MICROCHIP_EFFECTIVE_K)
        mids = np.array([observed[p] for p in products], dtype=float)

        def offset_error(offset: int) -> float:
            x = float(offset + local_tick)
            basis, _deriv_basis = _np_basis(MICROCHIP_EFFECTIVE_K, MICROCHIP_FOURIER_PERIOD, x)
            diff = mids - coef_matrix.dot(basis)
            return float(np.mean(diff * diff))
    else:
        def offset_error(offset: int) -> float:
            x = float(offset + local_tick)
            terms = _fourier_terms(MICROCHIP_EFFECTIVE_K, MICROCHIP_FOURIER_PERIOD, x)
            error = 0.0

            for product, mid in observed.items():
                fair = _fourier_value_with_terms(MICROCHIP_FOURIER_COEF[product], x, terms)
                diff = mid - fair
                error += diff * diff

            return error / len(observed)

    best_offset = 0
    best_error = 10 ** 18

    for offset in range(0, int(MICROCHIP_FOURIER_PERIOD), 100):
        error = offset_error(offset)
        if error < best_error:
            best_error = error
            best_offset = offset

    refine_start = max(0, best_offset - 150)
    refine_end = min(int(MICROCHIP_FOURIER_PERIOD), best_offset + 151)

    for offset in range(refine_start, refine_end, 10):
        error = offset_error(offset)
        if error < best_error:
            best_error = error
            best_offset = offset

    td["mc_offset"] = best_offset
    return float(best_offset)


def _microchip_fourier_value_and_derivative(product: str, x: float) -> Tuple[float, float]:
    terms = _fourier_terms(MICROCHIP_EFFECTIVE_K, MICROCHIP_FOURIER_PERIOD, x)
    return _fourier_value_and_derivative_with_terms(MICROCHIP_FOURIER_COEF[product], x, terms)


def microchip_fourier_strategy(state: TradingState, result: Dict[str, List[Order]], td: dict) -> None:
    offset = _microchip_offset(state, td)
    x = offset + state.timestamp // 100
    forecast_x = float(x + 3)

    np_basis, np_deriv_basis = _np_basis(MICROCHIP_EFFECTIVE_K, MICROCHIP_FOURIER_PERIOD, forecast_x)
    terms = None if np_basis is not None else _fourier_terms(
        MICROCHIP_EFFECTIVE_K,
        MICROCHIP_FOURIER_PERIOD,
        forecast_x,
    )

    for product in MICROCHIP_PRODUCTS:
        od = state.order_depths.get(product)
        if not (od and od.buy_orders and od.sell_orders):
            continue

        if np_basis is not None:
            coef = _np_coef(product, MICROCHIP_FOURIER_COEF[product], MICROCHIP_EFFECTIVE_K)
            fair = float(coef.dot(np_basis))
            deriv = float(coef.dot(np_deriv_basis))
        else:
            fair, deriv = _fourier_value_and_derivative_with_terms(
                MICROCHIP_FOURIER_COEF[product],
                forecast_x,
                terms,
            )

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2
        pos = state.position.get(product, 0)

        edge = fair - mid
        signal = edge + 20.0 * deriv

        target = 0
        if signal > 12:
            target = MICROCHIP_LIMIT
        elif signal < -12:
            target = -MICROCHIP_LIMIT

        if target > pos:
            qty = min(5, target - pos, MICROCHIP_LIMIT - pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_ask, qty))
        elif target < pos:
            qty = min(5, pos - target, MICROCHIP_LIMIT + pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_bid, -qty))


# ─────────────────────────────────────────────────────────────────────────────
# Robot basket strategy
# ─────────────────────────────────────────────────────────────────────────────

def _robot_top(od: OrderDepth):
    if not (od and od.buy_orders and od.sell_orders):
        return None

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)

    return best_bid, best_ask, od.buy_orders[best_bid], -od.sell_orders[best_ask]


def _robot_zscore(history: List[float], value: float, window: int, min_points: int):
    if len(history) < min_points:
        return None

    recent = history[-window:]
    n = len(recent)

    total = 0.0
    total_sq = 0.0

    for x in recent:
        total += x
        total_sq += x * x

    mean = total / n
    var = total_sq / n - mean * mean

    if var <= 0:
        return None

    return (value - mean) / math.sqrt(var)


def _robot_target_from_z(z, entry_z: float, exit_z: float, current_pos: int) -> int:
    if z is None:
        return current_pos
    if abs(z) <= exit_z:
        return 0
    if z > entry_z:
        return -ROBOT_POSITION_LIMIT
    if z < -entry_z:
        return ROBOT_POSITION_LIMIT
    return current_pos


def _robot_side_from_z(z, entry_z: float) -> int:
    if z is None:
        return 0
    if z > entry_z:
        return -1
    if z < -entry_z:
        return 1
    return 0


def _robot_add_aggressive_to_target(
    result: Dict[str, List[Order]],
    product: str,
    od: OrderDepth,
    planned_position: Dict[str, int],
    target_pos: int,
) -> None:
    top = _robot_top(od)

    if top is None:
        return

    best_bid, best_ask, bid_volume, ask_volume = top
    current_pos = planned_position[product]
    diff = target_pos - current_pos

    if diff == 0:
        return

    if abs(target_pos) > abs(current_pos) and best_ask - best_bid > ROBOT_MAX_TAKE_SPREAD:
        return

    if diff > 0:
        qty = min(diff, ask_volume, ROBOT_POSITION_LIMIT - current_pos)
        if qty > 0:
            result.setdefault(product, []).append(Order(product, best_ask, qty))
            planned_position[product] = current_pos + qty
    else:
        qty = min(-diff, bid_volume, ROBOT_POSITION_LIMIT + current_pos)
        if qty > 0:
            result.setdefault(product, []).append(Order(product, best_bid, -qty))
            planned_position[product] = current_pos - qty


def _robot_add_passive_overlay(
    result: Dict[str, List[Order]],
    product: str,
    od: OrderDepth,
    planned_position: Dict[str, int],
    side: int,
    max_qty: int,
) -> None:
    if not ROBOT_ENABLE_PASSIVE_OVERLAY or side == 0 or result.get(product):
        return

    top = _robot_top(od)

    if top is None:
        return

    best_bid, best_ask, _bid_volume, _ask_volume = top

    if best_ask - best_bid <= 1:
        return

    current_pos = planned_position[product]

    if side > 0:
        price = best_bid + 1
        qty = min(max_qty, ROBOT_POSITION_LIMIT - current_pos)

        if qty > 0 and price < best_ask:
            result.setdefault(product, []).append(Order(product, price, qty))
            planned_position[product] = current_pos + qty
    else:
        price = best_ask - 1
        qty = min(max_qty, ROBOT_POSITION_LIMIT + current_pos)

        if qty > 0 and price > best_bid:
            result.setdefault(product, []).append(Order(product, price, -qty))
            planned_position[product] = current_pos - qty


def robot_basket_strategy(state: TradingState, result: Dict[str, List[Order]], td: dict) -> None:
    global ROBOT_LAST_TIMESTAMP

    if ROBOT_LAST_TIMESTAMP is not None and state.timestamp <= ROBOT_LAST_TIMESTAMP:
        ROBOT_S1_HISTORY.clear()
        ROBOT_S2_HISTORY.clear()

    ROBOT_LAST_TIMESTAMP = state.timestamp

    mids: Dict[str, float] = {}

    for product in ROBOT_PRODUCTS:
        top = _robot_top(state.order_depths.get(product))
        if top is None:
            return

        best_bid, best_ask, _bid_volume, _ask_volume = top
        mids[product] = (best_bid + best_ask) / 2.0

    s1 = mids[ROBOT_VACUUMING] + 2 * mids[ROBOT_DISHES] + 2 * mids[ROBOT_MOPPING]
    s2 = 3 * mids[ROBOT_MOPPING] + mids[ROBOT_IRONING] + 2 * mids[ROBOT_LAUNDRY]

    s1_hist = ROBOT_S1_HISTORY
    s2_hist = ROBOT_S2_HISTORY

    z1_200 = _robot_zscore(s1_hist, s1, 200, 100)
    z1_500 = _robot_zscore(s1_hist, s1, 500, 200)
    z2_200 = _robot_zscore(s2_hist, s2, 200, 100)
    z2_1000 = _robot_zscore(s2_hist, s2, 1000, 200)

    planned_position = {product: state.position.get(product, 0) for product in ROBOT_PRODUCTS}

    aggressive_rules = (
        (ROBOT_DISHES, z1_200, 2.00, 0.25),
        (ROBOT_MOPPING, z1_500, 1.00, 0.00),
        (ROBOT_VACUUMING, z1_500, 2.00, 0.00),
        (ROBOT_LAUNDRY, z2_200, 2.50, 0.00),
        (ROBOT_IRONING, z2_1000, 1.50, 0.00),
    )

    for product, z, entry_z, exit_z in aggressive_rules:
        target = _robot_target_from_z(z, entry_z, exit_z, planned_position[product])
        _robot_add_aggressive_to_target(
            result,
            product,
            state.order_depths[product],
            planned_position,
            target,
        )

    _robot_add_passive_overlay(
        result,
        ROBOT_VACUUMING,
        state.order_depths[ROBOT_VACUUMING],
        planned_position,
        _robot_side_from_z(z1_500, 0.85),
        2,
    )

    _robot_add_passive_overlay(
        result,
        ROBOT_IRONING,
        state.order_depths[ROBOT_IRONING],
        planned_position,
        _robot_side_from_z(z2_1000, 1.00),
        2,
    )

    _robot_add_passive_overlay(
        result,
        ROBOT_LAUNDRY,
        state.order_depths[ROBOT_LAUNDRY],
        planned_position,
        _robot_side_from_z(z2_200, 1.25),
        2,
    )

    s1_hist.append(round(s1, 1))
    s2_hist.append(round(s2, 1))

    if len(s1_hist) > ROBOT_MAX_HISTORY:
        del s1_hist[: len(s1_hist) - ROBOT_MAX_HISTORY]

    if len(s2_hist) > ROBOT_MAX_HISTORY:
        del s2_hist[: len(s2_hist) - ROBOT_MAX_HISTORY]

    td.pop("robot_s1", None)
    td.pop("robot_s2", None)


# ─────────────────────────────────────────────────────────────────────────────
# Fourier helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fourier_terms(k_count: int, period: float, x: float) -> List[Tuple[float, float, float]]:
    base_w = 2 * math.pi / period
    terms = []

    for k in range(1, k_count + 1):
        w = base_w * k
        terms.append((math.sin(w * x), math.cos(w * x), w))

    return terms


def _np_weights(k_count: int, period: float):
    if np is None:
        return None

    key = (k_count, period)
    weights = NP_W_CACHE.get(key)

    if weights is None:
        weights = np.arange(1, k_count + 1, dtype=float) * (2 * math.pi / period)
        NP_W_CACHE[key] = weights

    return weights


def _np_basis(k_count: int, period: float, x: float):
    weights = _np_weights(k_count, period)

    if weights is None:
        return None, None

    angles = weights * x
    sin_x = np.sin(angles)
    cos_x = np.cos(angles)

    basis = np.empty(2 + 2 * k_count, dtype=float)
    basis[0] = 1.0
    basis[1] = x
    basis[2::2] = sin_x
    basis[3::2] = cos_x

    deriv_basis = np.empty_like(basis)
    deriv_basis[0] = 0.0
    deriv_basis[1] = 1.0
    deriv_basis[2::2] = weights * cos_x
    deriv_basis[3::2] = -weights * sin_x

    return basis, deriv_basis


def _np_coef(product: str, coef: List[float], k_count: int):
    if np is None:
        return None

    key = (product, k_count)
    cached = NP_COEF_CACHE.get(key)

    if cached is None:
        cached = np.array(coef[: 2 + 2 * k_count], dtype=float)
        NP_COEF_CACHE[key] = cached

    return cached


def _np_coef_matrix(products: List[str], coefs: Dict[str, List[float]], k_count: int):
    if np is None:
        return None

    key = tuple(id(coefs[p]) for p in products) + (k_count,)
    cached = NP_MATRIX_CACHE.get(key)

    if cached is None:
        coef_len = 2 + 2 * k_count
        cached = np.array([coefs[p][:coef_len] for p in products], dtype=float)
        NP_MATRIX_CACHE[key] = cached

    return cached


def _fourier_value_with_terms(coef: List[float], x: float, terms: List[Tuple[float, float, float]]) -> float:
    value = coef[0] + coef[1] * x
    idx = 2

    for sx, cx, _w in terms:
        value += coef[idx] * sx + coef[idx + 1] * cx
        idx += 2

    return value


def _fourier_value_and_derivative_with_terms(
    coef: List[float],
    x: float,
    terms: List[Tuple[float, float, float]],
) -> Tuple[float, float]:
    value = coef[0] + coef[1] * x
    deriv = coef[1]
    idx = 2

    for sx, cx, w in terms:
        sin_c = coef[idx]
        cos_c = coef[idx + 1]

        value += sin_c * sx + cos_c * cx
        deriv += w * (sin_c * cx - cos_c * sx)

        idx += 2

    return value, deriv


def _fourier_value_and_derivative(coef: List[float], k_count: int, period: float, x: float) -> Tuple[float, float]:
    terms = _fourier_terms(k_count, period, x)
    return _fourier_value_and_derivative_with_terms(coef, x, terms)


def _infer_fourier_offset(
    state: TradingState,
    td: dict,
    key: str,
    products: List[str],
    coefs: Dict[str, List[float]],
    k_count: int,
    period: float,
) -> float:
    if key in td:
        return float(td[key])

    observed = {}

    for product in products:
        od = state.order_depths.get(product)
        if od and od.buy_orders and od.sell_orders:
            observed[product] = (max(od.buy_orders) + min(od.sell_orders)) / 2

    if len(observed) < 3:
        return 0.0

    local_tick = state.timestamp // 100

    if np is not None:
        observed_products = list(observed)
        coef_matrix = _np_coef_matrix(observed_products, coefs, k_count)
        mids = np.array([observed[p] for p in observed_products], dtype=float)

        def offset_error(offset: int) -> float:
            x = float(offset + local_tick)
            basis, _deriv_basis = _np_basis(k_count, period, x)
            diff = mids - coef_matrix.dot(basis)
            return float(np.mean(diff * diff))
    else:
        def offset_error(offset: int) -> float:
            x = float(offset + local_tick)
            terms = _fourier_terms(k_count, period, x)
            error = 0.0

            for product, mid in observed.items():
                fair = _fourier_value_with_terms(coefs[product], x, terms)
                diff = mid - fair
                error += diff * diff

            return error / len(observed)

    best_offset = 0
    best_error = 10 ** 18

    for offset in range(0, int(period), 100):
        error = offset_error(offset)
        if error < best_error:
            best_error = error
            best_offset = offset

    refine_start = max(0, best_offset - 150)
    refine_end = min(int(period), best_offset + 151)

    for offset in range(refine_start, refine_end, 10):
        error = offset_error(offset)
        if error < best_error:
            best_error = error
            best_offset = offset

    td[key] = best_offset
    return float(best_offset)


# ─────────────────────────────────────────────────────────────────────────────
# Generic Fourier strategies
# ─────────────────────────────────────────────────────────────────────────────

def fourier_family_strategy(
    label: str,
    products: List[str],
    coefs: Dict[str, List[float]],
    k_count: int,
    period: float,
    state: TradingState,
    result: Dict[str, List[Order]],
    td: dict,
) -> None:
    offset = _infer_fourier_offset(state, td, f"{label}_offset", products, coefs, k_count, period)
    x = offset + state.timestamp // 100
    forecast_x = float(x + 3)

    np_basis, np_deriv_basis = _np_basis(k_count, period, forecast_x)
    terms = None if np_basis is not None else _fourier_terms(k_count, period, forecast_x)

    for product in products:
        od = state.order_depths.get(product)
        if not (od and od.buy_orders and od.sell_orders):
            continue

        if np_basis is not None:
            coef = _np_coef(product, coefs[product], k_count)
            fair = float(coef.dot(np_basis))
            deriv = float(coef.dot(np_deriv_basis))
        else:
            fair, deriv = _fourier_value_and_derivative_with_terms(coefs[product], forecast_x, terms)

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2
        pos = state.position.get(product, 0)

        signal = (fair - mid) + 20.0 * deriv

        target = 0
        if signal > 12:
            target = 10
        elif signal < -12:
            target = -10

        if target > pos:
            qty = min(5, target - pos, 10 - pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_ask, qty))
        elif target < pos:
            qty = min(5, pos - target, 10 + pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_bid, -qty))


def _fourier_single_mode_delta(coef: List[float], mode: int, period: float, x0: float, x1: float) -> float:
    idx = 2 + 2 * (mode - 1)

    if idx + 1 >= len(coef):
        return 0.0

    w = 2 * math.pi * mode / period
    sin_c = coef[idx]
    cos_c = coef[idx + 1]

    current = sin_c * math.sin(w * x0) + cos_c * math.cos(w * x0)
    forecast = sin_c * math.sin(w * x1) + cos_c * math.cos(w * x1)

    return forecast - current


def fourier_mode_strategy(
    label: str,
    products: List[str],
    coefs: Dict[str, List[float]],
    k_count: int,
    period: float,
    state: TradingState,
    result: Dict[str, List[Order]],
    td: dict,
    modes: List[int],
    horizon: int,
    threshold: float,
    limit: int,
    order_qty: int,
) -> None:
    offset = _infer_fourier_offset(state, td, f"{label}_mode_offset", products, coefs, k_count, period)
    x0 = offset + state.timestamp // 100
    x1 = float(x0 + horizon)

    for product in products:
        od = state.order_depths.get(product)
        if not (od and od.buy_orders and od.sell_orders):
            continue

        delta = coefs[product][1] * horizon

        for mode in modes:
            delta += _fourier_single_mode_delta(coefs[product], mode, period, float(x0), x1)

        pos = state.position.get(product, 0)
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        if delta > threshold:
            qty = min(order_qty, limit - pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_ask, qty))
        elif delta < -threshold:
            qty = min(order_qty, limit + pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_bid, -qty))


# ─────────────────────────────────────────────────────────────────────────────
# Main trader
# ─────────────────────────────────────────────────────────────────────────────

class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        td = load_trader_data(state.traderData)

        microchip_fourier_strategy(state, result, td)

        fourier_family_strategy(
            "galaxy",
            GALAXY_PRODUCTS,
            GALAXY_FOURIER_COEF,
            GALAXY_EFFECTIVE_K,
            GALAXY_FOURIER_PERIOD,
            state,
            result,
            td,
        )

        robot_basket_strategy(state, result, td)

        fourier_family_strategy(
            "pebbles",
            PEBBLES_FOURIER_PRODUCTS,
            PEBBLES_FOURIER_COEF,
            PEBBLES_EFFECTIVE_K,
            PEBBLES_FOURIER_PERIOD,
            state,
            result,
            td,
        )

        fourier_family_strategy(
            "sleep_pod",
            SLEEP_POD_PRODUCTS,
            SLEEP_POD_COEF,
            SLEEP_POD_EFFECTIVE_K,
            SLEEP_POD_PERIOD,
            state,
            result,
            td,
        )

        snackpack_strategy(
            "SNACKPACK_VANILLA",
            CLUSTER_SNACKPACK_VANILLA,
            SNPK_RB_LIMIT,
            SNPK_RB_LIMIT,
            SNPK_ENTRY_THRESH_VANILLA,
            SNPK_EXIT_THRESH_VANILLA,
            state,
            result,
            td,
            "snpk_vanilla_mr",
        )

        snackpack_strategy(
            "SNACKPACK_RASPBERRY",
            CLUSTER_SNACKPACK_RASPBERRY,
            SNPK_RB_LIMIT,
            SNPK_RB_LIMIT,
            SNPK_ENTRY_THRESH_RASPBERRY,
            SNPK_EXIT_THRESH_RASPBERRY,
            state,
            result,
            td,
            "snpk_raspberry_mr",
        )

        fourier_mode_strategy(
            "oxygen_shake",
            OXYGEN_SHAKE_PRODUCTS,
            OXYGEN_SHAKE_FOURIER_COEF,
            OXYGEN_SHAKE_FOURIER_K,
            OXYGEN_SHAKE_FOURIER_PERIOD,
            state,
            result,
            td,
            modes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            horizon=20,
            threshold=3.0,
            limit=10,
            order_qty=3,
        )

        fourier_mode_strategy(
            "panel",
            PANEL_PRODUCTS,
            PANEL_FOURIER_COEF,
            PANEL_FOURIER_K,
            PANEL_FOURIER_PERIOD,
            state,
            result,
            td,
            modes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            horizon=20,
            threshold=3.0,
            limit=10,
            order_qty=3,
        )

        fourier_mode_strategy(
            "translator",
            TRANSLATOR_PRODUCTS,
            TRANSLATOR_FOURIER_COEF,
            TRANSLATOR_FOURIER_K,
            TRANSLATOR_FOURIER_PERIOD,
            state,
            result,
            td,
            modes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            horizon=20,
            threshold=1.0,
            limit=10,
            order_qty=3,
        )

        fourier_mode_strategy(
            "uv_visor",
            UV_VISOR_PRODUCTS,
            UV_VISOR_FOURIER_COEF,
            UV_VISOR_FOURIER_K,
            UV_VISOR_FOURIER_PERIOD,
            state,
            result,
            td,
            modes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            horizon=20,
            threshold=3.0,
            limit=10,
            order_qty=3,
        )

        td.pop("robot_s1", None)
        td.pop("robot_s2", None)

        trader_data = json.dumps(td, separators=(",", ":")) if td else ""
        conversions = 0

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data
    





SNACKPACK_FOURIER_K = 14
SNACKPACK_FOURIER_PERIOD = 30000.0
SNACKPACK_FOURIER_COEF = {
    "SNACKPACK_CHOCOLATE": [
        10039.00370214,
        -0.01304257,
        42.73433261,
        -72.74615221,
        15.46827266,
        19.212331,
        63.23653569,
        21.60415631,
        -80.22996987,
        23.01494342,
        -59.29935719,
        -50.53737831,
        -17.85975347,
        24.81050804,
        43.22487834,
        -17.71371029,
        17.62863713,
        -8.58050252,
        -10.08703323,
        -66.67035524,
        1.98645819,
        25.85108303,
        -25.46658177,
        16.64005786,
        -4.99840341,
        35.35492773,
        -16.07602566,
        -6.30529342,
        -4.13281238,
        36.85503892,
    ],
    "SNACKPACK_PISTACHIO": [
        9966.2841186,
        -0.03136373,
        -151.60830312,
        8.03543829,
        -138.95609629,
        -16.33662021,
        -1.15133922,
        7.86113545,
        -27.50541495,
        5.16397429,
        17.81490866,
        7.87457295,
        1.50059181,
        -16.74192365,
        -20.55294286,
        -7.83602093,
        27.38249083,
        27.97846208,
        -17.6011605,
        26.69165029,
        -17.67204921,
        22.56514014,
        -10.46482681,
        15.96021732,
        10.5964217,
        20.13208892,
        29.3912849,
        -12.34196192,
        38.82330988,
        16.62470089,
    ],
    "SNACKPACK_RASPBERRY": [
        9897.47045156,
        0.01202318,
        117.49697702,
        4.06747843,
        112.77897283,
        117.6967118,
        -17.22962403,
        35.85686803,
        59.59475345,
        -8.13997499,
        -3.79154459,
        -39.19645072,
        9.00421627,
        41.88820262,
        7.07018282,
        -27.80714904,
        -41.34158325,
        -31.957671,
        28.47220692,
        -50.87937949,
        15.64472225,
        -31.39075219,
        23.19562206,
        -16.84945115,
        1.47448747,
        -24.7458376,
        -46.30371003,
        26.24868422,
        -65.3482493,
        -8.09774569,
    ],
    "SNACKPACK_STRAWBERRY": [
        10303.08343255,
        0.02690257,
        -64.11291774,
        -125.28518116,
        -147.12634096,
        -75.02770467,
        41.51169126,
        -51.51249653,
        -44.5299001,
        -36.13502804,
        15.1989863,
        37.90117503,
        -28.15914798,
        -61.86663624,
        -20.16539648,
        6.35864745,
        42.10825948,
        19.91996886,
        -13.90551631,
        49.91304947,
        -21.79873325,
        27.88014863,
        -17.4771375,
        28.45884719,
        -10.97311634,
        19.75313614,
        50.07654214,
        -14.60992326,
        60.29488741,
        7.75864771,
    ],
    "SNACKPACK_VANILLA": [
        9914.33854087,
        0.012198,
        31.05256284,
        90.93202755,
        13.01163267,
        -5.65752371,
        -82.44243117,
        1.22075553,
        83.66592375,
        -7.12940833,
        86.27210281,
        41.96355749,
        18.29539389,
        -38.63259658,
        -33.30404644,
        14.44179087,
        -21.07018947,
        14.29784215,
        21.63219779,
        64.30600621,
        2.24137864,
        -24.3015842,
        28.99416855,
        -23.40530539,
        -5.4699551,
        -37.11205037,
        13.81847601,
        4.92550235,
        9.0758489,
        -26.67754813,
    ],
}

OXYGEN_SHAKE_PRODUCTS = [
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_MORNING_BREATH",
]

OXYGEN_SHAKE_FOURIER_K = 14
OXYGEN_SHAKE_FOURIER_PERIOD = 30000.0
OXYGEN_SHAKE_FOURIER_COEF = {
    "OXYGEN_SHAKE_CHOCOLATE": [
        9096.16644607,
        0.0307152,
        16.50015399,
        479.22884941,
        -245.7320818,
        135.26318135,
        -199.33862103,
        129.95293522,
        50.70828049,
        -50.82641595,
        61.30793349,
        10.22229463,
        50.51652546,
        -11.5331191,
        67.5436931,
        -30.67585194,
        11.89828601,
        76.00195492,
        58.4026822,
        -35.88899653,
        18.65088909,
        28.85363253,
        -39.73674983,
        23.54021522,
        -6.4031109,
        19.97417259,
        14.13753606,
        -28.09667031,
        4.42949535,
        5.11733618,
    ],
    "OXYGEN_SHAKE_EVENING_BREATH": [
        9595.93463506,
        -0.02160336,
        -492.50910319,
        193.50023055,
        34.4217162,
        189.0554568,
        214.74021014,
        -19.08500409,
        -40.71224842,
        54.02998691,
        56.67921042,
        100.53117081,
        -39.14955273,
        -45.99951276,
        -16.96095349,
        -28.64329362,
        45.34060368,
        44.48052135,
        -21.4203185,
        66.96531088,
        -8.20550795,
        9.7662491,
        18.30295812,
        23.51220605,
        -14.09314982,
        9.29750075,
        1.26252324,
        -28.7222653,
        -4.46032402,
        -4.77514294,
    ],
    "OXYGEN_SHAKE_GARLIC": [
        9917.59630385,
        0.13387403,
        606.26791439,
        164.53926176,
        -22.9559558,
        -115.98100802,
        -29.3704764,
        -78.8269852,
        -3.07217808,
        -217.49076218,
        -109.42654345,
        111.04961443,
        -84.53510122,
        18.00268022,
        -27.22904814,
        8.81052955,
        -4.51463421,
        -22.42240722,
        8.85682716,
        -49.4880702,
        18.45096348,
        36.15157222,
        21.75623918,
        8.6311749,
        -18.52600621,
        -14.22094659,
        56.20356938,
        -1.0749777,
        -6.03198136,
        10.18891377,
    ],
    "OXYGEN_SHAKE_MINT": [
        9637.08835953,
        0.01342083,
        478.53604491,
        -321.09250649,
        104.1985765,
        410.3619556,
        -110.26986229,
        135.7810346,
        -39.56643029,
        -65.90113044,
        78.17273608,
        -82.77465833,
        3.0055038,
        -41.07940812,
        98.85711561,
        -28.11371494,
        70.55576965,
        -26.99869338,
        86.11178845,
        87.95769,
        31.44262586,
        55.0145842,
        -13.27196776,
        -2.61198177,
        0.69353872,
        -36.36572388,
        4.6696057,
        38.17587592,
        -19.95117898,
        27.10189381,
    ],
    "OXYGEN_SHAKE_MORNING_BREATH": [
        10193.58947157,
        -0.01287617,
        652.94750852,
        36.49751123,
        -196.08337058,
        -108.72144363,
        250.83560231,
        50.42315083,
        107.55752586,
        76.04659413,
        56.38088317,
        -47.33845159,
        137.83130109,
        27.91191772,
        -107.22421541,
        -158.43870984,
        4.83456524,
        -85.379665,
        20.91073422,
        -23.89926122,
        -45.04519236,
        -85.23038394,
        -21.16413702,
        12.83898941,
        14.5271216,
        20.50930169,
        -30.98018275,
        23.88585319,
        11.95770294,
        -20.43766679,
    ],
}

PEBBLES_PRODUCTS = ["PEBBLES_L", "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_XL"]
PEBBLES_BASKET_FAIR = 50000
PEBBLES_SOFT_CAP = 4
PEBBLES_BASE_IMPROVE = 1
PEBBLES_MOM_THRESH = 999

PEBBLES_MOM_STYLE = {
    "PEBBLES_L": 1,
    "PEBBLES_S": -1,
    "PEBBLES_XL": 1,
}

PEBBLES_MM_CFG = {
    "PEBBLES_L": (10, 2, 10),
    "PEBBLES_XS": (3, 2, 3),
    "PEBBLES_S": (10, 2, 10),
    "PEBBLES_M": (0, 2, 0),
    "PEBBLES_XL": (10, 2, 10),
}

PEBBLES_FOURIER_PRODUCTS = ["PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS"]
PEBBLES_FOURIER_K = 14
PEBBLES_FOURIER_PERIOD = 30000.0
PEBBLES_FOURIER_COEF = {
    "PEBBLES_L": [
        10522.20246115,
        -0.0232069,
        -770.58063642,
        -99.64448043,
        11.86251099,
        -267.4840482,
        322.22515871,
        -0.23453524,
        56.46982569,
        -285.39639184,
        -77.22714786,
        64.29090945,
        8.78968328,
        86.29166765,
        78.50315173,
        121.79197834,
        52.52177289,
        -46.22723535,
        -41.3129029,
        -5.00338976,
        84.78366893,
        -79.3981513,
        -31.04643277,
        6.05276719,
        -6.22408118,
        11.8481404,
        -54.02162421,
        12.35085391,
        67.23829245,
        -31.61981307,
    ],
    "PEBBLES_M": [
        9913.89911746,
        0.02329038,
        -546.60700999,
        120.40861437,
        281.06472442,
        91.92493422,
        -128.57801636,
        -115.66616365,
        -252.10427548,
        -16.73909409,
        77.78613053,
        -89.69096521,
        -155.19353225,
        83.67449322,
        14.70425572,
        77.2145837,
        -106.43655386,
        -108.41469971,
        -45.95146604,
        -37.66124622,
        73.09808441,
        37.02341416,
        9.4306467,
        15.05711191,
        -84.1595784,
        -18.50423023,
        17.95180678,
        -0.21671614,
        15.14081815,
        -5.87799069,
    ],
    "PEBBLES_S": [
        9932.21233931,
        -0.06665926,
        77.9846235,
        -255.85691932,
        233.61231772,
        100.55513006,
        72.11078337,
        -17.49632498,
        72.35344116,
        70.71993973,
        146.023342,
        231.28066325,
        94.44078459,
        -5.89718482,
        114.25569957,
        75.82260011,
        26.41091909,
        -18.12972594,
        12.32512463,
        64.12238263,
        -89.28077713,
        -43.66034451,
        3.91759298,
        -61.0001798,
        37.3501861,
        62.71784098,
        36.06560019,
        37.27463781,
        44.92855509,
        7.34031746,
    ],
    "PEBBLES_XL": [
        10351.60735254,
        0.19160518,
        863.12673166,
        -211.35094323,
        -665.57814146,
        184.35565271,
        -500.94960199,
        -115.49309743,
        151.95201047,
        107.26686647,
        -221.59716029,
        -221.27640205,
        -18.45077473,
        -232.05006079,
        -170.08721443,
        -290.56148073,
        62.26188758,
        140.03770143,
        51.2729215,
        -20.83509263,
        25.73563748,
        137.29525753,
        -35.45845246,
        47.5880168,
        63.50486147,
        -94.83107914,
        -22.49454555,
        -18.93074567,
        -98.37480396,
        30.39312902,
    ],
    "PEBBLES_XS": [
        9279.88665874,
        -0.12502063,
        376.13733082,
        446.42149178,
        139.099,
        -109.31755374,
        235.23726951,
        248.94680859,
        -28.64054107,
        124.16180939,
        75.00371052,
        15.36777537,
        70.42800489,
        67.9464655,
        -37.38828591,
        15.73302998,
        -34.76415086,
        32.74135408,
        23.68810117,
        -0.60730654,
        -94.2896514,
        -51.24053772,
        53.15625559,
        -7.67245459,
        -10.4443849,
        38.78176526,
        22.47374488,
        -30.47446946,
        -28.90743664,
        -0.22992781,
    ],
}

SLEEP_POD_PRODUCTS = [
    "SLEEP_POD_COTTON",
    "SLEEP_POD_LAMB_WOOL",
    "SLEEP_POD_NYLON",
    "SLEEP_POD_POLYESTER",
    "SLEEP_POD_SUEDE",
]

SLEEP_POD_K = 16
SLEEP_POD_PERIOD = 30000.0
SLEEP_POD_COEF = {
    "SLEEP_POD_COTTON": [
        10779.48317702,
        0.04987705,
        -542.72822961,
        -73.09924216,
        -38.79324461,
        -537.28986214,
        14.92571248,
        -174.77807247,
        79.90581099,
        60.44981511,
        11.21349578,
        -34.4410993,
        64.3827452,
        -40.09681858,
        82.74282896,
        -26.55775814,
        -90.62419449,
        -25.38796469,
        87.67206033,
        34.99069759,
        6.15331735,
        54.13827205,
        22.01010276,
        -79.95397757,
        -33.36530637,
        -75.70013143,
        46.39355536,
        -39.80073335,
        14.54316384,
        -4.39451391,
    ],
    "SLEEP_POD_LAMB_WOOL": [
        10012.2425137,
        0.04594815,
        328.93984254,
        333.68633009,
        327.28533678,
        94.7603348,
        139.7473564,
        -141.43042439,
        222.60328911,
        -141.89192016,
        154.03522645,
        -197.15269691,
        -87.72845042,
        55.72461677,
        -29.14299988,
        -64.64894895,
        16.83902088,
        3.1895429,
        -14.6855771,
        3.01318233,
        -7.96405344,
        -77.64846475,
        6.35457993,
        12.52140978,
        13.21819989,
        -63.67267678,
        20.54872317,
        -39.12215298,
        40.75405693,
        -18.55270443,
    ],
    "SLEEP_POD_NYLON": [
        9263.42746333,
        0.0248705,
        -156.00215146,
        500.58628178,
        75.09341729,
        115.53663188,
        -73.24171369,
        -15.43197153,
        -29.41160535,
        103.26992938,
        6.85968022,
        -58.48790172,
        11.27965961,
        -15.37440236,
        36.71763612,
        12.96402423,
        68.72831669,
        2.13244964,
        -44.58621821,
        -5.37461846,
        -6.4253253,
        95.34929108,
        -46.77401639,
        -4.35598075,
        12.38490762,
        62.56167089,
        25.77165447,
        10.47998979,
        -10.76115988,
        -10.62466607,
    ],
    "SLEEP_POD_POLYESTER": [
        10829.01704943,
        0.06743851,
        -445.31453446,
        -356.22761886,
        -181.56761392,
        -313.78584855,
        56.98519126,
        65.79851352,
        -53.3802067,
        -52.80307037,
        87.47642232,
        -52.99892963,
        13.04110616,
        -44.01398478,
        9.89924979,
        -101.56124044,
        -34.82519428,
        -58.23813092,
        -111.54402643,
        10.58398199,
        0.45017536,
        -1.99726982,
        14.18527252,
        -22.52244318,
        -77.23878848,
        -4.33526877,
        42.24459495,
        -4.96750116,
        -7.00873868,
        40.03446937,
    ],
    "SLEEP_POD_SUEDE": [
        10391.44234352,
        0.06706744,
        -319.32346101,
        -565.92054703,
        -87.52125509,
        144.02215037,
        66.75260372,
        78.08840991,
        117.07215262,
        -4.91107056,
        139.42245592,
        -77.94325804,
        -55.71074085,
        83.36996331,
        10.48467333,
        113.52042323,
        -74.80118647,
        13.3012508,
        25.18565722,
        -33.01845948,
        3.09655745,
        -10.56802863,
        36.19907298,
        -58.58282251,
        53.80953778,
        -65.62814647,
        24.60679356,
        -18.73155981,
        -21.87980831,
        47.11818312,
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Microchip config
# ─────────────────────────────────────────────────────────────────────────────

MICROCHIP_PRODUCTS = [
    "MICROCHIP_CIRCLE",
    "MICROCHIP_OVAL",
    "MICROCHIP_RECTANGLE",
    "MICROCHIP_SQUARE",
    "MICROCHIP_TRIANGLE",
]

MICROCHIP_LIMIT = 10
MICROCHIP_MM_QTY = 4
MICROCHIP_TREND_QTY = 5
MICROCHIP_FAST_ALPHA = 0.12
MICROCHIP_SLOW_ALPHA = 0.015

MICROCHIP_TREND_THRESH = {
    "MICROCHIP_CIRCLE": 18,
    "MICROCHIP_OVAL": 22,
    "MICROCHIP_RECTANGLE": 18,
    "MICROCHIP_SQUARE": 35,
    "MICROCHIP_TRIANGLE": 22,
}

MICROCHIP_FLOW_KEY = "mc_flow"
MICROCHIP_FLOW_HOLD_MS = 20000
MICROCHIP_FLOW_LAG_MS = 0
MICROCHIP_FLOW_THRESHOLD = 0
MICROCHIP_FLOW_MIN_SECTORS = 2

MICROCHIP_FLOW_SOURCE_SECTORS = {
    "GALAXY_SOUNDS",
    "OXYGEN_SHAKE",
    "PANEL",
    "ROBOT",
    "SLEEP_POD",
    "SNACKPACK",
    "TRANSLATOR",
    "UV_VISOR",
}

MICROCHIP_FOURIER_K = 14
MICROCHIP_FOURIER_PERIOD = 30000.0
MICROCHIP_FOURIER_COEF = {
    "MICROCHIP_CIRCLE": [
        8996.59346212,
        0.01455327,
        134.01362628,
        496.24308321,
        -305.81545172,
        244.05903719,
        -177.95471479,
        28.30916413,
        -10.53229102,
        53.40546461,
        -74.56943241,
        -3.36975825,
        9.59209689,
        -30.19484575,
        -98.92367866,
        -41.55886868,
        -25.82437096,
        50.49503112,
        -26.42105756,
        -25.15147445,
        -13.72027416,
        80.4130478,
        62.15769084,
        38.00113177,
        0.33323792,
        -12.73475922,
        -31.58063068,
        -8.34412403,
        8.75853403,
        19.44060335,
    ],
    "MICROCHIP_OVAL": [
        10258.52132952,
        -0.13859946,
        486.14055842,
        -471.52734077,
        32.7932349,
        -55.32713724,
        -53.3435391,
        69.04189354,
        17.41167105,
        17.37085832,
        182.36837253,
        -15.34471639,
        -72.70125284,
        -63.69621119,
        -72.77406243,
        -18.670603,
        -33.6707622,
        110.79842147,
        -12.38484125,
        -30.39874276,
        33.39497825,
        66.73731023,
        29.45190972,
        -40.56332104,
        58.3527476,
        -1.80722307,
        16.25881711,
        -31.10699509,
        10.57701279,
        -24.4141057,
    ],
    "MICROCHIP_RECTANGLE": [
        9343.47390709,
        -0.040737,
        341.49284574,
        503.85224714,
        -178.66222386,
        -98.62538185,
        192.59154408,
        61.73396307,
        227.38742119,
        -40.59643851,
        118.17803729,
        31.69530342,
        0.10015364,
        21.55298786,
        -12.58490355,
        1.44518567,
        0.05919692,
        61.02166001,
        -80.6429239,
        -34.82733818,
        -33.74136713,
        49.13391259,
        44.77967687,
        -32.7437979,
        2.58562075,
        28.78061753,
        22.05171543,
        -35.72175235,
        27.46706573,
        23.91575697,
    ],
    "MICROCHIP_SQUARE": [
        11893.90239694,
        0.11339351,
        -952.48296394,
        -1124.15515309,
        83.90963671,
        33.31038156,
        -50.95379799,
        -313.24784683,
        -48.21069393,
        -225.80222716,
        44.88322719,
        10.23895781,
        -162.24870999,
        -20.92487874,
        -233.01341277,
        -0.192173,
        -113.1630433,
        9.84747502,
        53.03028476,
        -68.06167735,
        -138.22138114,
        -39.63025687,
        88.16043388,
        -79.021719,
        98.41144053,
        132.05884149,
        -35.2778788,
        122.00866776,
        -18.50674708,
        -42.46298954,
    ],
    "MICROCHIP_TRIANGLE": [
        10765.54765503,
        -0.07194617,
        56.10302765,
        -567.15647075,
        132.25519867,
        85.88201181,
        -92.10800619,
        -17.70765493,
        28.81374904,
        -166.26127072,
        -77.89979824,
        113.27233793,
        21.23447069,
        -90.8930176,
        15.86688269,
        39.81928483,
        -77.41906171,
        6.10222947,
        -22.57392246,
        12.2070496,
        72.014299,
        20.43695596,
        -31.99016522,
        -4.68493278,
        -16.47316484,
        -35.06032297,
        80.31549478,
        -9.23133327,
        -58.07042216,
        -61.22182432,
    ],
}

GALAXY_PRODUCTS = [
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
]

GALAXY_FOURIER_K = 14
GALAXY_FOURIER_PERIOD = 30000.0
GALAXY_FOURIER_COEF = {
    "GALAXY_SOUNDS_BLACK_HOLES": [
        9539.2084603,
        0.12851519,
        459.88230707,
        278.94951891,
        -52.07906116,
        -264.85987697,
        165.9355438,
        157.68184532,
        171.53612275,
        75.75398208,
        -9.26663673,
        8.12442273,
        -29.76770438,
        68.47930612,
        -157.5466558,
        44.2000469,
        10.20420327,
        -10.96488959,
        -63.58734167,
        55.52634886,
        -63.30619984,
        -23.79189154,
        -11.16080542,
        -6.18827804,
        -68.9784547,
        -26.96436392,
        -4.18394308,
        -31.31739562,
        -64.46043766,
        -25.52629019,
    ],
    "GALAXY_SOUNDS_DARK_MATTER": [
        10134.86910053,
        0.00611972,
        -26.38029418,
        -294.57664977,
        180.04276382,
        -160.15075028,
        -81.52900562,
        118.90880674,
        -10.91379164,
        -40.68457351,
        57.24314387,
        29.21618002,
        1.88035444,
        32.23555428,
        -31.88648763,
        -21.31380365,
        66.26150331,
        43.03778064,
        -23.56206828,
        32.918634,
        -19.29952794,
        11.82556365,
        -68.32716456,
        21.52924636,
        2.69207308,
        25.1701998,
        -28.72557245,
        15.81467791,
        -34.05328011,
        -18.84073772,
    ],
    "GALAXY_SOUNDS_PLANETARY_RINGS": [
        10929.3270155,
        -0.01084395,
        -547.61972327,
        -790.72715183,
        -251.47398232,
        307.50090512,
        -146.71925275,
        109.88001977,
        211.8213777,
        -271.65446626,
        66.74676151,
        29.42001216,
        -19.42926015,
        87.3440047,
        13.63985899,
        -45.146727,
        20.36718721,
        -54.53547414,
        24.18083395,
        -24.52166095,
        -7.46348325,
        -42.86800897,
        -16.2750808,
        -40.51962786,
        7.64947623,
        20.07224084,
        -14.18150472,
        -7.0649997,
        -13.64192379,
        66.91724582,
    ],
    "GALAXY_SOUNDS_SOLAR_FLAMES": [
        10580.95360079,
        0.03410901,
        625.45843512,
        -149.62819895,
        -60.30769324,
        -50.89670443,
        18.7810721,
        -188.58498146,
        -183.72748753,
        -197.68878281,
        180.53635421,
        87.84794088,
        66.48896289,
        -122.0707403,
        85.55414778,
        -48.41543985,
        -36.19365351,
        -6.52914659,
        20.53449479,
        -12.64502415,
        9.14746586,
        -33.78209478,
        3.28969799,
        -21.26236373,
        47.28721097,
        -2.00927331,
        7.04279215,
        -15.34827914,
        23.50152842,
        -26.02697985,
    ],
    "GALAXY_SOUNDS_SOLAR_WINDS": [
        10100.20376406,
        0.0224901,
        -403.47403094,
        196.46244947,
        224.152816,
        72.30725132,
        14.02372716,
        -92.53495034,
        -41.65713532,
        -23.82610668,
        -23.43155375,
        -73.39905576,
        0.64850883,
        -189.90662186,
        131.51491165,
        -138.43630565,
        -6.28231068,
        18.66055613,
        -80.54447666,
        -12.58068292,
        27.56986934,
        43.02949399,
        -28.52009038,
        -7.46171766,
        -46.48266672,
        35.44945386,
        -14.21615958,
        25.42559467,
        -31.71297692,
        10.0255324,
    ],
}

TRANSLATOR_PRODUCTS = [
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_VOID_BLUE",
]

TRANSLATOR_FOURIER_K = 14
TRANSLATOR_FOURIER_PERIOD = 30000.0
TRANSLATOR_FOURIER_COEF = {
    "TRANSLATOR_ASTRO_BLACK": [
        10048.84750831,
        -0.04424336,
        -15.53502122,
        312.49213997,
        -116.57253766,
        -293.48899558,
        7.50534441,
        45.24518142,
        -48.07138914,
        -64.0444261,
        11.84495429,
        5.25475819,
        97.71691321,
        52.06227317,
        -6.45399414,
        -24.47640773,
        -51.42740318,
        -10.64228332,
        18.09952604,
        72.14895104,
        -0.2011893,
        13.77410405,
        -15.54907054,
        3.55424348,
        -51.6349592,
        10.35078993,
        -32.62407513,
        -62.76731957,
        -41.27350603,
        0.11100066,
    ],
    "TRANSLATOR_ECLIPSE_CHARCOAL": [
        10051.08165709,
        -0.01582318,
        -146.57783189,
        -289.62198326,
        -369.84694168,
        -67.71877948,
        41.00147114,
        123.0403512,
        -5.20332896,
        90.10135738,
        20.74019616,
        18.84964151,
        20.19265977,
        -38.40477403,
        13.05326531,
        54.62883504,
        19.66409934,
        17.04595875,
        -31.69305798,
        -20.29844112,
        10.98145931,
        -26.59737228,
        -9.67149441,
        43.43448874,
        -31.87245437,
        -54.95797313,
        -7.74441411,
        22.51286365,
        -71.09746903,
        27.21486579,
    ],
    "TRANSLATOR_GRAPHITE_MIST": [
        10129.4775827,
        -0.00298929,
        -568.73464018,
        5.84572476,
        177.33854937,
        -222.45137031,
        158.46367078,
        76.54648338,
        -194.16765102,
        49.96986778,
        39.15978646,
        -68.6285252,
        116.14111139,
        9.69578317,
        -12.9844952,
        -26.99627848,
        -32.61336529,
        25.29838168,
        -11.60097017,
        -15.94588843,
        -26.92131887,
        -34.35603193,
        17.7867209,
        9.93031987,
        -28.49362303,
        -11.7482187,
        -7.37287291,
        -9.85057857,
        -36.95502542,
        20.22178734,
    ],
    "TRANSLATOR_SPACE_GRAY": [
        10088.45770775,
        -0.04377185,
        -294.92834593,
        -223.02980494,
        276.83377387,
        42.62112858,
        -12.70010444,
        77.50240031,
        100.13574759,
        -45.61014527,
        110.85743941,
        -90.97391827,
        -51.04278269,
        112.07614068,
        46.13185068,
        20.45639156,
        -61.19696043,
        -65.26182658,
        7.55456144,
        40.06924043,
        -21.60137044,
        -7.72278145,
        69.43883655,
        -50.49115316,
        11.63841152,
        0.5735982,
        5.13306143,
        17.34725197,
        65.04332599,
        -29.12459758,
    ],
    "TRANSLATOR_VOID_BLUE": [
        10252.70807941,
        0.04039276,
        -33.66205813,
        -295.24539939,
        -330.1925451,
        68.35082833,
        -46.83768055,
        -26.06483339,
        162.19828646,
        -41.65405966,
        -32.37373444,
        85.12829744,
        51.14929732,
        7.10365122,
        -24.06613366,
        3.53689234,
        -47.85077715,
        -12.82425183,
        44.4177319,
        64.81886971,
        -38.15464967,
        14.22104671,
        39.09892614,
        -14.18740135,
        -18.51052792,
        -75.5501602,
        -13.26521933,
        -21.43259943,
        20.87096052,
        -29.85648054,
    ],
}

UV_VISOR_PRODUCTS = [
    "UV_VISOR_AMBER",
    "UV_VISOR_MAGENTA",
    "UV_VISOR_ORANGE",
    "UV_VISOR_RED",
    "UV_VISOR_YELLOW",
]

UV_VISOR_FOURIER_K = 14
UV_VISOR_FOURIER_PERIOD = 30000.0
UV_VISOR_FOURIER_COEF = {
    "UV_VISOR_AMBER": [
        9193.06927318,
        -0.08542772,
        284.50801524,
        304.66175819,
        168.57492049,
        67.33789376,
        -42.62523138,
        73.16085149,
        81.52820002,
        81.63612164,
        2.31338662,
        22.68299296,
        -30.25989175,
        68.69831195,
        27.13245785,
        27.8500772,
        57.70194532,
        -53.09863151,
        -17.38599842,
        13.10406087,
        11.72875789,
        -4.78543385,
        32.83009252,
        18.24445167,
        -32.3963318,
        47.8420976,
        1.67577059,
        -28.12788633,
        53.45990114,
        42.06304113,
    ],
    "UV_VISOR_MAGENTA": [
        10387.21962499,
        0.0483065,
        -118.90965164,
        -329.0675004,
        -67.53400031,
        -63.7390875,
        -27.87796532,
        65.95920413,
        105.32982614,
        181.99077619,
        -63.14796492,
        83.91975534,
        32.56840188,
        -64.96427517,
        42.79535465,
        -70.46288797,
        -117.10730741,
        44.20502988,
        -18.1267476,
        33.9485369,
        5.60652171,
        -117.61401515,
        29.86870734,
        6.29758923,
        -17.82318288,
        -30.00114681,
        -26.60020899,
        -12.13113667,
        -23.96872476,
        -3.78738292,
    ],
    "UV_VISOR_ORANGE": [
        10901.86570942,
        -0.03169169,
        -855.59933818,
        -166.34466039,
        -472.6296356,
        42.52622684,
        59.22483327,
        -281.27444764,
        -24.38682101,
        -20.70520927,
        -33.18128903,
        -97.93781811,
        6.07893625,
        -23.1812758,
        -16.88460585,
        -36.14377504,
        -26.53721837,
        16.63289066,
        -53.26420069,
        2.00238729,
        43.47357788,
        23.38404489,
        25.08746699,
        -31.53125829,
        -69.738225,
        -23.03554014,
        -24.9819476,
        10.98739856,
        -35.45710057,
        -16.31190449,
    ],
    "UV_VISOR_RED": [
        10161.12353814,
        0.06014667,
        123.20347668,
        443.13545074,
        30.88905733,
        -117.34431781,
        219.00327144,
        -91.48622574,
        -100.79621489,
        -27.5501695,
        22.47541989,
        -163.01431844,
        6.69725412,
        25.49695347,
        48.3133929,
        -30.32312609,
        64.80960359,
        -15.6772203,
        -27.59242112,
        -60.95518817,
        22.560468,
        33.29284601,
        -58.31272074,
        -54.35252993,
        11.95310051,
        -7.78323398,
        45.44806821,
        -31.71936099,
        -37.72015898,
        14.5059723,
    ],
    "UV_VISOR_YELLOW": [
        10984.0056153,
        -0.00176972,
        26.53849968,
        -781.84864206,
        135.62208776,
        -445.33964448,
        -31.84100128,
        146.97479539,
        -100.40518067,
        46.5980593,
        12.34047179,
        48.59764047,
        -32.85447821,
        34.85309005,
        -103.12651314,
        -12.14626623,
        22.80349215,
        -20.53717049,
        63.00559721,
        78.26252709,
        -1.96894038,
        19.31054509,
        45.50745019,
        -51.4563888,
        -52.48324149,
        -12.43764596,
        13.86742526,
        19.96705589,
        -52.84603357,
        -8.09568002,
    ],
}

PANEL_PRODUCTS = [
    "PANEL_1X2",
    "PANEL_1X4",
    "PANEL_2X2",
    "PANEL_2X4",
    "PANEL_4X4",
]

PANEL_FOURIER_K = 14
PANEL_FOURIER_PERIOD = 30000.0
PANEL_FOURIER_COEF = {
    "PANEL_1X2": [
        9043.32116714,
        -0.00803974,
        -187.73492799,
        763.06182751,
        57.7003932,
        42.50266669,
        -132.39566694,
        48.97378424,
        33.15937877,
        82.62045948,
        65.43377398,
        -44.66061807,
        -95.52600814,
        -59.21214427,
        -5.10890751,
        -7.14401547,
        -15.29998302,
        4.58029557,
        -45.26548976,
        125.02048162,
        37.99313496,
        -0.53613109,
        80.04334584,
        -12.68433895,
        22.44016657,
        -6.688929,
        14.650579,
        26.80913054,
        -0.66291905,
        44.83408305,
    ],
    "PANEL_1X4": [
        9671.24274959,
        -0.01824475,
        863.88612235,
        -256.81427705,
        -254.7114012,
        84.32208336,
        132.56816823,
        228.82704306,
        3.02697987,
        44.49743184,
        68.81917821,
        -70.2351017,
        -110.52189962,
        86.6653505,
        83.00126997,
        -83.70155672,
        73.70335649,
        0.47835048,
        -2.99527223,
        2.41005885,
        96.55382833,
        67.88596521,
        -10.00983249,
        48.78487225,
        55.36173457,
        34.18931627,
        71.66235718,
        -45.12327998,
        -2.91449919,
        -14.96982611,
    ],
    "PANEL_2X2": [
        9698.52429871,
        -0.00812866,
        505.34344238,
        -533.39124587,
        167.45107901,
        390.8844995,
        -65.03743344,
        -105.14088561,
        112.3092373,
        39.69314332,
        -71.13170706,
        51.6920886,
        -13.56441808,
        44.83573026,
        22.8410518,
        87.84092004,
        57.50726581,
        -10.63603389,
        78.3495127,
        63.42849949,
        3.83939661,
        -7.85677056,
        28.56402805,
        -54.82364222,
        18.65126226,
        58.88806703,
        26.21472892,
        -3.626236,
        31.16357173,
        2.71863917,
    ],
    "PANEL_2X4": [
        10330.25211321,
        0.06234347,
        74.48041559,
        34.20404992,
        -97.90020182,
        -34.53962036,
        -233.81089253,
        -108.80190069,
        108.98144984,
        99.17890646,
        -22.38130067,
        -45.70851712,
        -24.09631258,
        -73.25290349,
        -50.03549065,
        -4.07564116,
        85.31393284,
        63.96820424,
        -44.21996017,
        4.76028071,
        18.66924476,
        -20.22583379,
        20.67420325,
        -43.25955688,
        1.98993053,
        2.03846967,
        7.28066219,
        -65.14066723,
        56.70207836,
        78.34586648,
    ],
    "PANEL_4X4": [
        10354.38811289,
        -0.03171231,
        -461.09656112,
        142.47548513,
        89.53984799,
        -388.70066631,
        80.65260209,
        196.64862897,
        -57.42734504,
        147.61756922,
        22.79588341,
        -125.22087073,
        -71.41905932,
        -68.16795934,
        -18.74802832,
        25.17058276,
        -105.24886336,
        -17.96693185,
        6.56688092,
        -54.12496181,
        -44.15765128,
        -17.86748796,
        -5.67160164,
        15.72954554,
        -5.05864628,
        -49.2393477,
        -6.73443743,
        -15.38578809,
        -3.91435042,
        -7.87079528,
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Robot config
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_PRODUCTS = [
    "ROBOT_DISHES",
    "ROBOT_IRONING",
    "ROBOT_LAUNDRY",
    "ROBOT_MOPPING",
    "ROBOT_VACUUMING",
]

ROBOT_FOURIER_K = 16
ROBOT_FOURIER_PERIOD = 30000.0
ROBOT_FOURIER_COEF = {
    "ROBOT_DISHES": [9377.68368304, 0.04271017, -60.41857323, 52.08442184, -249.80158389, 20.43988204, 18.29671294, 87.5386236, -55.75890609, 154.5202718, -78.75330613, -11.59199884, -46.57887368, 74.85496095, 34.85167138, 6.72303463, -21.24407218, 75.7912632, 26.48714476, 11.34371542, -40.83435075, 90.48125783, 9.22498328, -0.35399193, 31.83571922, 49.12931129, -46.13690439, -38.64678555, 3.8034582, -61.49362409, 53.82727158, -15.72947688, -7.40743744, 15.1831036],
    "ROBOT_IRONING": [9846.31044201, -0.07631856, 95.92745308, -196.73353382, -308.08024716, 156.54857405, 141.01178894, 67.093124, 124.67745882, 195.71739451, -0.35711726, -29.00923606, -14.69363541, -8.93486469, 75.63609057, -30.27614314, -3.94879417, -2.99949074, -16.67878413, -75.16123275, -5.04192761, 10.64636645, -26.08150254, 27.27991438, 14.7011355, 10.18671453, 19.70571162, 0.63484301, -36.64789429, -0.68670427, 26.36997436, 9.58491789, 1.50179301, -10.20385525],
    "ROBOT_LAUNDRY": [10283.33927768, -0.0307062, 432.43486156, -294.43408647, 67.25787888, 125.37661343, -101.85379475, -67.71016682, -49.14053047, -99.75979126, -43.20161653, 154.49502424, 3.03024298, 25.50406757, 12.7908668, 10.47527502, 9.16796799, -17.00594505, 11.01003744, 2.48025414, -19.46991759, -7.56485631, 22.02867365, -3.13841652, -73.84449946, 31.67617824, -55.97615802, -44.77922481, -19.03129366, 18.67141435, -10.27809768, -17.2398076, -28.09437167, 15.54871299],
    "ROBOT_MOPPING": [10426.69253767, 0.04490288, -486.42594605, 11.19118021, 394.98624841, -196.0090258, -263.62794026, -77.50192038, 159.95430707, 40.65943523, 118.06711158, 3.48460031, 17.72379453, -110.81849881, -65.25935218, -5.82331443, -14.31856675, -35.84144428, 24.70623885, 20.66695023, 102.90471409, -82.05861243, 29.69545902, 45.44360286, 42.02765709, 5.69329629, 25.80677528, 45.43873356, 18.31217102, 32.88691206, -11.76129613, 3.59187894, -22.03582813, -1.93125497],
    "ROBOT_VACUUMING": [10066.89002118, -0.06000962, 93.40208942, 11.78735581, -182.34985357, -111.88125848, -98.50712051, 7.40157884, -125.0239882, -23.65204391, -89.22858115, 59.96669583, -53.51773829, 65.30978023, 21.35930929, -31.4834729, -112.27695316, -7.52922733, 3.74849658, 13.81638975, -19.78540796, 17.36440997, 46.3322823, 41.98464962, 37.4958869, -6.47746676, 3.55451368, 45.48737118, -28.35530749, -47.87036219, -1.69812631, 1.17549435, 27.78383348, -4.46342211],
}