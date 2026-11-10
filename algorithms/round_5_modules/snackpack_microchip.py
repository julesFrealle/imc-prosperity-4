

from datamodel import Order, OrderDepth, TradingState, ProsperityEncoder
from typing import Dict, List, Tuple, Any
import json
import math


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

# ── SNACKPACK config ─────────────────────────────────────────────────────────
SNPK_RB_LIMIT = 10

# Fixed fair values calibrated from full-round data (means are stable)
SNPK_FAIR = {
    "SNACKPACK_VANILLA":   10097,
    "SNACKPACK_RASPBERRY": 10080,
}
SNPK_ENTRY_THRESH_VANILLA= 130   # aggressive MR entry when |mid - fair| > this
SNPK_ENTRY_THRESH_RASPBERRY= 150   # aggressive MR entry when |mid - fair| > this
SNPK_EXIT_THRESH_VANILLA= 20    # unwind when price reverts to within this of fair
SNPK_EXIT_THRESH_RASPBERRY= 30    # unwind when price reverts to within this of fair
SNPK_MM_MAX_QTY   = 4     # max qty per MM quote (active whenever sig == 0 and spread >= 2)

# Leverage basket per main product
CLUSTER_SNACKPACK_VANILLA    = ["SNACKPACK_CHOCOLATE"]
CLUSTER_SNACKPACK_RASPBERRY  = ["SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY"]

# ── PEBBLES config ───────────────────────────────────────────────────────────
# Background agent trades all 5 simultaneously, same qty, 50/50 buy/sell,
# always at L1. Spreads are 8-17 ticks wide every tick → pure MM edge.
# The basket sum is tightly mean-reverting around 50_000, so we use that
# basket fair to skew quotes without crossing the spread.
#
# Inventory skew:
#   |pos| <= PEBBLES_SOFT_CAP : symmetric quotes at bid+1 / ask-1
#   |pos| >  PEBBLES_SOFT_CAP : shift BOTH quotes 1 tick toward the flat side
#                                 (reservation-price skew, Avellaneda-Stoikov style)
#   |pos| == limit             : quote only the reducing side, still at inside price
#
PEBBLES_PRODUCTS = ["PEBBLES_L", "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_XL"]
PEBBLES_BASKET_FAIR = 50000
PEBBLES_SOFT_CAP = 4   # threshold to start inventory skewing
PEBBLES_BASE_IMPROVE = 1
PEBBLES_MOM_THRESH = 999  # disabled after backtest: trade-side prediction is ~50/50
PEBBLES_MOM_STYLE = {
    "PEBBLES_L": 1,    # mild continuation
    "PEBBLES_S": -1,   # mild mean reversion
    "PEBBLES_XL": 1,   # strongest continuation
}
PEBBLES_MM_CFG = {
    # product: (position_limit, min_spread_to_quote, max_qty_per_side)
    "PEBBLES_L":  (10, 2, 10),
    "PEBBLES_XS": (3, 2, 3),
    "PEBBLES_S":  (10, 2, 10),
    "PEBBLES_M":  (0, 2, 0),
    "PEBBLES_XL": (10, 2, 10),
}

# ── MICROCHIP config ─────────────────────────────────────────────────────────
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
MICROCHIP_FOURIER_K = 16
MICROCHIP_FOURIER_PERIOD = 30000.0
MICROCHIP_FOURIER_COEF = {
    "MICROCHIP_CIRCLE": [9035.00376574, 0.0119925, 109.5600525, 496.24052243, -318.04223847, 244.05647642, -186.10590581, 28.30660336, -16.64568413, 53.40290384, -79.46014674, -3.37231903, 5.51650178, -30.19740652, -102.41704573, -41.56142945, -28.88106698, 50.49247035, -29.13812052, -25.15403522, -16.16563066, 80.41048703, 59.93463965, 37.998571, -1.70455883, -12.73731999, -33.46167366, -8.3466848, 7.01185143, 19.43804258, -6.54852843, 8.305692, -20.01827642, 10.07247853],
    "MICROCHIP_OVAL": [10339.37791523, -0.14399008, 434.66394153, -471.53273139, 7.05492674, -55.33252786, -70.50241089, 69.03650292, 4.54251754, 17.3654677, 172.07305006, -15.35010701, -81.28068789, -63.70160181, -80.12786355, -18.67599362, -40.10533783, 110.79303085, -18.10446368, -30.40413338, 28.24731842, 66.73191961, 24.77221933, -40.56871166, 54.06303177, -1.81261368, 12.29907978, -31.11238571, 6.90011421, -24.41949632, -28.31376957, -56.95767136, -26.54201478, 35.50449526],
    "MICROCHIP_RECTANGLE": [9395.422936, -0.04420038, 308.41996448, 503.84878376, -195.19866431, -98.62884523, 181.56725065, 61.73049968, 219.11920132, -40.59990189, 111.56346162, 31.69184004, -5.41199254, 21.54952447, -17.30960004, 1.44172229, -4.07491229, 61.01819662, -84.31768741, -34.83080157, -37.04865406, 49.1304492, 41.77305262, -32.74726128, -0.17045125, 28.77715414, 19.50764921, -35.72521574, 25.10471875, 23.91229359, 11.94144226, 12.83916452, -49.19344576, -27.08760597],
    "MICROCHIP_SQUARE": [11776.55998901, 0.12121659, -877.77797703, -1124.14733, 121.26212976, 33.31820464, -26.05213641, -313.24002374, -29.53444822, -225.79440407, 59.82422326, 10.2467809, -149.79788043, -20.91705565, -222.34127365, -0.18434991, -103.82492208, 9.85529811, 61.33083644, -68.05385426, -130.75088515, -39.62243378, 94.95179334, -79.01389591, 104.63685286, 132.06666458, -29.53134487, 122.01649085, -13.17068039, -42.45516645, 79.744644, -7.77181883, -2.452156, -116.56147504],
    "MICROCHIP_TRIANGLE": [10725.52946193, -0.0692782, 81.58024978, -567.15380278, 144.99380959, 85.88467977, -83.61559907, -17.70498697, 35.18305422, -166.25860275, -72.80435426, 113.2750059, 25.48067383, -90.89034963, 19.50648521, 39.8219528, -74.23440968, 6.10489744, -19.74312083, 12.20971757, 74.56202029, 20.43962393, -29.67405514, -4.68226481, -14.35006411, -35.057655, 82.2752799, -9.2286653, -56.25062188, -61.21915636, 0.77936663, 19.51581623, 27.21334953, 14.50558556],
}
MICROCHIP_OPENING_FINGERPRINTS = {
    0.0: {
        "MICROCHIP_CIRCLE": 10000.0,
        "MICROCHIP_OVAL": 10000.0,
        "MICROCHIP_RECTANGLE": 10000.0,
        "MICROCHIP_SQUARE": 10000.0,
        "MICROCHIP_TRIANGLE": 10000.0,
    },
    10000.0: {
        "MICROCHIP_CIRCLE": 8887.5,
        "MICROCHIP_OVAL": 9252.5,
        "MICROCHIP_RECTANGLE": 9411.5,
        "MICROCHIP_SQUARE": 12457.5,
        "MICROCHIP_TRIANGLE": 10714.5,
    },
    20000.0: {
        "MICROCHIP_CIRCLE": 8535.0,
        "MICROCHIP_OVAL": 7416.5,
        "MICROCHIP_RECTANGLE": 7830.5,
        "MICROCHIP_SQUARE": 15911.0,
        "MICROCHIP_TRIANGLE": 9020.0,
    },
}

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
# SNACKPACK: MR + Directional MM
# ─────────────────────────────────────────────────────────────────────────────

def _lev_trade(
    leverage_products: List[str],
    direction: int,          # +1 = buy, -1 = sell
    lev_limit: int,
    state: TradingState,
    result: Dict[str, List[Order]],
) -> None:
    """Trade leverage basket in a single direction (hit best available price)."""
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
                logger.print(f"  LEV BUY  {product} {qty}@{best_ask}  pos={pos}")
        else:
            if not od.buy_orders or pos <= -lev_limit:
                continue
            best_bid = max(od.buy_orders)
            qty = min(od.buy_orders[best_bid], lev_limit + pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_bid, -qty))
                logger.print(f"  LEV SELL {product} {qty}@{best_bid}  pos={pos}")


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

    pos      = state.position.get(product, 0)
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread   = best_ask - best_bid
    mid      = (best_bid + best_ask) / 2
    fair     = SNPK_FAIR[product]
    dev      = mid - fair
    sig = int(td.get(f"{key}_sig", 0))

    if   sig ==  0 and dev < -entry_threshold:   sig =  1   # enter long
    elif sig ==  0 and dev >  entry_threshold:   sig = -1   # enter short
    elif sig ==  1 and dev >= -exit_threshold:   sig =  2   # price reverted → unwind
    elif sig == -1 and dev <=  exit_threshold:   sig = -2   # price reverted → unwind
    elif sig ==  2 and pos <= 0:                    sig =  0   # fully unwound
    elif sig == -2 and pos >= 0:                    sig =  0   # fully unwound

    td[f"{key}_sig"] = sig

    logger.print(
        f"{product}  dev={dev:.1f}  fair={fair}  pos={pos}  sig={sig}"
    )

    orders: List[Order] = []
    net_qty = 0

    # ── Aggressive MR entry ───────────────────────────────────────────────
    if sig == 1 and pos < limit:
        qty = min(-od.sell_orders.get(best_ask, 0), limit - pos)
        if qty > 0:
            orders.append(Order(product, best_ask, qty))
            net_qty += qty
            logger.print(f"  ENTRY BUY  {qty}@{best_ask}")
        _lev_trade(leverage_products, -1, lev_limit, state, result)  # short leverage

    elif sig == -1 and pos > -limit:
        qty = min(od.buy_orders.get(best_bid, 0), limit + pos)
        if qty > 0:
            orders.append(Order(product, best_bid, -qty))
            net_qty -= qty
            logger.print(f"  ENTRY SELL {qty}@{best_bid}")
        _lev_trade(leverage_products, +1, lev_limit, state, result)  # long leverage

    # ── Unwind ────────────────────────────────────────────────────────────
    elif sig == 2 and pos > 0:
        qty = min(od.buy_orders.get(best_bid, 0), pos)
        if qty > 0:
            orders.append(Order(product, best_bid, -qty))
            net_qty -= qty
            logger.print(f"  UNWIND SELL {qty}@{best_bid}  pos={pos}")
        _lev_trade(leverage_products, +1, lev_limit, state, result)  # buy back leverage

    elif sig == -2 and pos < 0:
        qty = min(-od.sell_orders.get(best_ask, 0), -pos)
        if qty > 0:
            orders.append(Order(product, best_ask, qty))
            net_qty += qty
            logger.print(f"  UNWIND BUY  {qty}@{best_ask}  pos={pos}")
        _lev_trade(leverage_products, -1, lev_limit, state, result)  # sell back leverage

    # ── Symmetric MM (flat only, any zone below entry threshold) ─────────
    # MR position is not open → free to market-make whenever spread allows.
    if sig == 0 and spread >= 2 and abs(dev) < entry_threshold:
        buy_cap  = limit - pos - max(0, net_qty)
        sell_cap = limit + pos + max(0, -net_qty)
        bid_px   = best_bid + 1
        ask_px   = best_ask - 1
        if bid_px < ask_px:
            bid_qty = min(buy_cap, SNPK_MM_MAX_QTY)
            ask_qty = min(sell_cap, SNPK_MM_MAX_QTY)
            if bid_qty > 0:
                orders.append(Order(product, bid_px, bid_qty))
                logger.print(f"  MM BID  {bid_qty}@{bid_px}  dev={dev:.0f}")
            if ask_qty > 0:
                orders.append(Order(product, ask_px, -ask_qty))
                logger.print(f"  MM ASK  {ask_qty}@{ask_px}  dev={dev:.0f}")

    for o in orders:
        result.setdefault(product, []).append(o)


def pebbles_market_make(state: TradingState, result: Dict[str, List[Order]], td: dict) -> None:
    mids = {}
    for product in PEBBLES_PRODUCTS:
        od = state.order_depths.get(product)
        if od and od.buy_orders and od.sell_orders:
            mids[product] = (max(od.buy_orders) + min(od.sell_orders)) / 2

    basket_dev = sum(mids.values()) - PEBBLES_BASKET_FAIR if len(mids) == len(PEBBLES_PRODUCTS) else 0
    basket_skew = 0
    if basket_dev >2:
        basket_skew = 1     # basket rich: lean sell / back off bids
    elif basket_dev < -2:
        basket_skew = -1    # basket cheap: lean buy / back off asks
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

        # ── Basket-fair + inventory reservation-price skew ────────────────
        # Positive skew pushes both quotes down:
        #   - ask gets more competitive (sell more)
        #   - bid backs off (buy less)
        # Negative skew mirrors this when the basket is below fair / we are short.
        #
        # For each leg:
        #   product_fair = 50000 - sum(other four mids)
        #   fair_dev = mid_i - product_fair
        # which is algebraically the same as sum(all five mids) - 50000.
        # So the signal compares the whole pebble basket to 50k, not one
        # isolated pebble level.
        product_fair = PEBBLES_BASKET_FAIR - sum(
            mids.get(other, 0) for other in PEBBLES_PRODUCTS if other != product
        ) if len(mids) == len(PEBBLES_PRODUCTS) else (best_bid + best_ask) / 2
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
                # Continuation style: up momentum -> favor buying (negative skew).
                # Mean-reversion style mirrors it.
                mom_skew = -PEBBLES_MOM_STYLE[product] * (1 if momentum > 0 else -1)

        skew = max(-2, min(2, fair_skew + inv_skew + mom_skew))

        max_improve = max(0, (spread - 1) // 2)
        bid_improve = max(0, min(max_improve, PEBBLES_BASE_IMPROVE - skew))
        ask_improve = max(0, min(max_improve, PEBBLES_BASE_IMPROVE + skew))

        bid_px = best_bid + bid_improve
        ask_px = best_ask - ask_improve

        if bid_px >= ask_px:
            continue

        # Capacity: naturally hits 0 at the position limit.  The fair gate
        # avoids quoting the wrong side when the 50k basket says the whole
        # pebble basket is already rich/cheap.
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

        # Keep both sides live, but reduce the side opposite the curve direction.
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
        pending.append([
            state.timestamp + MICROCHIP_FLOW_LAG_MS,
            state.timestamp + MICROCHIP_FLOW_LAG_MS + MICROCHIP_FLOW_HOLD_MS,
            direction,
            score,
        ])

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
        spread = best_ask - best_bid
        pos = state.position.get(product, 0)

        # Use the signal direction as a microchip-specific curve bias, but still
        # quote inside the spread to monetize the synchronized background flow.
        if direction > 0:
            qty = min(5, MICROCHIP_LIMIT - pos)
            if qty > 0:
                price = best_ask
                result.setdefault(product, []).append(Order(product, price, qty))
                logger.print(f"{product} MC_FLOW BID {qty}@{price} score={active_score:.0f} pos={pos}")
        else:
            qty = min(5, MICROCHIP_LIMIT + pos)
            if qty > 0:
                price = best_bid
                result.setdefault(product, []).append(Order(product, price, -qty))
                logger.print(f"{product} MC_FLOW ASK {qty}@{price} score={active_score:.0f} pos={pos}")


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

    def offset_error(offset: int) -> float:
        x = float(offset + local_tick)
        error = 0.0
        for product, mid in observed.items():
            fair, _deriv = _microchip_fourier_value_and_derivative(product, x)
            diff = mid - fair
            error += diff * diff
        return error / len(observed)

    # Infer phase from the Fourier model itself rather than hardcoding
    # "round 2/3/4". This preserves the strong common curve while allowing
    # unseen offsets if hidden data is a continuation or phase shift.
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
    coef = MICROCHIP_FOURIER_COEF[product]
    value = coef[0] + coef[1] * x
    deriv = coef[1]
    idx = 2
    for k in range(1, MICROCHIP_FOURIER_K + 1):
        sin_c = coef[idx]
        cos_c = coef[idx + 1]
        idx += 2
        w = 2 * math.pi * k / MICROCHIP_FOURIER_PERIOD
        sx = math.sin(w * x)
        cx = math.cos(w * x)
        value += sin_c * sx + cos_c * cx
        deriv += w * (sin_c * cx - cos_c * sx)
    return value, deriv


def microchip_fourier_strategy(state: TradingState, result: Dict[str, List[Order]], td: dict) -> None:
    offset = _microchip_offset(state, td)
    x = offset + state.timestamp // 100

    for product in MICROCHIP_PRODUCTS:
        od = state.order_depths.get(product)
        if not (od and od.buy_orders and od.sell_orders):
            continue

        fair, deriv = _microchip_fourier_value_and_derivative(product, float(x + 3))
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
                logger.print(f"{product} MC_FOURIER BUY {qty}@{best_ask} sig={signal:.1f} pos={pos}")
        elif target < pos:
            qty = min(5, pos - target, MICROCHIP_LIMIT + pos)
            if qty > 0:
                result.setdefault(product, []).append(Order(product, best_bid, -qty))
                logger.print(f"{product} MC_FOURIER SELL {qty}@{best_bid} sig={signal:.1f} pos={pos}")


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        td = load_trader_data(state.traderData)
        # Isolated Microchip run: keep Snackpack/Pebbles disabled so backtest
        # PnL reflects only the Microchip Fourier strategy.
        microchip_fourier_strategy(state, result, td)
        trader_data = json.dumps(td, separators=(",", ":"))
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data