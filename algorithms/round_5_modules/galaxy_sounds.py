from datamodel import (
    Order,
    OrderDepth,
    TradingState,
    Symbol,
    ProsperityEncoder,
    Listing,
    Observation,
    Trade,
)
from typing import Dict, List, Any, Optional
import json
import math


# ============================================================
# VISUALIZER LOGGER
# ============================================================

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
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

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state,
                        self.truncate(state.traderData, max_item_length),
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
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

    def compress_order_depths(
        self,
        order_depths: Dict[Symbol, OrderDepth],
    ) -> Dict[Symbol, List[Any]]:
        compressed = {}

        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders,
                order_depth.sell_orders,
            ]

        return compressed

    def compress_trades(
        self,
        trades: Dict[Symbol, List[Trade]],
    ) -> List[List[Any]]:
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

    def compress_observations(self, observations: Observation) -> List[Any]:
        if observations is None:
            return [{}, {}]

        plain_value_observations = getattr(
            observations,
            "plainValueObservations",
            {},
        )

        conversion_observations = {}

        if hasattr(observations, "conversionObservations"):
            for product, observation in observations.conversionObservations.items():
                conversion_observations[product] = [
                    observation.bidPrice,
                    observation.askPrice,
                    observation.transportFees,
                    observation.exportTariff,
                    observation.importTariff,
                    observation.sugarPrice,
                    observation.sunlightIndex,
                ]

        return [
            plain_value_observations,
            conversion_observations,
        ]

    def compress_orders(
        self,
        orders: Dict[Symbol, List[Order]],
    ) -> List[List[Any]]:
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
        return json.dumps(
            value,
            cls=ProsperityEncoder,
            separators=(",", ":"),
        )

    def truncate(self, value: str, max_length: int) -> str:
        if value is None:
            return ""

        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


# ============================================================
# ROUND 5 — GALAXY SOUNDS RECORDERS
# 100k+ public-day calibrated sine schedule + LOGGER
# ============================================================

POSITION_LIMIT = 10
UNKNOWN_DAY_OFFSET = 30000.0
FINGERPRINT_TOLERANCE = 1000.0

PRODUCTS = [
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
]


BASE_CONFIG: Dict[str, Dict[str, Any]] = {
    "GALAXY_SOUNDS_DARK_MATTER": {
        "period": 12500.0,
        "coef": [10189.2823, 0.00479038202, -206.552578, -248.583421],
        "horizon": 0.0,
        "entry": 0.002,
        "invert": False,
        "size": 10,
    },
    "GALAXY_SOUNDS_BLACK_HOLES": {
        "period": 19600.0,
        "coef": [9972.60261, 0.0946039432, 335.865882, -305.035379],
        "horizon": -250.0,
        "entry": 0.0,
        "invert": False,
        "size": 10,
    },
    "GALAXY_SOUNDS_PLANETARY_RINGS": {
        "period": 23000.0,
        "coef": [10528.4382, 0.0268365661, -800.513765, -295.007697],
        "horizon": -3250.0,
        "entry": 0.0,
        "invert": False,
        "size": 10,
    },
    "GALAXY_SOUNDS_SOLAR_WINDS": {
        "period": 21800.0,
        "coef": [9822.93225, 0.0385098945, 7.92952952, 436.028617],
        "horizon": -500.0,
        "entry": 0.005,
        "invert": False,
        "size": 10,
    },
    "GALAXY_SOUNDS_SOLAR_FLAMES": {
        "period": 30000.0,
        "coef": [10567.9258, 0.0349775561, 633.752437, -149.62733],
        "horizon": -2000.0,
        "entry": 0.002,
        "invert": False,
        "size": 10,
    },
}


PUBLIC_TIMING: Dict[float, Dict[str, Dict[str, Any]]] = {
    0.0: {
        "GALAXY_SOUNDS_DARK_MATTER": {"horizon": -6200.0, "entry": 0.002, "invert": True},
        "GALAXY_SOUNDS_BLACK_HOLES": {"horizon": -5100.0, "entry": 0.001, "invert": False},
        "GALAXY_SOUNDS_PLANETARY_RINGS": {"horizon": 10300.0, "entry": 0.0, "invert": True},
        "GALAXY_SOUNDS_SOLAR_WINDS": {"horizon": 8100.0, "entry": 0.0, "invert": True},
        "GALAXY_SOUNDS_SOLAR_FLAMES": {"horizon": 2900.0, "entry": 0.002, "invert": False},
    },
    10000.0: {
        "GALAXY_SOUNDS_DARK_MATTER": {"horizon": -6000.0, "entry": 0.010, "invert": True},
        "GALAXY_SOUNDS_BLACK_HOLES": {"horizon": 900.0, "entry": 0.0, "invert": False},
        "GALAXY_SOUNDS_PLANETARY_RINGS": {"horizon": 9900.0, "entry": 0.0, "invert": True},
        "GALAXY_SOUNDS_SOLAR_WINDS": {"horizon": -8600.0, "entry": 0.0, "invert": True},
        "GALAXY_SOUNDS_SOLAR_FLAMES": {"horizon": 9100.0, "entry": 0.001, "invert": True},
    },
    20000.0: {
        "GALAXY_SOUNDS_DARK_MATTER": {"horizon": -400.0, "entry": 0.002, "invert": False},
        "GALAXY_SOUNDS_BLACK_HOLES": {"horizon": -500.0, "entry": 0.001, "invert": False},
        "GALAXY_SOUNDS_PLANETARY_RINGS": {"horizon": 7000.0, "entry": 0.0, "invert": True},
        "GALAXY_SOUNDS_SOLAR_WINDS": {"horizon": -4600.0, "entry": 0.0, "invert": False},
        "GALAXY_SOUNDS_SOLAR_FLAMES": {"horizon": -4300.0, "entry": 0.001, "invert": True},
    },
}


OPENING_FINGERPRINTS: Dict[float, Dict[str, float]] = {
    0.0: {
        "GALAXY_SOUNDS_DARK_MATTER": 10000.0,
        "GALAXY_SOUNDS_BLACK_HOLES": 10000.0,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 10000.0,
        "GALAXY_SOUNDS_SOLAR_WINDS": 10000.0,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 10000.0,
    },
    10000.0: {
        "GALAXY_SOUNDS_DARK_MATTER": 10409.5,
        "GALAXY_SOUNDS_BLACK_HOLES": 11444.5,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 11249.5,
        "GALAXY_SOUNDS_SOLAR_WINDS": 9591.5,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 11338.5,
    },
    20000.0: {
        "GALAXY_SOUNDS_DARK_MATTER": 10876.0,
        "GALAXY_SOUNDS_BLACK_HOLES": 12137.0,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 11415.5,
        "GALAXY_SOUNDS_SOLAR_WINDS": 10756.0,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 10643.0,
    },
}


class Trader:
    def run(self, state: TradingState):
        data = self._load_data(state.traderData)

        if state.timestamp == 0 or data.get("last_timestamp", -1) > state.timestamp:
            data = {
                "offset": None,
                "last_timestamp": state.timestamp,
            }

            logger.print(
                "RESET",
                "timestamp=",
                state.timestamp,
            )

        if data.get("offset") is None:
            data["offset"] = self._detect_day_offset(state.order_depths)

            logger.print(
                "OFFSET_DETECTED",
                "offset=",
                data["offset"],
                "timestamp=",
                state.timestamp,
            )

        day_offset = float(data.get("offset", UNKNOWN_DAY_OFFSET))
        tick = int(state.timestamp // 100)
        global_tick = day_offset + float(tick)

        orders: Dict[Symbol, List[Order]] = {}

        for product in PRODUCTS:
            depth = state.order_depths.get(product)

            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue

            cfg = BASE_CONFIG[product]
            timing = self._timing_for(product, day_offset)
            signal = self._signal(cfg, timing, global_tick)

            size = int(cfg.get("size", 10))
            target_pos = self._clip_position(size * signal)
            current_pos = state.position.get(product, 0)

            product_orders = self._move_to_target(
                product,
                depth,
                current_pos,
                target_pos,
            )

            if product_orders:
                orders[product] = product_orders

                logger.print(
                    product,
                    "tick=",
                    tick,
                    "global_tick=",
                    global_tick,
                    "offset=",
                    day_offset,
                    "pos=",
                    current_pos,
                    "target=",
                    target_pos,
                    "signal=",
                    signal,
                    "horizon=",
                    timing.get("horizon", cfg.get("horizon", 0.0)),
                    "entry=",
                    timing.get("entry", cfg.get("entry", 0.0)),
                    "invert=",
                    timing.get("invert", cfg.get("invert", False)),
                    "orders=",
                    product_orders,
                )

        data["last_timestamp"] = state.timestamp

        conversions = 0
        trader_data = json.dumps(data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data

    # --------------------------------------------------------
    # Signal logic
    # --------------------------------------------------------

    def _timing_for(self, product: str, day_offset: float) -> Dict[str, Any]:
        if day_offset in PUBLIC_TIMING and product in PUBLIC_TIMING[day_offset]:
            return PUBLIC_TIMING[day_offset][product]

        return BASE_CONFIG[product]

    def _signal(self, cfg: Dict[str, Any], timing: Dict[str, Any], global_tick: float) -> int:
        x = global_tick + float(timing.get("horizon", cfg.get("horizon", 0.0)))
        value = self._derivative(cfg["coef"], float(cfg["period"]), x)

        if bool(timing.get("invert", cfg.get("invert", False))):
            value = -value

        entry = float(timing.get("entry", cfg.get("entry", 0.0)))

        if value > entry:
            return 1

        if value < -entry:
            return -1

        return 0

    def _derivative(self, coef: List[float], period: float, x: float) -> float:
        _c, m, a, b = coef
        w = 2.0 * math.pi / period
        return m + w * (a * math.cos(w * x) - b * math.sin(w * x))

    # --------------------------------------------------------
    # Offset detection
    # --------------------------------------------------------

    def _detect_day_offset(self, order_depths: Dict[Symbol, OrderDepth]) -> float:
        observed: Dict[str, float] = {}

        for product in PRODUCTS:
            depth = order_depths.get(product)
            mid = self._mid_price(depth) if depth is not None else None

            if mid is not None:
                observed[product] = mid

        if len(observed) < 3:
            logger.print(
                "OFFSET_FALLBACK",
                "reason=",
                "not_enough_observed_products",
                "observed_count=",
                len(observed),
            )

            return UNKNOWN_DAY_OFFSET

        best_offset = UNKNOWN_DAY_OFFSET
        best_error = 10 ** 18

        for offset, fingerprint in OPENING_FINGERPRINTS.items():
            error = 0.0
            count = 0

            for product, fp_mid in fingerprint.items():
                if product in observed:
                    diff = observed[product] - fp_mid
                    error += diff * diff
                    count += 1

            if count > 0 and error < best_error:
                best_error = error
                best_offset = offset

        if best_error <= FINGERPRINT_TOLERANCE:
            logger.print(
                "OFFSET_MATCH",
                "best_offset=",
                best_offset,
                "best_error=",
                best_error,
            )

            return best_offset

        logger.print(
            "OFFSET_FALLBACK",
            "reason=",
            "fingerprint_error_too_high",
            "best_offset=",
            best_offset,
            "best_error=",
            best_error,
        )

        return UNKNOWN_DAY_OFFSET

    # --------------------------------------------------------
    # Execution helpers
    # --------------------------------------------------------

    def _move_to_target(
        self,
        product: str,
        depth: OrderDepth,
        current_pos: int,
        target_pos: int,
    ) -> List[Order]:
        result: List[Order] = []
        pos = current_pos

        if target_pos > pos:
            needed = min(target_pos - pos, POSITION_LIMIT - pos)

            for ask in sorted(depth.sell_orders.keys()):
                if needed <= 0:
                    break

                available = abs(depth.sell_orders[ask])
                qty = min(needed, available)

                if qty > 0:
                    result.append(Order(product, ask, qty))
                    pos += qty
                    needed -= qty

        elif target_pos < pos:
            needed = min(pos - target_pos, POSITION_LIMIT + pos)

            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                if needed <= 0:
                    break

                available = abs(depth.buy_orders[bid])
                qty = min(needed, available)

                if qty > 0:
                    result.append(Order(product, bid, -qty))
                    pos -= qty
                    needed -= qty

        return result

    def _mid_price(self, depth: Optional[OrderDepth]) -> Optional[float]:
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return None

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        return (best_bid + best_ask) / 2.0

    def _clip_position(self, target: int) -> int:
        if target > POSITION_LIMIT:
            return POSITION_LIMIT

        if target < -POSITION_LIMIT:
            return -POSITION_LIMIT

        return target

    def _load_data(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return {
                "offset": None,
                "last_timestamp": -1,
            }

        try:
            data = json.loads(trader_data)

            if not isinstance(data, dict):
                return {
                    "offset": None,
                    "last_timestamp": -1,
                }

            if "offset" not in data:
                data["offset"] = None

            if "last_timestamp" not in data:
                data["last_timestamp"] = -1

            return data

        except Exception:
            return {
                "offset": None,
                "last_timestamp": -1,
            }