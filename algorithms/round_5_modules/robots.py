from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import Any, Dict, List
import json
import math


# ======================================================================
# VISUALIZER LOGGER
# ======================================================================

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
            compressed.append(
                [
                    listing.symbol,
                    listing.product,
                    listing.denomination,
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
        conversion_observations = {}

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
            observations.plainValueObservations,
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
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if max_length <= 0:
            return ""

        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


# ======================================================================
# ROUND 5 — DOMESTIC ROBOTS SIGNAL STRATEGY + SAFE PASSIVE OVERLAY
# ======================================================================
#
# Equations:
#
#   S1 = ROBOT_VACUUMING + 2*ROBOT_DISHES + 2*ROBOT_MOPPING
#   S2 = 3*ROBOT_MOPPING + ROBOT_IRONING + 2*ROBOT_LAUNDRY
#
# Important correction:
#   Do NOT trade these as exact ETF baskets aggressively.
#   The synthetic spread is too expensive and fill tracking becomes unstable.
#
# Instead:
#   1. Use S1 and S2 as mean-reversion signals.
#   2. Trade the individual robot legs that historically respond best.
#   3. Use extra spare capacity only as conservative passive overlay.
#
# ======================================================================


ROBOT_VACUUMING = "ROBOT_VACUUMING"
ROBOT_MOPPING = "ROBOT_MOPPING"
ROBOT_DISHES = "ROBOT_DISHES"
ROBOT_LAUNDRY = "ROBOT_LAUNDRY"
ROBOT_IRONING = "ROBOT_IRONING"

PRODUCTS = [
    ROBOT_VACUUMING,
    ROBOT_MOPPING,
    ROBOT_DISHES,
    ROBOT_LAUNDRY,
    ROBOT_IRONING,
]

POSITION_LIMIT = 10
MAX_HISTORY = 1100

# This should reproduce the old profitable structure.
ENABLE_AGGRESSIVE_SIGNAL_TRADES = True

# Safe improvement from the UV-style idea.
ENABLE_PASSIVE_OVERLAY = True

# Do not increase exposure aggressively if the spread is too wide.
MAX_TAKE_SPREAD = 10


class Trader:

    # ==============================================================
    # Data helpers
    # ==============================================================

    def empty_data(self) -> Dict[str, Any]:
        return {
            "s1": [],
            "s2": [],
        }

    def load_data(self, trader_data: str) -> Dict[str, Any]:
        if trader_data:
            try:
                data = json.loads(trader_data)
            except Exception:
                data = self.empty_data()
        else:
            data = self.empty_data()

        if "s1" not in data:
            data["s1"] = []

        if "s2" not in data:
            data["s2"] = []

        return data

    def get_top(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        bid_volume = depth.buy_orders[best_bid]
        ask_volume = -depth.sell_orders[best_ask]

        return best_bid, best_ask, bid_volume, ask_volume

    def get_mid(self, depth: OrderDepth):
        top = self.get_top(depth)

        if top is None:
            return None

        best_bid, best_ask, _, _ = top
        return (best_bid + best_ask) / 2.0

    # ==============================================================
    # Signal helpers
    # ==============================================================

    def zscore(self, history: List[float], value: float, window: int, min_points: int):
        if len(history) < min_points:
            return None

        recent = history[-window:]

        mean = sum(recent) / len(recent)

        var = 0.0
        for x in recent:
            var += (x - mean) ** 2
        var /= len(recent)

        std = math.sqrt(var)

        if std == 0:
            return None

        return (value - mean) / std

    def target_from_z(
        self,
        z,
        entry_z: float,
        exit_z: float,
        current_pos: int,
    ) -> int:

        if z is None:
            return current_pos

        if abs(z) <= exit_z:
            return 0

        if z > entry_z:
            return -POSITION_LIMIT

        if z < -entry_z:
            return POSITION_LIMIT

        return current_pos

    def side_from_z(self, z, entry_z: float) -> int:
        if z is None:
            return 0

        if z > entry_z:
            return -1

        if z < -entry_z:
            return 1

        return 0

    # ==============================================================
    # Execution helpers
    # ==============================================================

    def add_aggressive_to_target(
        self,
        orders: Dict[Symbol, List[Order]],
        product: str,
        depth: OrderDepth,
        planned_position: Dict[str, int],
        target_pos: int,
    ) -> None:

        top = self.get_top(depth)

        if top is None:
            return

        best_bid, best_ask, bid_volume, ask_volume = top
        spread = best_ask - best_bid

        current_pos = planned_position[product]
        diff = target_pos - current_pos

        if diff == 0:
            return

        # If spread is very wide, do not increase exposure.
        # But still allow reducing/flattening exposure.
        increasing_exposure = abs(target_pos) > abs(current_pos)

        if increasing_exposure and spread > MAX_TAKE_SPREAD:
            return

        if diff > 0:
            capacity = POSITION_LIMIT - planned_position[product]
            qty = min(diff, ask_volume, capacity)

            if qty > 0:
                orders[product].append(Order(product, best_ask, qty))
                planned_position[product] += qty

        else:
            capacity = POSITION_LIMIT + planned_position[product]
            qty = min(-diff, bid_volume, capacity)

            if qty > 0:
                orders[product].append(Order(product, best_bid, -qty))
                planned_position[product] -= qty

    def add_passive_overlay(
        self,
        orders: Dict[Symbol, List[Order]],
        product: str,
        depth: OrderDepth,
        planned_position: Dict[str, int],
        side: int,
        max_qty: int,
    ) -> None:

        if not ENABLE_PASSIVE_OVERLAY:
            return

        if side == 0:
            return

        # Important: do not mix aggressive target order and overlay order
        # for the same product in the same timestamp.
        if len(orders[product]) > 0:
            return

        top = self.get_top(depth)

        if top is None:
            return

        best_bid, best_ask, _, _ = top
        spread = best_ask - best_bid

        # Need at least one tick inside the spread.
        if spread <= 1:
            return

        if side > 0:
            price = best_bid + 1

            if price >= best_ask:
                return

            capacity = POSITION_LIMIT - planned_position[product]
            qty = min(max_qty, capacity)

            if qty > 0:
                orders[product].append(Order(product, price, qty))
                planned_position[product] += qty

        else:
            price = best_ask - 1

            if price <= best_bid:
                return

            capacity = POSITION_LIMIT + planned_position[product]
            qty = min(max_qty, capacity)

            if qty > 0:
                orders[product].append(Order(product, price, -qty))
                planned_position[product] -= qty

    # ==============================================================
    # Main run
    # ==============================================================

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        for product in PRODUCTS:
            result[product] = []

        data = self.load_data(state.traderData)

        mids: Dict[str, float] = {}

        for product in PRODUCTS:
            if product not in state.order_depths:
                trader_data = json.dumps(data)
                logger.print("missing", product)
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data

            mid = self.get_mid(state.order_depths[product])

            if mid is None:
                trader_data = json.dumps(data)
                logger.print("empty book", product)
                logger.flush(state, result, conversions, trader_data)
                return result, conversions, trader_data

            mids[product] = mid

        # ----------------------------------------------------------
        # Basket signals.
        # ----------------------------------------------------------
        s1 = (
            mids[ROBOT_VACUUMING]
            + 2 * mids[ROBOT_DISHES]
            + 2 * mids[ROBOT_MOPPING]
        )

        s2 = (
            3 * mids[ROBOT_MOPPING]
            + mids[ROBOT_IRONING]
            + 2 * mids[ROBOT_LAUNDRY]
        )

        s1_hist = data["s1"]
        s2_hist = data["s2"]

        z1_200 = self.zscore(s1_hist, s1, window=200, min_points=100)
        z1_500 = self.zscore(s1_hist, s1, window=500, min_points=200)

        z2_200 = self.zscore(s2_hist, s2, window=200, min_points=100)
        z2_1000 = self.zscore(s2_hist, s2, window=1000, min_points=200)

        planned_position: Dict[str, int] = {}

        for product in PRODUCTS:
            planned_position[product] = state.position.get(product, 0)

        # ----------------------------------------------------------
        # Original profitable aggressive signal layer.
        # These are not exact-ratio ETF trades.
        # They are individual products traded from basket signals.
        # ----------------------------------------------------------
        if ENABLE_AGGRESSIVE_SIGNAL_TRADES:

            aggressive_rules = [
                # Product, z, entry, exit
                (ROBOT_DISHES, z1_200, 2.00, 0.25),
                (ROBOT_MOPPING, z1_500, 1.00, 0.00),
                (ROBOT_VACUUMING, z1_500, 2.00, 0.00),

                (ROBOT_LAUNDRY, z2_200, 2.50, 0.00),
                (ROBOT_IRONING, z2_1000, 1.50, 0.00),
            ]

            for product, z, entry_z, exit_z in aggressive_rules:
                current_pos = planned_position[product]

                target_pos = self.target_from_z(
                    z,
                    entry_z,
                    exit_z,
                    current_pos,
                )

                self.add_aggressive_to_target(
                    result,
                    product,
                    state.order_depths[product],
                    planned_position,
                    target_pos,
                )

        # ----------------------------------------------------------
        # Safe passive overlay using spare-capacity idea.
        #
        # Conservative:
        #   - only spare/non-binding legs
        #   - only passive inside-spread quoting
        #   - small size
        #   - skipped if product already had aggressive order this tick
        # ----------------------------------------------------------

        # S1 spare leg: VACUUMING.
        # Use lower threshold than aggressive VACUUMING entry.
        s1_overlay_side = self.side_from_z(z1_500, 0.85)

        self.add_passive_overlay(
            result,
            ROBOT_VACUUMING,
            state.order_depths[ROBOT_VACUUMING],
            planned_position,
            s1_overlay_side,
            max_qty=2,
        )

        # S2 spare legs: IRONING and LAUNDRY.
        # Use different z windows matching their old successful signals.
        ironing_overlay_side = self.side_from_z(z2_1000, 1.00)
        laundry_overlay_side = self.side_from_z(z2_200, 1.25)

        self.add_passive_overlay(
            result,
            ROBOT_IRONING,
            state.order_depths[ROBOT_IRONING],
            planned_position,
            ironing_overlay_side,
            max_qty=2,
        )

        self.add_passive_overlay(
            result,
            ROBOT_LAUNDRY,
            state.order_depths[ROBOT_LAUNDRY],
            planned_position,
            laundry_overlay_side,
            max_qty=2,
        )

        # ----------------------------------------------------------
        # Update history after signal calculation.
        # ----------------------------------------------------------
        s1_hist.append(s1)
        s2_hist.append(s2)

        data["s1"] = s1_hist[-MAX_HISTORY:]
        data["s2"] = s2_hist[-MAX_HISTORY:]

        trader_data = json.dumps(data)

        logger.print(
            "t", state.timestamp,
            "S1", round(s1, 2),
            "z1_200", None if z1_200 is None else round(z1_200, 3),
            "z1_500", None if z1_500 is None else round(z1_500, 3),
            "S2", round(s2, 2),
            "z2_200", None if z2_200 is None else round(z2_200, 3),
            "z2_1000", None if z2_1000 is None else round(z2_1000, 3),
            "pos", {p: state.position.get(p, 0) for p in PRODUCTS},
            "planned", planned_position,
        )

        logger.print("orders", result)

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data