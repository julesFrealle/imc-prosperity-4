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

        plain_observations = {}

        if hasattr(observations, "plainValueObservations"):
            plain_observations = observations.plainValueObservations

        return [
            plain_observations,
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
# ROUND 5 — SLEEP POD REGIME STRATEGY
# ======================================================================
#
# Products:
#   SLEEP_POD_COTTON
#   SLEEP_POD_LAMB_WOOL
#   SLEEP_POD_NYLON
#   SLEEP_POD_POLYESTER
#   SLEEP_POD_SUEDE
#
# Main improvement over the old strategy:
#
#   The previous strategy only traded:
#       POLYESTER, SUEDE, COTTON
#
#   It ignored:
#       LAMB_WOOL, NYLON
#
#   From public days 2-4, the stronger edge is not residual market making.
#   The stronger edge is product-specific one-time regime detection:
#
#       POLYESTER: reversal after early 50k move
#       COTTON:    reversal after 200k move, only if move is large
#       LAMB_WOOL: reversal after 150k move
#       NYLON:     continuation after 200k move
#       SUEDE:     continuation after 350k move
#
#   Each product stores its opening mid, waits until its rule time, then
#   decides one target position and holds it.
#
# ======================================================================


class Trader:
    POSITION_LIMIT = 10

    COTTON = "SLEEP_POD_COTTON"
    LAMB_WOOL = "SLEEP_POD_LAMB_WOOL"
    NYLON = "SLEEP_POD_NYLON"
    POLYESTER = "SLEEP_POD_POLYESTER"
    SUEDE = "SLEEP_POD_SUEDE"

    PRODUCTS = [
        COTTON,
        LAMB_WOOL,
        NYLON,
        POLYESTER,
        SUEDE,
    ]

    RULES = {
        POLYESTER: {
            "time": 50000,
            "threshold": 20,
            "mode": "reversal",
        },
        COTTON: {
            "time": 200000,
            "threshold": 200,
            "mode": "reversal",
        },
        LAMB_WOOL: {
            "time": 150000,
            "threshold": 100,
            "mode": "reversal",
        },
        NYLON: {
            "time": 200000,
            "threshold": 100,
            "mode": "continuation",
        },
        SUEDE: {
            "time": 350000,
            "threshold": 100,
            "mode": "continuation",
        },
    }

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        return (best_bid + best_ask) / 2

    def clamp_position(self, position: int) -> int:
        return max(-self.POSITION_LIMIT, min(self.POSITION_LIMIT, position))

    def aggressive_trade_to_target(
        self,
        product: str,
        order_depth: OrderDepth,
        current_position: int,
        target_position: int,
    ) -> List[Order]:
        orders: List[Order] = []

        target_position = self.clamp_position(target_position)
        diff = target_position - current_position

        if diff > 0:
            remaining = diff

            for ask_price in sorted(order_depth.sell_orders.keys()):
                if remaining <= 0:
                    break

                available = -order_depth.sell_orders[ask_price]
                quantity = min(remaining, available)

                if quantity > 0:
                    orders.append(Order(product, ask_price, quantity))
                    remaining -= quantity

        elif diff < 0:
            remaining = -diff

            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining <= 0:
                    break

                available = order_depth.buy_orders[bid_price]
                quantity = min(remaining, available)

                if quantity > 0:
                    orders.append(Order(product, bid_price, -quantity))
                    remaining -= quantity

        return orders

    def load_trader_data(self, state: TradingState) -> Dict[str, Any]:
        if state.traderData == "":
            return {
                "open_mids": {},
                "targets": {},
                "decision_done": {},
            }

        data = json.loads(state.traderData)

        if "open_mids" not in data:
            data["open_mids"] = {}

        if "targets" not in data:
            data["targets"] = {}

        if "decision_done" not in data:
            data["decision_done"] = {}

        return data

    def get_target_from_rule(
        self,
        open_mid: float,
        current_mid: float,
        rule: Dict[str, Any],
    ) -> int:
        move = current_mid - open_mid
        threshold = rule["threshold"]
        mode = rule["mode"]

        if abs(move) <= threshold:
            return 0

        direction = 1 if move > 0 else -1

        if mode == "reversal":
            direction = -direction

        return self.clamp_position(direction * self.POSITION_LIMIT)

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        data = self.load_trader_data(state)

        # --------------------------------------------------------------
        # 1. Store opening mids.
        # --------------------------------------------------------------

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue

            if product not in data["open_mids"]:
                mid = self.get_mid_price(state.order_depths[product])

                if mid is not None:
                    data["open_mids"][product] = mid
                    logger.print("OPEN_MID", product, mid)

        # --------------------------------------------------------------
        # 2. Make one-time decision for each product at its own rule time.
        # --------------------------------------------------------------

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue

            if data["decision_done"].get(product, False):
                continue

            if product not in data["open_mids"]:
                continue

            rule = self.RULES[product]

            if state.timestamp < rule["time"]:
                continue

            current_mid = self.get_mid_price(state.order_depths[product])

            if current_mid is None:
                continue

            open_mid = data["open_mids"][product]

            target = self.get_target_from_rule(
                open_mid,
                current_mid,
                rule,
            )

            data["targets"][product] = target
            data["decision_done"][product] = True

            logger.print(
                "SLEEP_DECISION",
                product,
                "time",
                state.timestamp,
                "mode",
                rule["mode"],
                "open",
                open_mid,
                "current",
                current_mid,
                "move",
                current_mid - open_mid,
                "threshold",
                rule["threshold"],
                "target",
                target,
            )

        # --------------------------------------------------------------
        # 3. Trade aggressively toward stored target positions.
        # --------------------------------------------------------------

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue

            target = int(data["targets"].get(product, 0))
            current_position = state.position.get(product, 0)
            order_depth = state.order_depths[product]

            product_orders = self.aggressive_trade_to_target(
                product,
                order_depth,
                current_position,
                target,
            )

            if len(product_orders) > 0:
                result[product] = product_orders

        trader_data = json.dumps(data, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data