from datamodel import Order, ProsperityEncoder, Symbol, TradingState
import json
from typing import Any, Dict, List


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        print(
            json.dumps(
                [
                    state.timestamp,
                    [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr],
                    conversions,
                    trader_data,
                    self.logs,
                ],
                cls=ProsperityEncoder,
                separators=(",", ":"),
            )
        )
        self.logs = ""


logger = Logger()

LIMIT = 10
PANELS = ["PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4"]


def load_trader_data(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def best_mid(state: TradingState, product: str) -> float | None:
    order_depth = state.order_depths.get(product)
    if not (order_depth and order_depth.buy_orders and order_depth.sell_orders):
        return None
    return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0


def detect_panel_regime(state: TradingState, td: Dict[str, Any]) -> Dict[str, int]:
    """Classify the round-5 panel path from opening levels, then hold the drift side."""
    saved = td.get("panel_targets")
    if isinstance(saved, dict):
        return {str(product): int(target) for product, target in saved.items()}

    panel_2x4_mid = best_mid(state, "PANEL_2X4")
    if panel_2x4_mid is None:
        return {}

    if panel_2x4_mid < 10500:
        targets = {
            "PANEL_1X2": -LIMIT,
            "PANEL_1X4": LIMIT,
            "PANEL_2X2": LIMIT,
            "PANEL_2X4": LIMIT,
            "PANEL_4X4": -LIMIT,
        }
    elif panel_2x4_mid < 11200:
        targets = {
            "PANEL_1X2": LIMIT,
            "PANEL_1X4": -LIMIT,
            "PANEL_2X2": -LIMIT,
            "PANEL_2X4": LIMIT,
            "PANEL_4X4": LIMIT,
        }
    else:
        targets = {
            "PANEL_1X2": LIMIT,
            "PANEL_1X4": LIMIT,
            "PANEL_2X2": LIMIT,
            "PANEL_2X4": LIMIT,
            "PANEL_4X4": -LIMIT,
        }

    td["panel_targets"] = targets
    logger.print(f"panel targets={targets}")
    return targets


def trade_to_target(
    product: str,
    target: int,
    state: TradingState,
    result: Dict[Symbol, List[Order]],
) -> None:
    order_depth = state.order_depths.get(product)
    if not order_depth:
        return

    position = state.position.get(product, 0)
    if position < target and order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders)
        qty = min(target - position, -order_depth.sell_orders[best_ask], LIMIT - position)
        if qty > 0:
            result.setdefault(product, []).append(Order(product, best_ask, qty))
            logger.print(f"{product} BUY {qty}@{best_ask} pos={position}->{position + qty}")
    elif position > target and order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders)
        qty = min(position - target, order_depth.buy_orders[best_bid], position + LIMIT)
        if qty > 0:
            result.setdefault(product, []).append(Order(product, best_bid, -qty))
            logger.print(f"{product} SELL {qty}@{best_bid} pos={position}->{position - qty}")


class Trader:
    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}
        trader_data_dict = load_trader_data(state.traderData)

        targets = detect_panel_regime(state, trader_data_dict)
        for product in PANELS:
            if product in targets:
                trade_to_target(product, targets[product], state, result)

        trader_data = json.dumps(trader_data_dict, separators=(",", ":"))
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
