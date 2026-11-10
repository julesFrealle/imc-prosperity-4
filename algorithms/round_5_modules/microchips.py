from datamodel import Order, OrderDepth, TradingState, ProsperityEncoder
from typing import Dict, List, Tuple, Any, Optional
import json


# ─────────────────────────────────────────────────────────────────────────────
# MICROCHIP PRODUCTS
# ─────────────────────────────────────────────────────────────────────────────

MICROCHIP_TRIANGLE = "MICROCHIP_TRIANGLE"
MICROCHIP_OVAL = "MICROCHIP_OVAL"
MICROCHIP_RECTANGLE = "MICROCHIP_RECTANGLE"
MICROCHIP_CIRCLE = "MICROCHIP_CIRCLE"
MICROCHIP_SQUARE = "MICROCHIP_SQUARE"

PRODUCTS = [
    MICROCHIP_TRIANGLE,
    MICROCHIP_OVAL,
    MICROCHIP_RECTANGLE,
    MICROCHIP_CIRCLE,
    MICROCHIP_SQUARE,
]

POSITION_LIMIT = 10


# ─────────────────────────────────────────────────────────────────────────────
# MICROCHIP ETF CONFIG
# ─────────────────────────────────────────────────────────────────────────────
#
# ETF = 5*TRIANGLE - 3*OVAL + 2*RECTANGLE
#
# +1 ETF unit:
#   BUY  5 TRIANGLE
#   SELL 3 OVAL
#   BUY  2 RECTANGLE
#
# -1 ETF unit:
#   SELL 5 TRIANGLE
#   BUY  3 OVAL
#   SELL 2 RECTANGLE
#
# With max_units = 2:
#   TRIANGLE  reserve = 10 -> 0 free
#   OVAL      reserve = 6  -> 4 free
#   RECTANGLE reserve = 4  -> 6 free
#   CIRCLE    reserve = 0  -> 10 free
#   SQUARE    reserve = 0  -> 10 free
# ─────────────────────────────────────────────────────────────────────────────

ETF_CFG = {
    "name": "MICROCHIP_MAIN_ETF",

    "legs": {
        MICROCHIP_TRIANGLE: 5,
        MICROCHIP_OVAL: -3,
        MICROCHIP_RECTANGLE: 2,
    },

    # Calibrated from days 2-4:
    # residual all-day mean around 41358.
    # effective trading std chosen as 1200 to avoid overtrading day 3 noise.
    "fixed_fair": 41350.0,
    "std": 1200.0,

    # 1 unit when |residual| ≈ 0.9 * 1200 = 1080
    # 2 units when |residual| ≈ 1.4 * 1200 = 1680
    "entry_z": 0.5,
    "exit_z": 0.1,
    "max_units": 2,
    "add_z_step": 0.5,

    # EMA-blended fair, same spirit as VISOR.
    "ema_window": 300,
    "ema_weight": 0,
    "shift_cap_std": 1.5,
}


# Passive market making on spare capacity.
# reserve = capacity protected for ETF core.
# MM_CFG = {
#     MICROCHIP_OVAL: {
#         "reserve": 6,
#         "inside_tick": 1,
#         "min_spread": 6,
#         "base_qty": 3,
#     },
#     MICROCHIP_RECTANGLE: {
#         "reserve": 4,
#         "inside_tick": 1,
#         "min_spread": 6,
#         "base_qty": 6,
#     },
#     MICROCHIP_CIRCLE: {
#         "reserve": 0,
#         "inside_tick": 1,
#         "min_spread": 7,
#         "base_qty": 6,
#     },
#     MICROCHIP_SQUARE: {
#         "reserve": 0,
#         "inside_tick": 1,
#         "min_spread": 10,
#         "base_qty": 1,
#     },
# }

MM_CFG = {
    MICROCHIP_OVAL: {
        "reserve": 6,
        "inside_tick": 1,
        "min_spread": 7,
        "base_qty": 2,
    },
    MICROCHIP_RECTANGLE: {
        "reserve": 4,
        "inside_tick": 1,
        "min_spread": 8,
        "base_qty": 3,
    },
    MICROCHIP_CIRCLE: {
        "reserve": 0,
        "inside_tick": 1,
        "min_spread": 8,
        "base_qty": 5,
    },
    MICROCHIP_SQUARE: {
        "reserve": 0,
        "inside_tick": 1,
        "min_spread": 12,
        "base_qty": 1,
    },
}

class Trader:

    # ─────────────────────────────────────────────────────────────────────
    # TRADER DATA
    # ─────────────────────────────────────────────────────────────────────

    def load_data(self, trader_data: str) -> Dict[str, Any]:
        if trader_data:
            try:
                data = json.loads(trader_data)

                if isinstance(data, dict):
                    if "microchip" not in data or not isinstance(data["microchip"], dict):
                        data["microchip"] = {}

                    data["microchip"].setdefault("ema", None)
                    data["microchip"].setdefault("units", 0)

                    return data

            except Exception:
                pass

        return {
            "microchip": {
                "ema": None,
                "units": 0,
            }
        }

    def encode_data(self, data: Dict[str, Any]) -> str:
        try:
            return json.dumps(data, cls=ProsperityEncoder, separators=(",", ":"))
        except Exception:
            try:
                return json.dumps(data, separators=(",", ":"))
            except Exception:
                return ""

    # ─────────────────────────────────────────────────────────────────────
    # BOOK HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def best_bid_ask_mid(
        self,
        depth: Optional[OrderDepth],
    ) -> Tuple[Optional[int], Optional[int], Optional[float]]:

        if depth is None:
            return None, None, None

        if not depth.buy_orders or not depth.sell_orders:
            return None, None, None

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        return best_bid, best_ask, mid

    def visible_buy_volume(self, depth: OrderDepth, price: int) -> int:
        # Volume available to sell into at bid.
        return max(0, int(depth.buy_orders.get(price, 0)))

    def visible_sell_volume(self, depth: OrderDepth, price: int) -> int:
        # Volume available to buy from at ask.
        # In the live datamodel, sell_orders quantities are negative.
        return max(0, -int(depth.sell_orders.get(price, 0)))

    def clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    # ─────────────────────────────────────────────────────────────────────
    # ETF SIGNAL
    # ─────────────────────────────────────────────────────────────────────

    def basket_mid_value(
        self,
        state: TradingState,
        legs: Dict[str, int],
    ) -> Optional[float]:

        value = 0.0

        for product, coeff in legs.items():
            if product not in state.order_depths:
                return None

            _, _, mid = self.best_bid_ask_mid(state.order_depths[product])

            if mid is None:
                return None

            value += int(coeff) * mid

        return value

    def fair_value(
        self,
        data: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> float:

        fixed_fair = float(cfg["fixed_fair"])
        std = max(float(cfg["std"]), 1.0)

        ema = data["microchip"].get("ema")

        if ema is None:
            return fixed_fair

        raw_shift = float(ema) - fixed_fair
        cap = float(cfg["shift_cap_std"]) * std
        shift = self.clamp(raw_shift, -cap, cap)

        return fixed_fair + float(cfg["ema_weight"]) * shift

    def update_ema_after_signal(
        self,
        data: Dict[str, Any],
        basket_value: float,
        cfg: Dict[str, Any],
    ) -> None:

        old_ema = data["microchip"].get("ema")
        alpha = 2.0 / (float(cfg["ema_window"]) + 1.0)

        if old_ema is None:
            data["microchip"]["ema"] = float(basket_value)
        else:
            data["microchip"]["ema"] = float(old_ema) + alpha * (
                float(basket_value) - float(old_ema)
            )

    def target_units_from_z(
        self,
        z: float,
        current_units: int,
        cfg: Dict[str, Any],
    ) -> int:

        entry_z = float(cfg["entry_z"])
        exit_z = float(cfg["exit_z"])
        max_units = int(cfg["max_units"])
        add_z_step = max(float(cfg.get("add_z_step", 999.0)), 1e-9)

        # Basket expensive -> sell ETF.
        if z >= entry_z:
            size = 1 + int((abs(z) - entry_z) / add_z_step + 1e-9)
            return -min(max_units, max(1, size))

        # Basket cheap -> buy ETF.
        if z <= -entry_z:
            size = 1 + int((abs(z) - entry_z) / add_z_step + 1e-9)
            return min(max_units, max(1, size))

        # Exit long when residual returns close to zero.
        if current_units > 0:
            if z >= -exit_z:
                return 0
            return current_units

        # Exit short when residual returns close to zero.
        if current_units < 0:
            if z <= exit_z:
                return 0
            return current_units

        return 0

    # ─────────────────────────────────────────────────────────────────────
    # EXACT-RATIO ETF EXECUTION
    # Supports negative coefficients.
    # ─────────────────────────────────────────────────────────────────────

    def max_executable_etf_units(
        self,
        state: TradingState,
        legs: Dict[str, int],
        direction: int,
        desired_units: int,
        planned_position: Dict[str, int],
    ) -> int:

        possible_units = int(desired_units)

        if possible_units <= 0:
            return 0

        for product, coeff in legs.items():
            if product not in state.order_depths:
                return 0

            depth = state.order_depths[product]
            best_bid, best_ask, _ = self.best_bid_ask_mid(depth)

            if best_bid is None or best_ask is None:
                return 0

            qty_per_unit = direction * int(coeff)
            abs_qty = abs(qty_per_unit)

            if abs_qty == 0:
                continue

            pos = int(planned_position.get(product, 0))

            if qty_per_unit > 0:
                # Need to buy this leg at best ask.
                book_cap = self.visible_sell_volume(depth, best_ask)
                pos_cap = POSITION_LIMIT - pos
            else:
                # Need to sell this leg at best bid.
                book_cap = self.visible_buy_volume(depth, best_bid)
                pos_cap = POSITION_LIMIT + pos

            max_by_book = book_cap // abs_qty
            max_by_pos = pos_cap // abs_qty

            possible_units = min(possible_units, max_by_book, max_by_pos)

        return max(0, int(possible_units))

    def execute_etf_delta(
        self,
        state: TradingState,
        result: Dict[str, List[Order]],
        cfg: Dict[str, Any],
        delta_units: int,
        planned_position: Dict[str, int],
    ) -> int:

        if delta_units == 0:
            return 0

        direction = 1 if delta_units > 0 else -1
        desired_units = abs(int(delta_units))

        executable_units = self.max_executable_etf_units(
            state=state,
            legs=cfg["legs"],
            direction=direction,
            desired_units=desired_units,
            planned_position=planned_position,
        )

        if executable_units <= 0:
            return 0

        for product, coeff in cfg["legs"].items():
            depth = state.order_depths[product]
            best_bid, best_ask, _ = self.best_bid_ask_mid(depth)

            qty = direction * int(coeff) * executable_units

            if qty > 0:
                price = best_ask
            else:
                price = best_bid

            if product not in result:
                result[product] = []

            result[product].append(Order(product, price, qty))
            planned_position[product] = int(planned_position.get(product, 0)) + int(qty)

        return direction * executable_units

    # ─────────────────────────────────────────────────────────────────────
    # MARKET MAKING ON SPARE CAPACITY
    # ─────────────────────────────────────────────────────────────────────

    def etf_core_position_for_product(
        self,
        product: str,
        etf_units: int,
        cfg: Dict[str, Any],
    ) -> int:

        coeff = int(cfg["legs"].get(product, 0))
        return coeff * int(etf_units)

    def spare_capacity_for_product(
        self,
        mm_cfg: Dict[str, Any],
    ) -> int:

        reserve = abs(int(mm_cfg.get("reserve", 0)))
        return max(0, POSITION_LIMIT - reserve)

    def mm_position_band(
        self,
        product: str,
        etf_units: int,
        cfg: Dict[str, Any],
        mm_cfg: Dict[str, Any],
    ) -> Tuple[int, int]:

        core_pos = self.etf_core_position_for_product(
            product=product,
            etf_units=etf_units,
            cfg=cfg,
        )

        spare = self.spare_capacity_for_product(mm_cfg)

        lower = max(-POSITION_LIMIT, core_pos - spare)
        upper = min(POSITION_LIMIT, core_pos + spare)

        return int(lower), int(upper)

    def add_market_making_orders(
        self,
        product: str,
        depth: OrderDepth,
        result: Dict[str, List[Order]],
        planned_position: Dict[str, int],
        etf_units: int,
        cfg: Dict[str, Any],
        mm_cfg: Dict[str, Any],
    ) -> None:

        best_bid, best_ask, _ = self.best_bid_ask_mid(depth)

        if best_bid is None or best_ask is None:
            return

        spread = best_ask - best_bid

        min_spread = int(mm_cfg.get("min_spread", 4))

        # Need enough spread to quote strictly inside.
        if spread < min_spread:
            return

        inside_tick = max(1, int(mm_cfg.get("inside_tick", 1)))

        bid_px = min(best_bid + inside_tick, best_ask - 1)
        ask_px = max(best_ask - inside_tick, best_bid + 1)

        # Avoid crossing / self-crossing.
        if not (best_bid < bid_px < ask_px < best_ask):
            return

        lower, upper = self.mm_position_band(
            product=product,
            etf_units=etf_units,
            cfg=cfg,
            mm_cfg=mm_cfg,
        )

        pos = int(planned_position.get(product, 0))

        spare = self.spare_capacity_for_product(mm_cfg)
        base_qty = max(1, int(mm_cfg.get("base_qty", 2)))

        # Inventory skew inside the allowed spare band.
        center = (lower + upper) / 2.0

        buy_qty = base_qty
        sell_qty = base_qty

        if pos > center:
            # Already long versus band center: sell more, buy less.
            skew = int((pos - center) // 2)
            sell_qty += min(3, max(0, skew))
            buy_qty -= min(3, max(0, skew))

        elif pos < center:
            # Already short versus band center: buy more, sell less.
            skew = int((center - pos) // 2)
            buy_qty += min(3, max(0, skew))
            sell_qty -= min(3, max(0, skew))

        buy_qty = max(0, buy_qty)
        sell_qty = max(0, sell_qty)

        # Hard band checks.
        buy_qty = min(
            buy_qty,
            spare,
            upper - pos,
            POSITION_LIMIT - pos,
        )

        sell_qty = min(
            sell_qty,
            spare,
            pos - lower,
            POSITION_LIMIT + pos,
        )

        buy_qty = max(0, int(buy_qty))
        sell_qty = max(0, int(sell_qty))

        if buy_qty <= 0 and sell_qty <= 0:
            return

        if product not in result:
            result[product] = []

        if buy_qty > 0:
            result[product].append(Order(product, bid_px, buy_qty))

        if sell_qty > 0:
            result[product].append(Order(product, ask_px, -sell_qty))

        # We deliberately do not update planned_position here.
        # Reason: these are passive two-sided quotes.
        # Each side is individually checked to be safe if only one side fills.

    # ─────────────────────────────────────────────────────────────────────
    # MAIN
    # ─────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        data = self.load_data(state.traderData)

        planned_position: Dict[str, int] = {}

        for product in PRODUCTS:
            planned_position[product] = int(state.position.get(product, 0))

        cfg = ETF_CFG

        # ------------------------------------------------------------
        # 1) Exact-ratio ETF residual core
        # ------------------------------------------------------------

        basket_value = self.basket_mid_value(
            state=state,
            legs=cfg["legs"],
        )

        if basket_value is not None:
            fair = self.fair_value(
                data=data,
                cfg=cfg,
            )

            std = max(float(cfg["std"]), 1.0)
            residual = basket_value - fair
            z = residual / std

            current_units = int(data["microchip"].get("units", 0))

            target_units = self.target_units_from_z(
                z=z,
                current_units=current_units,
                cfg=cfg,
            )

            delta_units = int(target_units - current_units)

            executed_delta = self.execute_etf_delta(
                state=state,
                result=result,
                cfg=cfg,
                delta_units=delta_units,
                planned_position=planned_position,
            )

            if executed_delta != 0:
                data["microchip"]["units"] = int(current_units + executed_delta)

            # Important: use current signal first, then update EMA.
            self.update_ema_after_signal(
                data=data,
                basket_value=basket_value,
                cfg=cfg,
            )

            data["microchip"]["last_basket"] = round(float(basket_value), 2)
            data["microchip"]["last_fair"] = round(float(fair), 2)
            data["microchip"]["last_residual"] = round(float(residual), 2)
            data["microchip"]["last_z"] = round(float(z), 4)
            data["microchip"]["last_target"] = int(target_units)

        # Updated ETF unit count after possible execution.
        etf_units_now = int(data["microchip"].get("units", 0))

        # ------------------------------------------------------------
        # 2) Passive MM on spare capacity
        # ------------------------------------------------------------

        for product, mm_cfg in MM_CFG.items():
            if product not in state.order_depths:
                continue

            self.add_market_making_orders(
                product=product,
                depth=state.order_depths[product],
                result=result,
                planned_position=planned_position,
                etf_units=etf_units_now,
                cfg=cfg,
                mm_cfg=mm_cfg,
            )

        result = {
            product: orders
            for product, orders in result.items()
            if orders
        }

        traderData = self.encode_data(data)

        return result, conversions, traderData