"""
IMC PROSPERITY 4 — ROUND 3 (V6: L1+L2 + CRAZY VOUCHERS)
=========================================================
Same fixed-fair aggressive-take logic as V5, plus optional handling of the
"crazy" vouchers (VEV_4000, VEV_4500, VEV_6000, VEV_6500).

OPTIONAL TOGGLE:
  TRADE_CRAZY_VOUCHERS = True/False at the top.
  When False, the crazy vouchers are skipped entirely.

CRAZY VOUCHER ANALYSIS (from 3-day data):

(A) VEV_4000 and VEV_4500 — DEEP ITM, dynamic intrinsic fair
    These options track VFE almost perfectly:
      - VEV_4000 mid ≈ VFE − 4000 (premium mean = +0.01, std = 0.83)
      - VEV_4500 mid ≈ VFE − 4500 (premium mean = +0.01, std = 0.76)

    Therefore the FAIR VALUE IS DYNAMIC: fair = VFE_mid − K (not fixed).
    A fixed fair would catastrophically fail when VFE drifts (day 2 mean is
    5255 vs 5250 on days 0/1 — that's a 5-tick shift in fair).

    The book is structurally:
      - bid ≈ intrinsic − 10  (VEV_4000) or intrinsic − 8  (VEV_4500)
      - ask ≈ intrinsic + 10  (VEV_4000) or intrinsic + 8  (VEV_4500)
    Spreads are ~21 and ~16. Trigger fires only ~73 times per 10000 ticks
    on the BUY side at edge=0.5, and ~3 times on the SELL side. Very rare.

    L2 IS NOT USED for these — the bid/ask gap is so wide that L2 is never
    close enough to intrinsic to be profitable in our backtest.

    Edges chosen:
      - buy_edge = 0.5  (fires sometimes when bot misprices ask down 1 tick)
      - sell_edge = 2.0 (sells almost never fire anyway; wide for safety)

    Day-by-day PnL: D0=+305, D1=+4566, D2=+8401 (very uneven; concentrated
    on the trending day, but never negative).

(B) VEV_6000 and VEV_6500 — WORTHLESS, NEVER TRADE
    bid = 0 every single tick (30,000/30,000)
    ask = 1 every single tick
    No L2 ever exists.
    Mid is exactly 0.5 with std = 0.00.

    Trading round trip = (buy at 1, sell at 0) = guaranteed -1 per trade.
    These are excluded permanently. We list them in the config only with
    enabled=False to make the deliberate exclusion explicit.

WHY ASYMMETRIC EDGES STILL MATTER FOR DEEP ITM:
The premium distribution is symmetric around 0 (ask is always +10 above,
bid is always −10 below) but the absolute magnitudes of dislocations differ.
At buy_edge=0.5 we get ~73 triggers/day; at sell_edge=0.5 only ~3 triggers/day.
Setting sell_edge wider doesn't lose us much (rare anyway) but protects against
bad sell fills if the regime shifts.
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import json


# ─────────────────────────────────────────────────────────────────────────────
# TOGGLE: enable/disable the deep-ITM crazy vouchers (VEV_4000, VEV_4500)
# VEV_6000 and VEV_6500 are always disabled (they're guaranteed losers).
# ─────────────────────────────────────────────────────────────────────────────
TRADE_CRAZY_VOUCHERS = True


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Standard CFG entries: fixed fair value
#   fair, buy_edge, sell_edge, limit, cap, l2_buy_extra, l2_sell_extra
#
# Crazy CFG entries (deep-ITM only): dynamic fair = VFE_mid − strike
#   strike, buy_edge, sell_edge, limit, cap, dynamic=True
#   l2 is intentionally NOT used (book gap too wide to ever fire L2 profitably)

CFG = {
    # "HYDROGEL_PACK": {"fair": 10005, "buy_edge": 8.0, "sell_edge": 12.0, "limit": 200, "cap": 15,
    #                    "l2_buy_extra": 2, "l2_sell_extra": 2},
    "VELVETFRUIT_EXTRACT": {"fair": 5255, "buy_edge": 10, "sell_edge": 17, "limit": 300, "cap": 300,
                            "l2_buy_extra": 2, "l2_sell_extra": 2},
    "VEV_5000": {"fair": 258, "buy_edge": 7, "sell_edge": 9, "limit": 300, "cap": 300,
                 "l2_buy_extra": 1, "l2_sell_extra": 1},
    "VEV_5100": {"fair": 174, "buy_edge": 9, "sell_edge": 9, "limit": 300, "cap": 300,
                 "l2_buy_extra": 1, "l2_sell_extra": 1},
    "VEV_5200": {"fair": 98,  "buy_edge": 5, "sell_edge": 8, "limit": 300, "cap": 300,
                 "l2_buy_extra": 1, "l2_sell_extra": 1},
    "VEV_5300": {"fair": 48,  "buy_edge": 3, "sell_edge": 4, "limit": 300, "cap": 300,
                 "l2_buy_extra": 1, "l2_sell_extra": 1},
    "VEV_5400": {"fair": 16,  "buy_edge": 1, "sell_edge": 1.5, "limit": 300, "cap": 300,
                 "l2_buy_extra": 1, "l2_sell_extra": 1},
    "VEV_5500": {"fair": 7.5,   "buy_edge": 0.5, "sell_edge": 0.5, "limit": 300, "cap": 300,
                 "l2_buy_extra": 1, "l2_sell_extra": 1},
}

# Deep-ITM dynamic-fair config (only active if TRADE_CRAZY_VOUCHERS=True).
# Intentionally L2-disabled: the bid/ask gap (16-21 ticks) is so wide that
# L2 is never close enough to intrinsic to be profitable.
CRAZY_CFG_DEEP_ITM = {
    "VEV_4000": {"strike": 4000, "buy_edge": 0.5, "sell_edge": 2.0, "limit": 300, "cap": 300},
    "VEV_4500": {"strike": 4500, "buy_edge": 0.5, "sell_edge": 2.0, "limit": 300, "cap": 300},
}


HARD_LIMIT = {
    # "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300, "VEV_5100": 300,
    "VEV_5200": 300, "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}

TRADABLE_PRODUCTS = list(CFG.keys())


# ─────────────────────────────────────────────────────────────────────────────
# HYDROGEL_PACK — Round 3, strategy v7
#
# Taker-only laddered z-score mean reversion.
# Skips shallow rung entirely. Concentrates trading on real dislocations
# where edge per trade is large enough to overcome the spread.
#
# v6 finding: passive orders don't fill in this simulator. v6 only made
# money when escalation took the book at z >= 1.4. v7 leans into that.
# ─────────────────────────────────────────────────────────────────────────────

HYD = 'HYDROGEL_PACK'
HYD_POS_LIMIT = 200

HYD_FAIR  = 9990
HYD_SIGMA = 32

# Tighter ladder, no shallow noise rung.
HYD_LADDER = [
    (1.0, 0.5),   # half size at 1σ
    (1.5, 1.0),   # full size at 1.5σ
]

HYD_ENTRY_Z = 1.0    # minimum |z| to enter
HYD_EXIT_Z  = 0.2    # exit toward 0 once |z| drops below this


class Trader:
     # ─────────────────────────────────────────────────────────────────────────
    def _ladder_target(self, abs_z, sign):
        frac = 0.0
        for threshold, f in HYD_LADDER:
            if abs_z >= threshold:
                frac = f
        return int(sign * frac * HYD_POS_LIMIT)

    # ─────────────────────────────────────────────────────────────────────────
    def _hyd_trade(self, state, td, new_td):
        od  = state.order_depths[HYD]
        pos = state.position.get(HYD, 0)

        buys  = od.buy_orders
        sells = od.sell_orders
        orders = []

        if not buys or not sells:
            return orders, new_td

        best_bid = max(buys)
        best_ask = min(sells)
        bid_vol = buys[best_bid]
        ask_vol = abs(sells[best_ask])
        mid = (best_bid + best_ask) / 2.0
        new_td["hyd_prev_mid"] = mid

        z_mid = (mid - HYD_FAIR) / HYD_SIGMA

        max_buy  = HYD_POS_LIMIT - pos
        max_sell = HYD_POS_LIMIT + pos

        # ── Determine target ─────────────────────────────────────────────────
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
                ladder_t = self._ladder_target(abs(z_mid), +1)
                target = max(pos, ladder_t)  # only ADD

        else:  # pos < 0
            if z_mid <= HYD_EXIT_Z:
                target = 0
            elif z_mid >= HYD_ENTRY_Z:
                ladder_t = self._ladder_target(abs(z_mid), -1)
                target = min(pos, ladder_t)

        delta = target - pos
        if delta == 0:
            return orders, new_td

        # ── Execute (taker only) ─────────────────────────────────────────────
        if delta > 0:
            qty = min(delta, max_buy, ask_vol)
            if qty > 0:
                orders.append(Order(HYD, best_ask, qty))

        else:
            qty = min(-delta, max_sell, bid_vol)
            if qty > 0:
                orders.append(Order(HYD, best_bid, -qty))

        return orders, new_td
    

    # ── STANDARD FIXED-FAIR L1+L2 SWEEP ─────────────────────────────────────
    def fixed_fair_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
    ) -> List[Order]:
        """
        Aggressive take with L1 + L2 sweeping around a fixed fair value.
        Used for HYDROGEL, VFE, VEV_5000–5500.
        """
        orders: List[Order] = []
        cfg = CFG[product]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        fair       = cfg["fair"]
        buy_edge   = cfg["buy_edge"]
        sell_edge  = cfg["sell_edge"]
        limit      = cfg["limit"]
        cap        = cfg["cap"]
        l2_b_extra = cfg["l2_buy_extra"]
        l2_s_extra = cfg["l2_sell_extra"]

        return self._do_l1_l2_sweep(
            product, order_depth, position,
            fair, buy_edge, sell_edge, limit, cap,
            l2_b_extra, l2_s_extra,
        )

    # ── DEEP-ITM DYNAMIC-FAIR SWEEP (L1 ONLY) ───────────────────────────────
    def crazy_deep_itm_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        vfe_mid: float,
    ) -> List[Order]:
        """
        Aggressive take with DYNAMIC fair = VFE_mid − strike (intrinsic value).
        Used for VEV_4000 and VEV_4500.

        L2 is NOT used here: the bid/ask gap is structurally 16-21 ticks wide,
        so L2 is always >= 1 tick further from intrinsic than L1, which means
        L2 is never close enough to intrinsic to give us positive edge in
        backtest. Adding L2 here would only ever hurt or do nothing.
        """
        orders: List[Order] = []
        cfg = CRAZY_CFG_DEEP_ITM[product]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
        if vfe_mid is None:
            return orders  # cannot compute dynamic fair

        K          = cfg["strike"]
        fair       = vfe_mid - K        # dynamic intrinsic fair
        buy_edge   = cfg["buy_edge"]
        sell_edge  = cfg["sell_edge"]
        limit      = cfg["limit"]
        cap        = cfg["cap"]

        return self._do_l1_l2_sweep(
            product, order_depth, position,
            fair, buy_edge, sell_edge, limit, cap,
            l2_buy_extra=999, l2_sell_extra=999,  # disables L2 sweep
        )

    # ── SHARED L1+L2 EXECUTION CORE ─────────────────────────────────────────
    def _do_l1_l2_sweep(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        fair: float,
        buy_edge: float,
        sell_edge: float,
        limit: int,
        cap: int,
        l2_buy_extra: float,
        l2_sell_extra: float,
    ) -> List[Order]:
        """
        Take L1 first (best price), then L2 if dislocation is large enough.
        Position limit and cap enforced jointly across both levels.
        Pass l2_*_extra=999 to effectively disable L2 sweeping.
        """
        orders: List[Order] = []
        pos = position

        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        asks = sorted(order_depth.sell_orders.keys())

        # ── BUY SIDE ──
        buy_budget = cap

        l1_ask = asks[0]
        l1_ask_vol = abs(order_depth.sell_orders[l1_ask])
        if l1_ask <= fair - buy_edge and pos < limit and buy_budget > 0:
            qty = min(l1_ask_vol, limit - pos, buy_budget)
            if qty > 0:
                orders.append(Order(product, l1_ask, qty))
                pos += qty
                buy_budget -= qty

        if len(asks) >= 2 and pos < limit and buy_budget > 0:
            l2_ask = asks[1]
            l2_ask_vol = abs(order_depth.sell_orders[l2_ask])
            if l2_ask <= fair - (buy_edge + l2_buy_extra):
                qty = min(l2_ask_vol, limit - pos, buy_budget)
                if qty > 0:
                    orders.append(Order(product, l2_ask, qty))
                    pos += qty
                    buy_budget -= qty

        # ── SELL SIDE ──
        sell_budget = cap

        l1_bid = bids[0]
        l1_bid_vol = order_depth.buy_orders[l1_bid]
        if l1_bid >= fair + sell_edge and pos > -limit and sell_budget > 0:
            qty = min(l1_bid_vol, limit + pos, sell_budget)
            if qty > 0:
                orders.append(Order(product, l1_bid, -qty))
                pos -= qty
                sell_budget -= qty

        if len(bids) >= 2 and pos > -limit and sell_budget > 0:
            l2_bid = bids[1]
            l2_bid_vol = order_depth.buy_orders[l2_bid]
            if l2_bid >= fair + (sell_edge + l2_sell_extra):
                qty = min(l2_bid_vol, limit + pos, sell_budget)
                if qty > 0:
                    orders.append(Order(product, l2_bid, -qty))
                    pos -= qty
                    sell_budget -= qty

        return orders

    # ── HELPER: VFE mid for crazy voucher fair calc ─────────────────────────
    @staticmethod
    def _vfe_mid(state: TradingState):
        od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if od is None or not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0

    # ── MAIN ────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        td = {}
        try:
            if state.traderData:
                td = json.loads(state.traderData)
        except Exception:
            pass

        new_td = {}

        if HYD in state.order_depths:
            hyd_orders, new_td = self._hyd_trade(state, td, new_td)
            result[HYD] = hyd_orders

        try:
            trader_data = json.dumps(new_td)
        except Exception:
            trader_data = ''

        # Standard products with fixed fair
        for product in TRADABLE_PRODUCTS:
            if product not in state.order_depths:
                continue
            position = state.position.get(product, 0)
            orders = self.fixed_fair_orders(product, state.order_depths[product], position)
            if orders:
                result[product] = orders

        # Crazy deep-ITM with dynamic intrinsic fair
        if TRADE_CRAZY_VOUCHERS:
            vfe_mid = self._vfe_mid(state)
            if vfe_mid is not None:
                for product in CRAZY_CFG_DEEP_ITM:
                    if product not in state.order_depths:
                        continue
                    position = state.position.get(product, 0)
                    orders = self.crazy_deep_itm_orders(
                        product, state.order_depths[product], position, vfe_mid
                    )
                    if orders:
                        result[product] = orders

        return result, conversions, ""
