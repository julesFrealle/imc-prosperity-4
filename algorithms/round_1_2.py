from datamodel import OrderDepth, TradingState, Order
import json
import math


OSM  = 'ASH_COATED_OSMIUM'
PEP  = 'INTARIAN_PEPPER_ROOT'
POS_LIMIT = 80

# ── OSMIUM ───────────────────────────────────────────────────────────────────
OSM_EMA_SPAN         = 30
OSM_EMA_ALPHA        = 2.0 / (OSM_EMA_SPAN + 1)
OSM_IMBALANCE_BETA   = 0.5
OSM_POSITION_PENALTY = 0.04
OSM_TAKING_EDGE      = 0
OSM_MAKING_SPREAD    = 3
OSM_WALL_LEAN        = 1.0
OSM_EXTREME_LEAN     = 0.5
OSM_SMALL_QTY_LEAN   = 1.0
OSM_QTY7_LEAN        = 0.5
OSM_QTY9_SELL_LEAN   = 0.8
OSM_QTY8_SELL_LEAN   = 0.8
OSM_SIGNAL_DECAY     = 0.6
OSM_MEAN_PRICE       = 10000
OSM_MEAN_REVERT_GAMMA = 0.3

# ── PEPPER (pure buy-and-hold) ───────────────────────────────────────────────
PEP_GROWTH_RATE      = 0.001
PEP_INTERCEPT_ALPHA  = 0.001
PEP_IC_MAX_DRIFT     = 200
PEP_FAST_ALPHA       = 0.02
PEP_SLOW_ALPHA_PEP   = 0.005
PEP_BEARISH_THR      = -30
PEP_IC_FAST_ALPHA    = 0.01
PEP_IC_SLOW_ALPHA    = 0.001
PEP_IC_REVERSAL_THR  = -15


class Trader:

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        td = {}
        try:
            if state.traderData:
                td = json.loads(state.traderData)
        except Exception:
            pass

        new_td = {}

        # ── OSMIUM ───────────────────────────────────────────────────────────
        if OSM in state.order_depths:
            osm_orders, new_td = self._osm_trade(state, td, new_td)
            result[OSM] = osm_orders

        # ── PEPPER ───────────────────────────────────────────────────────────
        if PEP in state.order_depths:
            pep_orders, new_td = self._pep_trade(state, td, new_td)
            result[PEP] = pep_orders

        try:
            trader_data = json.dumps(new_td)
        except Exception:
            trader_data = ''

        return result, conversions, trader_data

    # ──────────────────────────────────────────────────────────────────────────
    # OSMIUM: EMA market-making + market_trades reversal signals + mean-reversion pull to 10,000
    # ──────────────────────────────────────────────────────────────────────────
    def _osm_trade(self, state, td, new_td):
        od  = state.order_depths[OSM]
        pos = state.position.get(OSM, 0)

        prev_ema = td.get('osm_ema')
        prev_sig = td.get('osm_sig', 0.0)
        prev_pts = td.get('osm_pts', 0)

        buys  = od.buy_orders
        sells = od.sell_orders
        orders = []

        if not buys and not sells:
            new_td['osm_ema'] = prev_ema
            new_td['osm_sig'] = prev_sig * (1 - OSM_SIGNAL_DECAY)
            new_td['osm_pts'] = prev_pts
            return orders, new_td

        best_bid = max(buys)  if buys  else None
        best_ask = min(sells) if sells else None
        bid_wall = min(buys)  if buys  else None
        ask_wall = max(sells) if sells else None

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            mid = best_bid + 4
        else:
            mid = best_ask - 4

        bv  = buys.get(best_bid, 0)       if best_bid is not None else 0
        av  = abs(sells.get(best_ask, 0)) if best_ask is not None else 0
        tv  = bv + av
        imb = (bv - av) / tv if tv > 0 else 0.0

        ema = (
            OSM_EMA_ALPHA * mid + (1 - OSM_EMA_ALPHA) * prev_ema
            if prev_ema is not None else mid
        )

        sig = prev_sig * (1 - OSM_SIGNAL_DECAY)
        pts = prev_pts

        for trade in state.market_trades.get(OSM, []):
            p  = trade.price
            q  = trade.quantity
            bm = (best_bid + best_ask) / 2.0 if (best_bid is not None and best_ask is not None) else mid
            ts = 1 if p > bm else (-1 if p < bm else 0)

            if ask_wall is not None and p >= ask_wall:
                sig -= OSM_WALL_LEAN
            if bid_wall is not None and p <= bid_wall:
                sig += OSM_WALL_LEAN

            if p >= 10012:
                sig -= OSM_EXTREME_LEAN
            elif p <= 9986:
                sig += OSM_EXTREME_LEAN

            if q == 3:
                sig += ts * OSM_SMALL_QTY_LEAN
            if q == 2 and ts == -1:
                sig -= OSM_SMALL_QTY_LEAN * 0.5
            if q == 7:
                sig += OSM_QTY7_LEAN
            if q == 9 and ts == -1:
                sig += OSM_QTY9_SELL_LEAN
            if q == 8 and ts == -1:
                sig -= OSM_QTY8_SELL_LEAN

            if ts != 0:
                pts = ts

        sig = max(-6.0, min(6.0, sig))

        new_td['osm_ema'] = ema
        new_td['osm_sig'] = sig
        new_td['osm_pts'] = pts

        mean_revert_term = -OSM_MEAN_REVERT_GAMMA * (mid - OSM_MEAN_PRICE)
        fair = ema + OSM_IMBALANCE_BETA * imb + sig - OSM_POSITION_PENALTY * pos + mean_revert_term

        max_buy  = POS_LIMIT - pos
        max_sell = POS_LIMIT + pos
        bu = 0
        su = 0

        # TAKE
        if sells and max_buy > 0:
            for ap in sorted(sells.keys()):
                if ap <= fair - OSM_TAKING_EDGE:
                    v = min(abs(sells[ap]), max_buy - bu)
                    if v > 0:
                        orders.append(Order(OSM, ap, v))
                        bu += v
                elif ap <= fair and pos < 0:
                    v = min(abs(sells[ap]), abs(pos) - bu, max_buy - bu)
                    if v > 0:
                        orders.append(Order(OSM, ap, v))
                        bu += v

        if buys and max_sell > 0:
            for bp in sorted(buys.keys(), reverse=True):
                if bp >= fair + OSM_TAKING_EDGE:
                    v = min(buys[bp], max_sell - su)
                    if v > 0:
                        orders.append(Order(OSM, bp, -v))
                        su += v
                elif bp >= fair and pos > 0:
                    v = min(buys[bp], pos - su, max_sell - su)
                    if v > 0:
                        orders.append(Order(OSM, bp, -v))
                        su += v

        # MAKE
        rb = max_buy - bu
        rs = max_sell - su

        if rb > 0:
            mb = int(math.floor(fair - OSM_MAKING_SPREAD))
            if bid_wall is not None:
                mb = max(mb, bid_wall + 1)
            if best_bid is not None and mb > best_bid + 1:
                mb = best_bid + 1
            if mb < fair:
                orders.append(Order(OSM, mb, rb))

        if rs > 0:
            ma = int(math.ceil(fair + OSM_MAKING_SPREAD))
            if ask_wall is not None:
                ma = min(ma, ask_wall - 1)
            if best_ask is not None and ma < best_ask - 1:
                ma = best_ask - 1
            if ma > fair:
                orders.append(Order(OSM, ma, -rs))

        return orders, new_td

    # ──────────────────────────────────────────────────────────────────────────
    # PEPPER: pure buy-and-hold + short cover
    #
    # ONLY BUYS, NEVER SELLS.  Three buy paths:
    #   Phase 1: take any ask ≤ fair (cheap buys when bot spikes down)
    #   Phase 2: if somehow short, cover immediately at fair+1
    #   Phase 3: aggressive buy at best_ask every tick until pos=80
    #
    # Risk guards (dual-EMA + intercept trend) suppress Phase 3 only —
    # they never trigger a sell, only stop us from buying into a crash.
    # ──────────────────────────────────────────────────────────────────────────
    def _pep_fair(self, timestamp, td, mid):
        observed_ic = mid - PEP_GROWTH_RATE * timestamp
        if 'p_ic' not in td:
            td['p_ic'] = observed_ic
        else:
            if abs(td['p_ic'] - observed_ic) > PEP_IC_MAX_DRIFT:
                td['p_ic'] = observed_ic
            else:
                td['p_ic'] = (PEP_INTERCEPT_ALPHA * observed_ic
                              + (1 - PEP_INTERCEPT_ALPHA) * td['p_ic'])
        return td['p_ic'] + PEP_GROWTH_RATE * timestamp

    def _pep_trade(self, state, td, new_td):
        od  = state.order_depths[PEP]
        pos = state.position.get(PEP, 0)

        buys  = od.buy_orders
        sells = od.sell_orders
        orders = []

        best_bid = max(buys)  if buys  else None
        best_ask = min(sells) if sells else None

        if best_bid is None and best_ask is None:
            for key in ('p_ic', 'p_fast', 'p_slow', 'p_ic_fast', 'p_ic_slow'):
                new_td[key] = td.get(key)
            return orders, new_td

        mid = ((best_bid + best_ask) / 2.0 if best_bid and best_ask
               else float(best_bid or best_ask))

        # ── Fair value ────────────────────────────────────────────────────────
        fair     = self._pep_fair(state.timestamp, td, mid)
        fair_int = round(fair)
        new_td['p_ic'] = td.get('p_ic')

        # ── Dual-EMA risk guard ───────────────────────────────────────────────
        prev_fast = td.get('p_fast')
        prev_slow = td.get('p_slow')
        fast = (PEP_FAST_ALPHA * mid + (1 - PEP_FAST_ALPHA) * prev_fast
                if prev_fast is not None else mid)
        slow = (PEP_SLOW_ALPHA_PEP * mid + (1 - PEP_SLOW_ALPHA_PEP) * prev_slow
                if prev_slow is not None else mid)
        new_td['p_fast'] = fast
        new_td['p_slow'] = slow
        phase3_ok = (fast - slow) > PEP_BEARISH_THR

        # ── Intercept trend guard ─────────────────────────────────────────────
        observed_ic = mid - PEP_GROWTH_RATE * state.timestamp
        prev_ic_fast = td.get('p_ic_fast')
        prev_ic_slow = td.get('p_ic_slow')
        ic_fast = (PEP_IC_FAST_ALPHA * observed_ic
                   + (1 - PEP_IC_FAST_ALPHA) * prev_ic_fast
                   if prev_ic_fast is not None else observed_ic)
        ic_slow = (PEP_IC_SLOW_ALPHA * observed_ic
                   + (1 - PEP_IC_SLOW_ALPHA) * prev_ic_slow
                   if prev_ic_slow is not None else observed_ic)
        new_td['p_ic_fast'] = ic_fast
        new_td['p_ic_slow'] = ic_slow
        trend_ok = (ic_fast - ic_slow) > PEP_IC_REVERSAL_THR

        buy_vol = 0

        # ── Phase 1: take any ask ≤ fair (BUY ONLY) ──────────────────────────
        for ap in sorted(sells.keys()):
            if ap <= fair_int:
                qty = min(abs(sells[ap]), POS_LIMIT - pos - buy_vol)
                if qty > 0:
                    orders.append(Order(PEP, ap, qty))
                    buy_vol += qty

        # ── Phase 2: cover any accidental short (BUY ONLY) ───────────────────
        pos_after = pos + buy_vol
        if pos_after < 0 and sells:
            for ap in sorted(sells.keys()):
                if ap <= fair_int + 1:
                    qty = min(abs(sells[ap]), abs(pos_after),
                              POS_LIMIT - pos - buy_vol)
                    if qty > 0:
                        orders.append(Order(PEP, ap, qty))
                        buy_vol   += qty
                        pos_after += qty
                if pos_after >= 0:
                    break

        # ── Phase 3: aggressive buy to 80 (BUY ONLY) ─────────────────────────
        # Suppressed by risk guards — but guards NEVER trigger a sell.
        if sells and phase3_ok and trend_ok:
            remaining = POS_LIMIT - (pos + buy_vol)
            if remaining > 0:
                orders.append(Order(PEP, best_ask, remaining))

        return orders, new_td
