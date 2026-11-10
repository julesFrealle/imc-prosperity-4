from datamodel import Order, OrderDepth, Symbol, TradingState
from typing import Dict, List, Tuple


# ============================================================
# ROUND 5 — TRANSLATOR MARKET MAKING OPTIMISED
# ============================================================
#
# Strategy:
#   - Treat each Instant Translator independently.
#   - No ETF / basket strategy.
#   - No trend / momentum / mean-reversion prediction.
#   - Pure spread capture through passive market making.
#
# Logic:
#   - Quote inside the bid-ask spread.
#   - Use per-product spread thresholds.
#   - Use per-product inside-tick settings.
#   - Increase size when spread is wide.
#   - Skew prices and sizes based on inventory.
#   - Stop worsening inventory near soft limits.
#   - Only aggressively flatten at the absolute position limit.
#
# Position limit:
#   - All products have ±10 limit.
# ============================================================


POSITION_LIMIT = 10

TRANSLATOR_PRODUCTS = [
    "TRANSLATOR_VOID_BLUE",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_ASTRO_BLACK",
]


# ============================================================
# Per-product configuration
# ============================================================
#
# min_spread:
#   Only quote if spread is at least this wide.
#
# inside_tick:
#   How many ticks inside the current best bid/ask we quote.
#
# base_size:
#   Default quote size when inventory is close to flat.
#
# wide_spread_threshold:
#   If spread is at least this wide, increase quote size by +1.
# ============================================================

PRODUCT_CONFIG = {
    "TRANSLATOR_VOID_BLUE": {
        "min_spread": 8,
        "inside_tick": 2,
        "base_size": 2,
        "wide_spread_threshold": 10,
    },
    "TRANSLATOR_GRAPHITE_MIST": {
        "min_spread": 7,
        "inside_tick": 1,
        "base_size": 2,
        "wide_spread_threshold": 10,
    },
    "TRANSLATOR_ECLIPSE_CHARCOAL": {
        "min_spread": 7,
        "inside_tick": 1,
        "base_size": 2,
        "wide_spread_threshold": 10,
    },
    "TRANSLATOR_SPACE_GRAY": {
        "min_spread": 7,
        "inside_tick": 1,
        "base_size": 2,
        "wide_spread_threshold": 10,
    },
    "TRANSLATOR_ASTRO_BLACK": {
        "min_spread": 7,
        "inside_tick": 1,
        "base_size": 2,
        "wide_spread_threshold": 10,
    },
}


# ============================================================
# Inventory management parameters
# ============================================================

SOFT_LIMIT = 7

# Important:
#   HARD_LIMIT = 10 means we only cross the spread when we are
#   actually at the absolute position limit.
#
#   At position ±9, we still use passive flattening instead of
#   instantly dumping into the best bid / ask.
HARD_LIMIT = 10

INVENTORY_SKEW = 0.5

AGGRESSIVE_FLATTEN_SIZE = 4


class Trader:
    def __init__(self):
        pass

    # --------------------------------------------------------
    # Book helpers
    # --------------------------------------------------------

    def get_best_bid_ask(self, od: OrderDepth) -> Tuple[int, int]:
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        return best_bid, best_ask

    def get_available_buy_volume(self, od: OrderDepth, price: int) -> int:
        """
        Available volume to buy at a sell-order price.
        Sell orders are negative volumes in the datamodel.
        """
        if price in od.sell_orders:
            return -od.sell_orders[price]
        return 0

    def get_available_sell_volume(self, od: OrderDepth, price: int) -> int:
        """
        Available volume to sell at a buy-order price.
        Buy orders are positive volumes in the datamodel.
        """
        if price in od.buy_orders:
            return od.buy_orders[price]
        return 0

    def clamp_buy(self, position: int, size: int) -> int:
        """
        Maximum additional buy allowed before hitting +10.
        """
        max_buy = POSITION_LIMIT - position
        return max(0, min(size, max_buy))

    def clamp_sell(self, position: int, size: int) -> int:
        """
        Maximum additional sell allowed before hitting -10.
        """
        max_sell = POSITION_LIMIT + position
        return max(0, min(size, max_sell))

    # --------------------------------------------------------
    # Inventory-based sizing
    # --------------------------------------------------------

    def compute_quote_sizes(
        self,
        position: int,
        base_size: int,
        spread: int,
        wide_threshold: int,
    ) -> Tuple[int, int]:
        """
        Returns:
            buy_size, sell_size

        Sizing logic:
            - If flat, quote equal size on both sides.
            - If long, reduce buy size and increase sell size.
            - If short, increase buy size and reduce sell size.
            - Near soft limits, only quote the side that reduces inventory.
            - If spread is wide, increase size slightly.
        """

        size_bonus = 1 if spread >= wide_threshold else 0
        normal_size = base_size + size_bonus

        if position >= SOFT_LIMIT:
            buy_size = 0
            sell_size = AGGRESSIVE_FLATTEN_SIZE

        elif position <= -SOFT_LIMIT:
            buy_size = AGGRESSIVE_FLATTEN_SIZE
            sell_size = 0

        elif position >= 4:
            buy_size = max(1, normal_size - 2)
            sell_size = normal_size

        elif position <= -4:
            buy_size = normal_size
            sell_size = max(1, normal_size - 2)

        elif position > 0:
            buy_size = max(1, normal_size - 1)
            sell_size = normal_size

        elif position < 0:
            buy_size = normal_size
            sell_size = max(1, normal_size - 1)

        else:
            buy_size = normal_size
            sell_size = normal_size

        buy_size = self.clamp_buy(position, buy_size)
        sell_size = self.clamp_sell(position, sell_size)

        return buy_size, sell_size

    # --------------------------------------------------------
    # Quote price logic
    # --------------------------------------------------------

    def compute_quotes(
        self,
        best_bid: int,
        best_ask: int,
        position: int,
        inside_tick: int,
    ) -> Tuple[int, int]:
        """
        Quote directly inside the NBBO.

        This version intentionally does NOT anchor to weighted_mid,
        because anchoring around int(weighted_mid) can accidentally make
        one side too aggressive on odd spreads and reduce spread capture.

        Inventory skew:
            - Long inventory  -> lower quotes.
              This makes buying less aggressive and selling more aggressive.

            - Short inventory -> raise quotes.
              This makes buying more aggressive and selling less aggressive.
        """

        if position > 0:
            skew = -int(abs(position) * INVENTORY_SKEW + 0.5)
        elif position < 0:
            skew = int(abs(position) * INVENTORY_SKEW + 0.5)
        else:
            skew = 0

        raw_buy_price = best_bid + inside_tick + skew
        raw_sell_price = best_ask - inside_tick + skew

        # Do not cross the visible market.
        buy_price = min(raw_buy_price, best_ask - 1)
        sell_price = max(raw_sell_price, best_bid + 1)

        return buy_price, sell_price

    # --------------------------------------------------------
    # Hard flattening at absolute limit
    # --------------------------------------------------------

    def hard_flatten(
        self,
        product: Symbol,
        od: OrderDepth,
        position: int,
        orders: List[Order],
    ) -> bool:
        """
        Only aggressively flatten at the absolute limit.

        If position is +10:
            Sell into best bid to reduce long risk.

        If position is -10:
            Buy from best ask to reduce short risk.

        At +9 / -9, we do NOT cross the spread. We let passive
        inventory-control quotes handle the reduction.
        """

        best_bid, best_ask = self.get_best_bid_ask(od)

        if position >= HARD_LIMIT:
            visible_bid_volume = self.get_available_sell_volume(od, best_bid)
            quantity = min(position, visible_bid_volume, AGGRESSIVE_FLATTEN_SIZE)

            if quantity > 0:
                orders.append(Order(product, best_bid, -quantity))

            return True

        if position <= -HARD_LIMIT:
            visible_ask_volume = self.get_available_buy_volume(od, best_ask)
            quantity = min(-position, visible_ask_volume, AGGRESSIVE_FLATTEN_SIZE)

            if quantity > 0:
                orders.append(Order(product, best_ask, quantity))

            return True

        return False

    # --------------------------------------------------------
    # Market making for one translator
    # --------------------------------------------------------

    def market_make_translator(
        self,
        product: Symbol,
        state: TradingState,
    ) -> List[Order]:
        orders: List[Order] = []

        if product not in state.order_depths:
            return orders

        od = state.order_depths[product]

        if not od.buy_orders or not od.sell_orders:
            return orders

        position = state.position.get(product, 0)
        cfg = PRODUCT_CONFIG[product]

        best_bid, best_ask = self.get_best_bid_ask(od)
        spread = best_ask - best_bid

        # If already at the absolute position limit, flatten first.
        if self.hard_flatten(product, od, position, orders):
            return orders

        # Do not quote when spread is too tight.
        if spread < cfg["min_spread"]:
            return orders

        buy_price, sell_price = self.compute_quotes(
            best_bid=best_bid,
            best_ask=best_ask,
            position=position,
            inside_tick=cfg["inside_tick"],
        )

        buy_size, sell_size = self.compute_quote_sizes(
            position=position,
            base_size=cfg["base_size"],
            spread=spread,
            wide_threshold=cfg["wide_spread_threshold"],
        )

        # Safety check.
        if buy_price >= sell_price:
            return orders

        # Belt-and-suspenders soft-limit protection.
        if position >= SOFT_LIMIT:
            buy_size = 0

        if position <= -SOFT_LIMIT:
            sell_size = 0

        if buy_size > 0:
            orders.append(Order(product, buy_price, buy_size))

        if sell_size > 0:
            orders.append(Order(product, sell_price, -sell_size))

        return orders

    # --------------------------------------------------------
    # Main run function
    # --------------------------------------------------------

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}

        for product in TRANSLATOR_PRODUCTS:
            result[product] = self.market_make_translator(product, state)

        conversions = 0
        trader_data = ""

        return result, conversions, trader_data