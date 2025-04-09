import json
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import defaultdict, deque
import statistics
import math

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,         # When price deviates by this amount from fair value, take orders.
        "clear_width": 0,        # Width used when clearing positions.
        # For market making:
        "disregard_edge": 1,     # Disregard levels within this edge when joining or pennying.
        "join_edge": 2,          # Join orders within this edge.
        "default_edge": 4,       # Default offset if no levels qualify.
        "soft_position_limit": 10,  # Limit used when making orders.
    },
}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Observation) -> list[Any]:
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

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self, params: dict = None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.RAINFOREST_RESIN: 20}

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: list[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[int, int]:
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume
        
    
    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: list[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume
    
    def market_make(
        self,
        product: str,
        orders: list[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume
    
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[list[Order], int, int]:
        orders: list[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    
    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[list[Order], int, int]:
        orders: list[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ) -> tuple[list[Order], int, int]:
        orders: list[Order] = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    
    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: float, width: int, position: int, position_limit: int) -> list[Order]:
        orders: list[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Determine boundary prices for market making.
        baaf_candidates = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbbf_candidates = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(baaf_candidates) if len(baaf_candidates) > 0 else fair_value + 2
        bbbf = max(bbbf_candidates) if len(bbbf_candidates) > 0 else fair_value - 2

        # 1. Take Orders:
        buy_order_volume, sell_order_volume = self.take_best_orders(
            Product.RAINFOREST_RESIN, fair_value, self.params[Product.RAINFOREST_RESIN]["take_width"],
            orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        # 2. Clear Position Orders:
        buy_order_volume, sell_order_volume = self.clear_position_order(
            Product.RAINFOREST_RESIN, fair_value, self.params[Product.RAINFOREST_RESIN]["clear_width"],
            orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        # 3. Market Make:
        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN, orders, bbbf + 1, baaf - 1, position,
            buy_order_volume, sell_order_volume
        )

        return orders

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result = {}

        # Use the fair value from the parameters.
        resin_params = self.params[Product.RAINFOREST_RESIN]
        resin_fair_value = resin_params["fair_value"]
        resin_width = resin_params["clear_width"]  # Not directly used, but passed to clear orders.
        resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
        resin_position_limit = self.LIMIT[Product.RAINFOREST_RESIN]

        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_orders = self.rainforest_resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                resin_fair_value,
                resin_width,
                resin_position,
                resin_position_limit
            )
            result[Product.RAINFOREST_RESIN] = resin_orders

        conversions = 1
        traderData = ""
        return result, conversions, traderData
    
    # def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
    #     """
    #     Only method required. It takes all buy and sell orders for all symbols as an input,
    #     and outputs a list of orders to be sent
    #     """
    #     logger.print("Starting run at timestamp:", state.timestamp)

    #     result = {}
    #     conversions = 0
    #     trader_data = "Run completed successfully."

    #     order_size = 10
    #     limit = 100

    #     # Iterate over all the keys (the available products) contained in the order dephts
    #     for product in state.order_depths.keys():
    #         logger.print("Processing product:", product)
    #         # Initialize the list of Orders to be sent as an empty list
    #         orders: list[Order] = []
    #         # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
    #         order_depth: OrderDepth = state.order_depths[product]
            
    #         if product == "RAINFOREST_RESIN":
    #             best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    #             best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

    #             if best_bid is not None and best_ask is not None:
    #                 current_mid = 0.5 * (best_bid + best_ask)
    #             elif best_bid is not None:
    #                 current_mid = best_bid
    #             elif best_ask is not None:
    #                 current_mid = best_ask
    #             else:
    #                 continue
                
    #             self.prices_history[product].append(current_mid)
    #             logger.print("Updated mid price for resin:", current_mid)
    #             if len(self.prices_history[product]) >= 10:
    #                 historical = list(self.prices_history[product])
    #                 fair_value = statistics.mean(historical)
                    
    #                 if best_bid is not None and best_ask is not None:
    #                     spread = best_ask - best_bid
    #                 else:
    #                     spread = 0
    #                 offset = max(1, spread // 2)

    #                 buy_price = round(fair_value - offset)
    #                 sell_price = round(fair_value + offset)

    #                 position = state.position.get(product, 0)

    #                 buy_size = min(order_size, limit - position)
    #                 sell_size = min(order_size, limit + position)

    #                 if current_mid < fair_value:
    #                     orders.append(Order(product, buy_price, buy_size))
    #                 elif current_mid > fair_value:
    #                     orders.append(Order(product, sell_price, -sell_size))
    #             if orders:
    #                 result[product] = orders
            # Check if the current product is the 'PEARLS' product, only then run the order logic
            # if product == 'RAINFOREST_RESIN':
            #     # Define a fair value for the PEARLS.
            #     # Note that this value of 1 is just a dummy value, you should likely change it!
            #     acceptable_price = 10000

            #     # If statement checks if there are any SELL orders in the PEARLS market
            #     if len(order_depth.sell_orders) > 0:

            #         # Sort all the available sell orders by their price,
            #         # and select only the sell order with the lowest price
            #         best_ask = min(order_depth.sell_orders.keys())
            #         best_ask_volume = order_depth.sell_orders[best_ask]

            #         # Check if the lowest ask (sell order) is lower than the above defined fair value
            #         if best_ask < acceptable_price:

            #             # In case the lowest ask is lower than our fair value,
            #             # This presents an opportunity for us to buy cheaply
            #             # The code below therefore sends a BUY order at the price level of the ask,
            #             # with the same quantity
            #             # We expect this order to trade with the sell order
            #             logger.print("BUY", str(-best_ask_volume) + "x", best_ask)
            #             orders.append(Order(product, best_ask, -best_ask_volume))

            #     # The below code block is similar to the one above,
            #     # the difference is that it find the highest bid (buy order)
            #     # If the price of the order is higher than the fair value
            #     # This is an opportunity to sell at a premium
            #     if len(order_depth.buy_orders) != 0:
            #         best_bid = max(order_depth.buy_orders.keys())
            #         best_bid_volume = order_depth.buy_orders[best_bid]
            #         if best_bid > acceptable_price:
            #             logger.print("SELL", str(best_bid_volume) + "x", best_bid)
            #             orders.append(Order(product, best_bid, -best_bid_volume))

            #     # Add all the above the orders to the result dict
            #     result[product] = orders
            

            # if product == "KELP":
            #     if len(order_depth.buy_orders) != 0:
            #         best_bid = max(order_depth.buy_orders.keys())
            #     if len(order_depth.sell_orders) != 0:
            #         best_ask = min(order_depth.sell_orders.keys())
                
            #     if best_bid is not None and best_ask is not None:
            #         mid_price = 0.5 * (best_bid + best_ask)
            #     elif best_bid is not None:
            #         mid_price = best_bid
            #     elif best_ask is not None:
            #         mid_price = best_ask
            #     else:
            #         continue

            #     self.prices_history[product].append(mid_price)

            #     if len(self.prices_history[product]) >= 60:
            #         short_window = list(self.prices_history[product])[-20:]
            #         long_window = list(self.prices_history[product])

            #         sma_short = sum(short_window) / len(short_window)
            #         sma_long = sum(long_window) / len(long_window)
                    
            #         if sma_short > sma_long:
            #             if best_ask is not None:
            #                 orders.append(Order(product, best_ask, 5))
            #         else:
            #             if best_bid is not None:
            #                 orders.append(Order(product, best_bid, 5))   
            #     if orders:
            #         result[product] = orders

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above
        # logger.flush(state, result, conversions, trader_data)
        # return result, conversions, trader_data