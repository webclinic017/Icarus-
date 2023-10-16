from objects import Trade, ECommand, TradeResult, Limit, ECause, Market
from strategies.StrategyBase import StrategyBase
from analyzers.market_classification import Direction
from utils import time_scale_to_minute
import numpy as np
from typing import Dict

class SupportResistanceMarketFiltered(StrategyBase):

    def __init__(self, _tag, _config, _symbol_info):
        super().__init__(_tag, _config, _symbol_info)
        self.support_analyzer = self.config['kwargs'].get('support')
        self.resistance_analyzer = self.config['kwargs'].get('resistance')
        self.diretion_analyzer = self.config['kwargs'].get('direction')
        self.exit_duration = self.config['kwargs'].get('exit_duration', 24)
        return


    async def run(self, analysis_dict, lto_list, ikarus_time, strategy_capital):
        return await super().run_logic(self, analysis_dict, lto_list, ikarus_time, strategy_capital)


    async def make_decision(self, analysis_dict, ao_pair, ikarus_time, pairwise_alloc_share, **kwargs):


        analysis = analysis_dict[ao_pair][self.min_period]
        supports = analysis[self.support_analyzer]
        direction = analysis[self.diretion_analyzer][-1]

        support_relative_pos = np.array([round(100 * (sr.price_mean - analysis['close'][-1]) / analysis['close'][-1], 2) for sr in supports])

        entry_conditions = [
            any(support_relative_pos < 0 ),             # No support at below
            direction == Direction.UP
        ]

        if not all(entry_conditions):
            return False
        
        # Note support levels are already ordered. Lowest one is on the 0, and the highest one is on -1
        closest_below_sup_idx = len(support_relative_pos[support_relative_pos < 0]) - 1

        enter_price = supports[closest_below_sup_idx].price_mean
        enter_ref_amount=pairwise_alloc_share

        enter_order = Limit(
            price=enter_price,
            amount=enter_ref_amount,
            expire=StrategyBase._eval_future_candle_time(ikarus_time,24,time_scale_to_minute(self.min_period))
        )

        # Set decision_time to timestamp which is the open time of the current kline (newly started not closed kline)
        trade = Trade(int(ikarus_time), self.name, ao_pair, command=ECommand.EXEC_ENTER)
        trade.set_enter(enter_order)
        result = TradeResult()
        trade.result = result

        return trade


    async def on_open_enter(self, trade: Trade, ikarus_time: int, analysis_dict: Dict, strategy_capital):
        return True


    async def on_open_exit(self, trade: Trade, ikarus_time: int, analysis_dict: Dict, strategy_capital):
        return True


    async def on_waiting_exit(self, trade: Trade, ikarus_time: int, analysis_dict: Dict, strategy_capital):

        analysis = analysis_dict[trade.pair][self.min_period]
        direction = analysis[self.diretion_analyzer][-1]
        resistances = analysis[self.resistance_analyzer].copy()     # [ 1, 2, 4, 7]
        #resistances.reverse()                                       # [ 7, 4, 2, 1]

        resistance_relative_pos = np.array([round(100 * (sr.price_mean - analysis['close'][-1]) / analysis['close'][-1], 2) for sr in resistances])

        market_exit_conditions = [
            not any(resistance_relative_pos > 0),       # Also covers the case where there is no resistance at all
            direction == Direction.DOWN
        ]

        # Exit immediately if there is no resistance above and direction is down
        if all(market_exit_conditions):
            close_price = analysis_dict[trade.pair][self.min_period]['close'][-1]
            trade.set_exit( Market(quantity=trade.result.enter.quantity, price=close_price) )
            trade.command = ECommand.EXEC_EXIT
            return True

        closest_above_res_idx = np.argmax(resistance_relative_pos > 0)
        # NOTE: Ignore the entry price
        if direction == Direction.DOWN:
            # Exit from the closest above resistance
            exit_price = resistances[closest_above_res_idx].price_mean

        elif direction == Direction.UP:
            # Exit from the remotest above resistance
            exit_price = resistances[-1].price_mean

        else: #direction == Direction.SIDE:
            # Exit from the closest above resistance
            exit_price = resistances[closest_above_res_idx].price_mean

        exit_limit_order = Limit(
            exit_price,
            quantity=trade.result.enter.quantity,
            expire=StrategyBase._eval_future_candle_time(ikarus_time, self.exit_duration, time_scale_to_minute(self.min_period))
        )
        trade.set_exit(exit_limit_order)

        trade.command = ECommand.EXEC_EXIT

        # Apply the filters
        # TODO: Add min notional fix (No need to add the check because we are not gonna do anything with that)
        if not StrategyBase.apply_exchange_filters(trade.exit, self.symbol_info[trade.pair]):
            # TODO: This is a critical case where the exit order failed to pass filters. Decide what to do
            return False
        return True

    async def on_enter_expire(self, trade):
        trade.command = ECommand.CANCEL
        trade.result.cause = ECause.ENTER_EXP
        return True

    async def on_exit_expire(self, trade: Trade, ikarus_time: int, analysis_dict: Dict, strategy_capital):
        trade.stash_exit()
        is_success = await self.on_waiting_exit(trade, ikarus_time, analysis_dict, strategy_capital)
        trade.command = ECommand.UPDATE
        return is_success

    async def on_closed(self, lto):
        return lto
