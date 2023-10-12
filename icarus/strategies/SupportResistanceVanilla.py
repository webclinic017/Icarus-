from objects import Trade, ECommand, TradeResult, Limit, ECause
from strategies.StrategyBase import StrategyBase
from analyzers.market_classification import Direction
from utils import time_scale_to_minute
import numpy as np
from typing import Dict

class SupportResistanceVanilla(StrategyBase):

    def __init__(self, _tag, _config, _symbol_info):
        super().__init__(_tag, _config, _symbol_info)
        self.support_analyzer = self.config['kwargs'].get('support')
        self.resistance_analyzer = self.config['kwargs'].get('resistance')
        return


    async def run(self, analysis_dict, lto_list, ikarus_time, strategy_capital):
        return await super().run_logic(self, analysis_dict, lto_list, ikarus_time, strategy_capital)


    async def make_decision(self, analysis_dict, ao_pair, ikarus_time, pairwise_alloc_share, **kwargs):


        analysis = analysis_dict[ao_pair][self.min_period]
        supports = analysis[self.support_analyzer]
        #resistances = analysis[self.analyzer]['resistance']
        #sup_res = supports + resistances
        #sup_res.sort()

        support_relative_pos = np.array([round(100 * (sr.price_mean - analysis['close'][-1]) / analysis['close'][-1], 2) for sr in supports])

        entry_conditions = [
            any(support_relative_pos < 0 )             # No support at below
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
        #supports = analysis[self.analyzer]['support']
        resistances = analysis[self.resistance_analyzer]
        resistances.reverse()
        #sup_res = supports + resistances
        #sup_res.sort()

        resistance_relative_pos = np.array([round(100 * (sr.price_mean - analysis['close'][-1]) / analysis['close'][-1], 2) for sr in resistances])

        # There is resistance above
        if not any(resistance_relative_pos > 0):
            return True

        closest_above_res_idx = len(resistance_relative_pos[resistance_relative_pos > 0]) - 1
        
        # Find profitable resistance level by iterating closest above resistance to remotest
        exit_price = None
        for i in reversed(range(closest_above_res_idx+1)):
            if resistances[i].price_mean > trade.result.enter.price:
                exit_price = resistances[i].price_mean
                break
        
        # There is resistance above
        if exit_price == None:
            return True
   
        exit_limit_order = Limit(
            exit_price,
            quantity=trade.result.enter.quantity,
            expire=StrategyBase._eval_future_candle_time(ikarus_time, 24, time_scale_to_minute(self.min_period))
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

        analysis = analysis_dict[trade.pair][self.min_period]
        resistances = analysis[self.resistance_analyzer]
        resistances.reverse()

        resistance_relative_pos = np.array([round(100 * (sr.price_mean - analysis['close'][-1]) / analysis['close'][-1], 2) for sr in resistances])

        # There is resistance above
        if not any(resistance_relative_pos > 0):
            return True

        # Do not look for profitable exit anymore
        exit_price = resistances[0].price_mean
   
        exit_limit_order = Limit(
            exit_price,
            quantity=trade.result.enter.quantity,
            expire=StrategyBase._eval_future_candle_time(ikarus_time, 24, time_scale_to_minute(self.min_period))
        )
        trade.set_exit(exit_limit_order)

        trade.command = ECommand.UPDATE

        # Apply the filters
        # TODO: Add min notional fix (No need to add the check because we are not gonna do anything with that)
        if not StrategyBase.apply_exchange_filters(trade.exit, self.symbol_info[trade.pair]):
            # TODO: This is a critical case where the exit order failed to pass filters. Decide what to do
            return False

        return True

    async def on_closed(self, lto):
        return lto
