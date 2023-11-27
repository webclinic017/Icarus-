from objects import Trade, ECommand, TradeResult, Market, ECause, Limit, OCO
from strategies.StrategyBase import StrategyBase
from utils import time_scale_to_second
import numpy as np
from typing import Dict
from analyzers.support_resistance import SREventType
from itertools import chain
import position_sizing
from datetime import datetime

class Benchmark(StrategyBase):

    def __init__(self, _tag, _config, _symbol_info):
        super().__init__(_tag, _config, _symbol_info)

        self.strategy_config = self.config['kwargs']

        session_start_time = datetime.strptime(self.strategy_config['enter_time'], "%Y-%m-%d %H:%M:%S")
        self.enter_time = int(datetime.timestamp(session_start_time)) # UTC

        session_exit_time = datetime.strptime(self.strategy_config['exit_time'], "%Y-%m-%d %H:%M:%S")
        self.exit_time = int(datetime.timestamp(session_exit_time)) # UTC
        return


    async def run(self, analysis_dict, lto_list, ikarus_time, strategy_capital):
        return await super().run_logic(self, analysis_dict, lto_list, ikarus_time, strategy_capital)


    async def make_decision(self, analysis_dict, ao_pair, ikarus_time, pairwise_alloc_share, **kwargs):
        
        entry_conditions = [
            ikarus_time == self.enter_time
        ]

        if not all(entry_conditions):
            return False

        analysis = analysis_dict[ao_pair][self.min_period]
        enter_price = analysis['close'][-1]
        enter_ref_amount = pairwise_alloc_share
        enter_order = Market(creation_time=ikarus_time, amount=enter_ref_amount, price=enter_price)

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

        exit_conditions = [
            ikarus_time == self.exit_time
        ]

        if not all(exit_conditions):
            return True

        analysis = analysis_dict[trade.pair][self.min_period]
        close_price = analysis['close'][-1]
        trade.set_exit( Market(creation_time=ikarus_time, quantity=trade.result.enter.quantity, price=close_price) )
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
        if trade.command == ECommand.EXEC_EXIT:
            trade.command = ECommand.UPDATE
        return is_success

    async def on_closed(self, lto):
        return lto
