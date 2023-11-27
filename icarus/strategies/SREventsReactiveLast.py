from objects import Trade, ECommand, TradeResult, Market, ECause
from strategies.StrategyBase import StrategyBase
from analyzers.market_classification import Direction
from utils import time_scale_to_minute
import numpy as np
from typing import Dict
from analyzers.support_resistance import SREventType

class SREventsReactiveLast(StrategyBase):

    def __init__(self, _tag, _config, _symbol_info):
        super().__init__(_tag, _config, _symbol_info)
        self.support_analyzer = self.config['kwargs'].get('support')
        self.resistance_analyzer = self.config['kwargs'].get('resistance')
        self.exit_duration = self.config['kwargs'].get('exit_duration', 24)
        return


    async def run(self, analysis_dict, lto_list, ikarus_time, strategy_capital):
        return await super().run_logic(self, analysis_dict, lto_list, ikarus_time, strategy_capital)


    async def make_decision(self, analysis_dict, ao_pair, ikarus_time, pairwise_alloc_share, **kwargs):

        analysis = analysis_dict[ao_pair][self.min_period]
        supports = analysis[self.support_analyzer]

        # Find the SRClusters that has bounce event
        bounce_event_happened = False
        for cluster in supports:
            if cluster.events == []:
                continue

            last_event = cluster.events[-1]
            enter_condition = [
                last_event.type == SREventType.BOUNCE,
                last_event.after == 1,                      # Price should come and go from/to the top, compare to the cluster
                last_event.end_index == cluster.chunk_end_index-1      
                # NOTE: We only notice that en event is concluded when there is an non-event candle occured. Tha
                # is the difference between the IN_ZONE and other events. That is the reason why "cluster.chunk_end_index-1" 
            ]

            if all(enter_condition):
                bounce_event_happened = True
                break

        if not bounce_event_happened:
            return False
        
        enter_price = analysis['close'][-1]
        enter_ref_amount = pairwise_alloc_share

        enter_order = Market(amount=enter_ref_amount, price=enter_price)

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
        resistances = analysis[self.resistance_analyzer]

        # Find the SRClusters that has bounce event
        bounce_event_happened = False
        for cluster in resistances:
            if cluster.events == []:
                continue

            last_event = cluster.events[-1]
            exit_conditions = [
                last_event.type == SREventType.BOUNCE,
                last_event.after == -1,                      # Price should come and go from/to the bottom, compare to the cluster
                last_event.end_index == cluster.chunk_end_index-1     
                # NOTE: We only notice that en event is concluded when there is an non-event candle occured. Tha
                # is the difference between the IN_ZONE and other events. That is the reason why "cluster.chunk_end_index-1" 
            ]
            if all(exit_conditions):
                bounce_event_happened = True
                break

        if not bounce_event_happened:
            return True
   
        close_price = analysis_dict[trade.pair][self.min_period]['close'][-1]
        trade.set_exit( Market(quantity=trade.result.enter.quantity, price=close_price) )
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
        return True

    async def on_closed(self, lto):
        return lto
