from ..objects import ECause, Result, Trade, OCO, ECommand, TradeResult, Market
from .StrategyBase import StrategyBase
import json
from ..utils import time_scale_to_minute

class FixedOCOTarget(StrategyBase):

    def __init__(self, _config, _symbol_info, _resource_allocator):
        super().__init__("FixedOCOTarget", _config, _symbol_info, _resource_allocator)
        return


    async def run(self, analysis_dict, lto_list, ikarus_time, strategy_capital):
        return await super().run_logic(self, analysis_dict, lto_list, ikarus_time, strategy_capital)


    async def make_decision(self, analysis_dict, ao_pair, ikarus_time, pairwise_alloc_share):

            time_dict = analysis_dict[ao_pair]

            enter_price = time_dict[self.min_period]['close'][-1]
            enter_ref_amount=pairwise_alloc_share

            enter_order = Market(amount=enter_ref_amount, price=enter_price)

            # Set decision_time to timestamp which is the open time of the current kline (newly started not closed kline)
            trade = Trade(int(ikarus_time), self.name, ao_pair, command=ECommand.EXEC_ENTER)
            trade.set_enter(enter_order)
            result = TradeResult()
            trade.result = result

            return trade


    async def on_update(self, trade, ikarus_time, **kwargs):
        # NOTE: Things to change: price, limitPrice, stopLimitPrice, expire date
        trade.command = ECommand.UPDATE
        trade.stash_exit()
        close_price = kwargs['analysis_dict'][trade.pair][self.min_period]['close'][-1]
        trade.set_exit( Market(quantity=trade.result.enter.quantity, price=close_price) )


        # Apply the filters
        # TODO: Add min notional fix (No need to add the check because we are not gonna do anything with that)
        if not StrategyBase.apply_exchange_filters(trade.exit, self.symbol_info[trade.pair]):
            # TODO: This is a critical case where the exit order failed to pass filters. Decide what to do
            return False
        return True


    async def on_waiting_exit(self, trade, analysis_dict, **kwargs):


        target_price = trade.result.enter.price * 1.01
        stop_price = trade.result.enter.price * 0.95
        stop_limit_price = trade.result.enter.price * 0.949

        exit_oco_order = OCO(
            price=target_price,
            quantity=trade.result.enter.quantity,
            stop_price=stop_price,
            stop_limit_price=stop_limit_price,
            expire=StrategyBase._eval_future_candle_time(kwargs['ikarus_time'],24,time_scale_to_minute(self.min_period))
        )
        trade.set_exit(exit_oco_order)

        trade.command = ECommand.EXEC_EXIT
        if not StrategyBase.apply_exchange_filters(trade.exit, self.symbol_info[trade.pair]):
            return False
        return True

    async def on_closed(self, lto):
        return lto