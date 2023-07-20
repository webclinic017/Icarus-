from objects import ECause, Result, Trade, OCO, ECommand, TradeResult, Market
from strategies.StrategyBase import StrategyBase
import json
from utils import time_scale_to_minute
import position_sizing
from typing import Dict
import collections
from analyzers.market_classification import Direction

class TrendChangeMCI(StrategyBase):

    def __init__(self, _tag, _config, _symbol_info):
        super().__init__(_tag, _config, _symbol_info)
        self.max_loss_coeff = self.config['kwargs'].get('max_loss_coeff')
        self.target_profit_coeff = self.config['kwargs'].get('target_profit_coeff')
        self.exit_duration = self.config['kwargs'].get('exit_duration')

        self.enter_uptrend_th = self.config['kwargs'].get('enter_uptrend_th')
        self.exit_downtrend_th = self.config['kwargs'].get('exit_downtrend_th')

        return


    async def run(self, analysis_dict, lto_list, ikarus_time, strategy_capital):
        return await super().run_logic(self, analysis_dict, lto_list, ikarus_time, strategy_capital)


    async def make_decision(self, analysis_dict, ao_pair, ikarus_time, pairwise_alloc_share, **kwargs):


        time_dict = analysis_dict[ao_pair]
        if not is_change_to_uptrend(time_dict[self.min_period]['market_class_index'], self.enter_uptrend_th):
            return False
        
        enter_price = time_dict[self.min_period]['close'][-1]
        enter_ref_amount=pairwise_alloc_share

        enter_order = Market(amount=enter_ref_amount, price=enter_price)

        # Set decision_time to timestamp which is the open time of the current kline (newly started not closed kline)
        trade = Trade(int(ikarus_time), self.name, ao_pair, command=ECommand.EXEC_ENTER)
        trade.set_enter(enter_order)
        result = TradeResult()
        trade.result = result

        return trade


    async def on_exit_expire(self, trade: Trade, ikarus_time, analysis_dict, strategy_capital):
        # NOTE: Things to change: price, limitPrice, stopLimitPrice, expire date
        trade.command = ECommand.UPDATE
        trade.stash_exit()
        close_price = analysis_dict[trade.pair][self.min_period]['close'][-1]
        trade.set_exit( Market(quantity=trade.result.enter.quantity, price=close_price) )

        # Apply the filters
        # TODO: Add min notional fix (No need to add the check because we are not gonna do anything with that)
        if not StrategyBase.apply_exchange_filters(trade.exit, self.symbol_info[trade.pair]):
            # TODO: This is a critical case where the exit order failed to pass filters. Decide what to do
            return False
        return True


    async def on_open_exit(self, trade: Trade, ikarus_time: int, analysis_dict: Dict, strategy_capital):
        
        # Check exit downtrend condition 
        if not is_change_to_downtrend(analysis_dict[trade.pair][self.min_period]['market_class_index'], self.exit_downtrend_th):
            return True

        trade.command = ECommand.UPDATE
        trade.stash_exit()
        close_price = analysis_dict[trade.pair][self.min_period]['close'][-1]
        trade.set_exit( Market(quantity=trade.result.enter.quantity, price=close_price) )

        # Apply the filters
        # TODO: Add min notional fix (No need to add the check because we are not gonna do anything with that)
        if not StrategyBase.apply_exchange_filters(trade.exit, self.symbol_info[trade.pair]):
            # TODO: This is a critical case where the exit order failed to pass filters. Decide what to do
            return False
        return True


    async def on_waiting_exit(self, trade: Trade, ikarus_time: int, analysis_dict: Dict, strategy_capital):

        stop_loss_price = position_sizing.evaluate_stop_loss(strategy_capital, self.max_loss_coeff, trade)

        target_price = trade.result.enter.price * self.target_profit_coeff
        stop_price = trade.result.enter.price * 0.95

        # If the stop_price causes more capital loss than max_loss_percentage, than do the adjustment
        stop_limit_price = max(stop_loss_price, stop_price)
        stop_price = stop_limit_price*1.001
        #stop_limit_price = trade.result.enter.price * 0.949

        exit_oco_order = OCO(
            price=target_price,
            quantity=trade.result.enter.quantity,
            stop_price=stop_price,
            stop_limit_price=stop_limit_price,
            expire=StrategyBase._eval_future_candle_time(ikarus_time, self.exit_duration, time_scale_to_minute(self.min_period))
        )
        trade.set_exit(exit_oco_order)

        trade.command = ECommand.EXEC_EXIT
        if not StrategyBase.apply_exchange_filters(trade.exit, self.symbol_info[trade.pair]):
            return False
        return True

    async def on_closed(self, lto):
        return lto


def is_change_to_uptrend(mci, enter_uptrend_th):


    mci_counts_prev_1 = dict(collections.Counter(mci.iloc[-1]))
    mci_counts_prev_2 = dict(collections.Counter(mci.iloc[-2]))

    uptrend_possibility_1 = mci_counts_prev_1.get(Direction.UP,0) / len(mci.iloc[-1])
    uptrend_possibility_2 = mci_counts_prev_2.get(Direction.UP,0) / len(mci.iloc[-2])

    change_condition = [
        uptrend_possibility_1 >= enter_uptrend_th,
        uptrend_possibility_2 < enter_uptrend_th
    ]

    if all(change_condition):
        return True
    
    return False

def is_change_to_downtrend(mci, exit_downtrend_th):
    mci_counts_prev_1 = dict(collections.Counter(mci.iloc[-1]))
    mci_counts_prev_2 = dict(collections.Counter(mci.iloc[-2]))

    downtrend_possibility_1 = mci_counts_prev_1.get(Direction.DOWN,0) / len(mci.iloc[-1])
    downtrend_possibility_2 = mci_counts_prev_2.get(Direction.DOWN,0) / len(mci.iloc[-2])

    change_condition = [
        downtrend_possibility_1 >= exit_downtrend_th,
        downtrend_possibility_2 < exit_downtrend_th
    ]

    if all(change_condition):
        return True
    
    return False