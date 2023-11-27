from objects import Trade, ECommand, TradeResult, Market, ECause, Limit, OCO, EState
from strategies.StrategyBase import StrategyBase
from utils import time_scale_to_minute
import numpy as np
from typing import Dict
from analyzers.support_resistance import SREventType
from itertools import chain
import position_sizing
import logging

logger = logging.getLogger('app')

class SREventsPredictiveVanilla(StrategyBase):

    def __init__(self, _tag, _config, _symbol_info):
        super().__init__(_tag, _config, _symbol_info)

        self.strategy_config = self.config['kwargs']
        return


    async def run(self, analysis_dict, lto_list, ikarus_time, strategy_capital):
        return await super().run_logic(self, analysis_dict, lto_list, ikarus_time, strategy_capital)


    async def make_decision(self, analysis_dict, ao_pair, ikarus_time, pairwise_alloc_share, **kwargs):

        analysis = analysis_dict[ao_pair][self.min_period]
        enter_clusters = np.array(list(chain(*[analysis[analyzer] for analyzer in self.strategy_config['enter_analyzers']])))

        cluster_relative_pos = np.array([round(100 * (sr.price_mean - analysis['close'][-1]) / analysis['close'][-1], 2) for sr in enter_clusters])
        below_clusters = enter_clusters[cluster_relative_pos < 0]
        #below_cluster_pos = cluster_relative_pos[cluster_relative_pos < 0]

        entry_conditions = [
            len(below_clusters) > 0,
            -len(below_clusters) <= self.strategy_config['enter_cluster_index'] < len(below_clusters)
        ]

        if not all(entry_conditions):
            return False

        enter_price = below_clusters[self.strategy_config['enter_cluster_index']].price_mean
        enter_ref_amount=pairwise_alloc_share

        enter_order = Limit(
            creation_time=ikarus_time,
            price=enter_price,
            amount=enter_ref_amount,
            expire=StrategyBase._eval_future_candle_time(ikarus_time, self.strategy_config['enter_expire_period'], time_scale_to_minute(self.min_period))
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
        '''
            The clusters below the entry level is eliminated
            The clusters below the current level is eliminated
            exit_cluster_index = 0 # Closest above cluster
            exit_cluster_index = -1 # Remotest above cluster
        '''
        analysis = analysis_dict[trade.pair][self.min_period]
        exit_clusters = np.array(list(chain(*[analysis[analyzer] for analyzer in self.strategy_config['exit_analyzers']])))

        base_price = max(trade.result.enter.price, analysis['close'][-1])
        cluster_relative_pos = np.array([round(100 * (sr.price_mean - base_price) / base_price, 2) for sr in exit_clusters])
        above_clusters = exit_clusters[cluster_relative_pos > 0.05]

        # If there is no upper level
        if len(above_clusters) <= 0:
            return True
        
        # If the given exit_cluster_index is not valid
        if not (-len(above_clusters) <= self.strategy_config['exit_cluster_index'] < len(above_clusters)):
            return True
        
        exit_price = above_clusters[self.strategy_config['enter_cluster_index']].price_mean

        stop_loss_price = position_sizing.evaluate_stop_loss(strategy_capital, self.strategy_config['max_loss_coeff'], trade)
        stop_price = trade.result.enter.price * 0.99
        stop_limit_price = max(stop_loss_price, stop_price)
        stop_price = stop_price*1.001

        exit_oco_order = OCO(
            creation_time=ikarus_time,
            price=exit_price,
            quantity=trade.result.enter.quantity,
            stop_price=stop_price,
            stop_limit_price=stop_limit_price,
            expire=StrategyBase._eval_future_candle_time(ikarus_time, self.strategy_config['exit_expire_period'], time_scale_to_minute(self.min_period))
        )
        trade.set_exit(exit_oco_order)
        trade.command = ECommand.EXEC_EXIT

        if not StrategyBase.apply_exchange_filters(trade.exit, self.symbol_info[trade.pair]):
            # TODO: This is a critical case where the exit order failed to pass filters. Decide what to do
            return False
        return True

    async def on_enter_expire(self, trade):
        trade.command = ECommand.CANCEL
        trade.result.cause = ECause.ENTER_EXP
        return True

    async def on_exit_expire(self, trade: Trade, ikarus_time: int, analysis_dict: Dict, strategy_capital):

        analysis = analysis_dict[trade.pair][self.min_period]
        exit_clusters = np.array(list(chain(*[analysis[analyzer] for analyzer in self.strategy_config['exit_analyzers']])))

        base_price = max(trade.result.enter.price, analysis['close'][-1])
        cluster_relative_pos = np.array([round(100 * (sr.price_mean - base_price) / base_price, 2) for sr in exit_clusters])
        above_clusters = exit_clusters[cluster_relative_pos > 0]

        # If there is no upper level
        if len(above_clusters) == 0:
            trade.command = ECommand.CANCEL
            trade.status = EState.WAITING_EXIT
            logger.debug('No above cluster: {}'.format(trade._id))
            return True
        
        # If the given exit_cluster_index is not valid
        if not (-len(above_clusters) <= self.strategy_config['exit_cluster_index'] < len(above_clusters)):
            trade.command = ECommand.CANCEL
            trade.status = EState.WAITING_EXIT
            logger.debug('exit_cluster_index is not valid: {}'.format(trade._id))
            return True
        
        trade.stash_exit()

        exit_price = above_clusters[self.strategy_config['enter_cluster_index']].price_mean

        stop_loss_price = position_sizing.evaluate_stop_loss(strategy_capital, self.strategy_config['max_loss_coeff'], trade)
        stop_price = trade.result.enter.price * 0.99
        stop_limit_price = max(stop_loss_price, stop_price)
        stop_price = stop_price*1.001

        exit_oco_order = OCO(
            creation_time=ikarus_time,
            price=exit_price,
            quantity=trade.result.enter.quantity,
            stop_price=stop_price,
            stop_limit_price=stop_limit_price,
            expire=StrategyBase._eval_future_candle_time(ikarus_time, self.strategy_config['exit_expire_period'], time_scale_to_minute(self.min_period))
        )
        trade.set_exit(exit_oco_order)
        trade.command = ECommand.UPDATE

        if not StrategyBase.apply_exchange_filters(trade.exit, self.symbol_info[trade.pair]):
            # TODO: This is a critical case where the exit order failed to pass filters. Decide what to do
            return False
        return True


    async def on_closed(self, lto):
        return lto
