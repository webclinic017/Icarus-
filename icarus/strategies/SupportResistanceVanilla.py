from objects import Trade, ECommand, TradeResult, Market
from strategies.StrategyBase import StrategyBase
from typing import Dict
from analyzers.market_classification import Direction

class SupportResistanceVanilla(StrategyBase):

    def __init__(self, _tag, _config, _symbol_info):
        super().__init__(_tag, _config, _symbol_info)
        self.analyzer = self.config['kwargs'].get('analyzer')
        return


    async def run(self, analysis_dict, lto_list, ikarus_time, strategy_capital):
        return await super().run_logic(self, analysis_dict, lto_list, ikarus_time, strategy_capital)


    async def make_decision(self, analysis_dict, ao_pair, ikarus_time, pairwise_alloc_share, **kwargs):


        analysis = analysis_dict[ao_pair][self.min_period]
        supports = analysis[self.analyzer]['support']
        resistances = analysis[self.analyzer]['resistance']
        sup_res = supports + resistances
        sup_res.sort()

        pos_in_supports = [support.relative_position(analysis['close'][-1])  for support in supports]
        pos_in_resistances = [resistance.relative_position(analysis['close'][-1])  for resistance in resistances]

        # TODO: Develop entry evaluating logic based on sup res
        if not all([]):
            return False
        
        enter_price = analysis['close'][-1]
        enter_ref_amount=pairwise_alloc_share

        enter_order = Market(amount=enter_ref_amount, price=enter_price)

        # Set decision_time to timestamp which is the open time of the current kline (newly started not closed kline)
        trade = Trade(int(ikarus_time), self.name, ao_pair, command=ECommand.EXEC_ENTER)
        trade.set_enter(enter_order)
        result = TradeResult()
        trade.result = result

        return trade


    async def on_open_exit(self, trade: Trade, ikarus_time: int, analysis_dict: Dict, strategy_capital):
        return True


    async def on_waiting_exit(self, trade: Trade, ikarus_time: int, analysis_dict: Dict, strategy_capital):

        analysis = analysis_dict[trade.pair][self.min_period]
        current_direction = analysis[self.analyzer][-1]

        exit_conditions = [
            current_direction == Direction.DOWN
        ]

        # Check exit downtrend condition 
        if not all(exit_conditions):
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

    async def on_closed(self, lto):
        return lto
