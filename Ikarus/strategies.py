import logging
import statistics as st
import json
from Ikarus.objects import GenericObject, ObjectEncoder
from binance.helpers import round_step_size
from Ikarus.enums import *
import bson
import copy
import abc
import time

class StrategyManager():

    def __init__(self, _config , _symbol_info) -> None:

        self.strategies = {
            "OCOBackTest": OCOBackTest,
            "AlwaysEnter": AlwaysEnter,
            "AlwaysEnter90": AlwaysEnter90,
            "FallingKnifeCatcher": FallingKnifeCatcher,
        }

        self.strategy_list = []
        for strategy_name in _config['strategy'].keys():
            strategy_class = self.strategies[strategy_name]
            self.strategy_list.append(strategy_class(_config, _symbol_info))
        pass

    def get_strategies(self): return self.strategy_list

    def remove_strategy(self): return True

    def add_strategy(self): return True

class StrategyBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'run') and 
                callable(subclass.run) and 
                hasattr(subclass, 'dump_to') and 
                callable(subclass.dump_to) or 
                NotImplemented)

    @abc.abstractmethod
    async def run(self, analysis_dict, lto_list, df_balance, dt_index=None):
        """Load in the data set"""
        raise NotImplementedError

    @staticmethod
    def _eval_future_candle_time(start_time, count, minute): return bson.Int64(start_time + count*minute*60*1000)


    @staticmethod
    async def _postpone(lto, phase, type, expire_time):
        lto['action'] = ACTN_POSTPONE
        lto[phase][type]['expire'] = expire_time
        return lto


    @staticmethod
    async def _config_market_exit(lto, type):

        lto['action'] = ACTN_MARKET_EXIT
        lto['exit'][TYPE_MARKET] = {
            'amount': lto['exit'][type]['amount'],
            'quantity': lto['exit'][type]['quantity'],
            'orderId': '',
        }
        return lto


    @staticmethod
    async def _create_enter_module(type, enter_price, enter_quantity, enter_ref_amount, expire_time):

        if type == TYPE_LIMIT:
            enter_module = {
                "limit": {
                    "price": float(enter_price),
                    "quantity": float(enter_quantity),
                    "amount": float(enter_ref_amount),
                    "expire": expire_time,
                    "orderId": ""
                    },
                }
        elif type == TYPE_MARKET:
            # TODO: Create TYPE_MARKET orders to enter
            pass
        else: pass # Internal Error
        return enter_module


    @staticmethod
    async def _create_exit_module(type, enter_price, enter_quantity, exit_price, exit_ref_amount, expire_time):

        if type == TYPE_OCO:
            exit_module = {
                "oco": {
                    "limitPrice": float(exit_price),
                    "stopPrice": float(enter_price)*0.995,           # Auto-execute stop loss if the amount go below %0.05
                    "stopLimitPrice": float(enter_price)*0.994,      # Lose max %0.06 of the amount
                    "quantity": float(enter_quantity),
                    "amount": float(exit_ref_amount),
                    "expire": expire_time,
                    "orderId": "",
                    "stopLimit_orderId": ""
                }
            }
        elif type == TYPE_LIMIT:
            exit_module = {
                "limit": {
                    "price": float(exit_price),
                    "quantity": float(enter_quantity),
                    "amount": float(exit_ref_amount),
                    "expire": expire_time,
                    "orderId": ""
                    },
                }
        elif type == TYPE_MARKET:
            pass
        else: pass # Internal Error
        return exit_module

    @staticmethod
    async def apply_exchange_filters(phase, type, module, symbol_info, exit_qty=0):
        """
        - Call this method prior to any order placement
        - Apply the filter of exchange pair
        - This methhod does not check if the current conditiones are good to go.
            If a filter is not satisfied then it would create an exception. Validation
            costs time. Maybe in future 
        - Separating enter and exit does not make any sense since the filters are valid for both side.
        Returns:
            dict: enter or exit module
        """ 
        
        if phase == 'enter':
            module['price'] = round_step_size(module['price'], float(symbol_info['filters'][0]['tickSize']))                            # Fixing PRICE_FILTER: tickSize
            module['quantity'] = round_step_size(module['amount'] /module['price'], float(symbol_info['filters'][2]['minQty']))         # Fixing LOT_SIZE: minQty

        elif phase == 'exit':
            if type == TYPE_OCO:
                module['limitPrice'] = round_step_size(module['limitPrice'], float(symbol_info['filters'][0]['tickSize']))              # Fixing PRICE_FILTER: tickSize
                module['stopPrice'] = round_step_size(module['stopPrice'], float(symbol_info['filters'][0]['tickSize']))
                module['stopLimitPrice'] = round_step_size(module['stopLimitPrice'], float(symbol_info['filters'][0]['tickSize']))
                module['quantity'] = round_step_size(exit_qty, float(symbol_info['filters'][2]['minQty']))                              # NOTE: Enter quantity will be used to exit
            
            elif type == TYPE_LIMIT:
                module['price'] = round_step_size(module['price'], float(symbol_info['filters'][0]['tickSize']))
                module['quantity'] = round_step_size(exit_qty, float(symbol_info['filters'][2]['minQty']))                              # NOTE: Enter quantity will be used to exit

        else: pass

        return module


    @staticmethod
    async def check_min_notional(price, quantity, symbol_info): return ((price*quantity) > float(symbol_info['filters'][3]['minNotional']))  # if valid: True, else: False


class FallingKnifeCatcher(StrategyBase):

    def __init__(self, _config, _symbol_info={}):
        self.name = "FallingKnifeCatcher"
        self.logger = logging.getLogger('{}.{}'.format(__name__,self.name))
        self.config = _config['strategy'][self.name]

        self.quote_currency = _config['broker']['quote_currency']
        self.scales_in_minute = _config['data_input']['scales_in_minute']
        # TODO: Make proper handling for symbol_info
        self.symbol_info = _symbol_info

        return


    async def _handle_lto(self, lto, dt_index):
        """
        This function decides what to do for the LTOs based on their 'status'
        """        
        skip_calculation = False
        
        if lto['status'] == STAT_ENTER_EXP:
            if self.config['action_mapping'][STAT_ENTER_EXP] == ACTN_CANCEL or lto['history'].count(STAT_ENTER_EXP) > 1:
                lto['action'] = ACTN_CANCEL
                lto['result']['cause'] = STAT_ENTER_EXP

            elif self.config['action_mapping'][STAT_ENTER_EXP] == ACTN_POSTPONE and lto['history'].count(STAT_ENTER_EXP) <= 1:
                # NOTE: postponed_candles = 1 means 2 candle
                #       If only 1 candle is desired to be postponed, then it means we will wait for newly started candle to close so postponed_candles will be 0
                postponed_candles = 1
                lto = await StrategyBase._postpone(lto,'enter', self.config['enter']['type'], StrategyBase._eval_future_candle_time(dt_index,postponed_candles,self.scales_in_minute[0])) 
                skip_calculation = True
            else: pass

        elif lto['status'] == STAT_EXIT_EXP:
            if self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_MARKET_EXIT or lto['history'].count(STAT_EXIT_EXP) > 1:
                lto = await StrategyBase._config_market_exit(lto, self.config['exit']['type'])
                self.logger.info(f'LTO: market exit configured') # TODO: Add orderId

            elif self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_POSTPONE and lto['history'].count(STAT_EXIT_EXP) <= 1:
                postponed_candles = 1
                lto = await StrategyBase._postpone(lto,'exit', self.config['exit']['type'], StrategyBase._eval_future_candle_time(dt_index,postponed_candles,self.scales_in_minute[0]))
                skip_calculation = True

            elif self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_UPDATE:
                # TODO: Give a call to methods that calculates exit point
                # NOTE: Things to change: price, limitPrice, stopLimitPrice, expire date
                lto['action'] = ACTN_UPDATE 
                lto['update_history'].append(copy.deepcopy(lto['exit'][self.config['exit']['type']])) # This is good for debugging and will be good for perf. evaluation in future
                if self.config['exit']['type'] == TYPE_LIMIT:
                    lto['exit'][TYPE_LIMIT]['price'] *= 0.99
                    lto['exit'][TYPE_LIMIT]['amount'] = lto['exit'][TYPE_LIMIT]['price'] * lto['exit'][TYPE_LIMIT]['quantity']
                    lto['exit'][TYPE_LIMIT]['expire'] = StrategyBase._eval_future_candle_time(dt_index,3,self.scales_in_minute[0])

                elif self.config['exit']['type'] == TYPE_OCO:
                    lto['exit'][TYPE_OCO]['limitPrice'] *= 0.99
                    lto['exit'][TYPE_OCO]['amount'] = lto['exit'][TYPE_OCO]['limitPrice'] * lto['exit'][TYPE_OCO]['quantity']
                    lto['exit'][TYPE_OCO]['expire'] = StrategyBase._eval_future_candle_time(dt_index,3,self.scales_in_minute[0])
                skip_calculation = True

                # Apply the filters
                # TODO: Add min notional fix (No need to add the check because we are not gonna do anything with that)
                lto['exit'][self.config['exit']['type']] = await StrategyBase.apply_exchange_filters('exit', 
                                                                                                    self.config['exit']['type'], 
                                                                                                    lto['exit'][self.config['exit']['type']], 
                                                                                                    self.symbol_info, 
                                                                                                    exit_qty=lto['result']['enter']['quantity'])

            else: pass

        elif lto['status'] == STAT_WAITING_EXIT:
            # LTO is entered succesfully, so exit order should be executed
            # TODO: expire of the exit_module can be calculated after the trade entered

            lto['action'] = ACTN_EXEC_EXIT
            lto['exit'][self.config['exit']['type']] = await StrategyBase.apply_exchange_filters('exit', 
                                                                                                self.config['exit']['type'], 
                                                                                                lto['exit'][self.config['exit']['type']], 
                                                                                                self.symbol_info, 
                                                                                                exit_qty=lto['result']['enter']['quantity'])
            skip_calculation = True

        elif lto['status'] != STAT_CLOSED:
            # If the status is not closed, just skip the iteration. otherwise go on to make a decision
            # NOTE: This logic contains the status: STAT_OPEN_EXIT, STAT_OPEN_ENTER, STAT_PART_CLOSED_ENTER, STAT_PART_CLOSED_EXIT
            skip_calculation = True

        return skip_calculation, lto


    async def run(self, analysis_dict, lto_list, df_balance, dt_index=None):
        """
        It requires to feed analysis_dict and lto_dict so that it may decide to:
        - not to enter a new trade if there is already an open trade
        - cancel the trade if an drawdown is detected

        Args:
            analysis_dict (dict): analysis.json
            - analysis objects contains where to buy and and where to sell

            lto_list (list): only the ltos that belongs to this strategy

            df_balance (pd.DataFrame): live-trade-objects coming from the [live-trades]

            dt_index (int): timestamp in ms for trade_object identifier
            
        Returns:
            list: nto_list
        """
        # Preliminary condition: all of the config['pairs'] exist in analysis_dict
        if not set(self.config['pairs']).issubset(analysis_dict.keys()):
            self.logger.warn(f"Configured pair \"{self.config['pairs']}\" does not exist in analysis_dict. Skipping {self.name}.run")
            return []

        # Initialize trade_dict to be filled
        trade_objects = []

        # Create a mapping between the pair and orderId such as {'BTCUSDT':['123','456']}
        pair_key_mapping = {}
        for i, lto in enumerate(lto_list):
            pair = lto['pair']
            if pair not in pair_key_mapping.keys():
                pair_key_mapping[pair] = []

            pair_key_mapping[pair].append(i)

        # This implementation enable to check number of trades and compare the value with the one in the config file.

        # NOTE: Only iterate for the configured pairs. Do not run the strategy if any of them is missing in analysis_dict
        for ao_pair in self.config['pairs']:

            # Check if there is already an LTO that has that 'pair' item. If so skip the evaluation (one pair one LTO rule)
            if ao_pair in pair_key_mapping.keys():

                # NOTE: If a pair contains multiple LTO then there should be another level of iteration as well
                skip_calculation, lto_list[pair_key_mapping[ao_pair][0]] = await self._handle_lto(lto_list[pair_key_mapping[ao_pair][0]], dt_index)
                if skip_calculation: continue;

            else: pass # Make a brand new decision
            
            time_dict = analysis_dict[ao_pair]
            # Since all parameters are handled in a different way, 
            # there needs to be different handlers for each type of indicator
            # TODO: Create a list of indicator handlers: [atr_handler()]

            trange_mean5 = st.mean(time_dict['15m']['trange'][-5:])
            trange_mean20 = st.mean(time_dict['15m']['trange'][-20:])

            # Make decision to enter or not
            if trange_mean5 < trange_mean20:
                self.logger.info(f"{ao_pair}: BUY SIGNAL")
                trade_obj = copy.deepcopy(GenericObject.trade)
                trade_obj['status'] = STAT_OPEN_ENTER
                trade_obj['strategy'] = self.name
                trade_obj['pair'] = ao_pair
                trade_obj['history'].append(trade_obj['status'])
                trade_obj['decision_time'] = int(dt_index) # Set decision_time to timestamp which is the open time of the current kline
                # TODO: give proper values to limit

                # Calculate enter/exit prices
                enter_price = min(time_dict['15m']['low'][-10:]) * 0.99
                exit_price = max(time_dict['15m']['high'][-10:])
                # Calculate enter/exit amount value

                #TODO: Amount calculation is performed to decide how much of the 'free' amount of 
                # the base asset will be used.
                
                free_ref_asset = df_balance.loc[self.quote_currency,'free']

                # Example: Buy XRP with 100$ in your account
                enter_ref_amount=20

                # TODO: HIGH: In order to not to face with an issue with dust, exit amount might be "just a bit less" then what it should be
                # Example:
                #   Buy XRP from the price XRPUSDT: 0.66 (Price of 1XRP = 0.66$), use 100$ to make the trade
                #   151,51 = 100$ / 0.66
                enter_quantity = enter_ref_amount / enter_price

                #   Sell the bought XRP from the price 0.70
                #   exit_ref_amount = 151,4 * 0.70 = 105.98
                exit_ref_amount = enter_quantity * exit_price

                # Fill enter and exit modules
                enter_type = self.config['enter']['type']
                exit_type = self.config['exit']['type']

                trade_obj['enter'] = await StrategyBase._create_enter_module(enter_type, enter_price, enter_quantity, enter_ref_amount, 
                                                                        StrategyBase._eval_future_candle_time(dt_index,2,self.scales_in_minute[0])) # NOTE: Multiple scale is not supported
                trade_obj['exit'] = await StrategyBase._create_exit_module(exit_type, enter_price, enter_quantity, exit_price, exit_ref_amount, 
                                                                        StrategyBase._eval_future_candle_time(dt_index,9,self.scales_in_minute[0])) # NOTE: Multiple scale is not supported

                # TODO: Check the free amount of quote currency
                free_ref_asset = df_balance.loc[self.quote_currency,'free']

                trade_obj['enter'][self.config['enter']['type']] = await StrategyBase.apply_exchange_filters('enter', 
                                                                                                            enter_type, 
                                                                                                            trade_obj['enter'][enter_type], 
                                                                                                            self.symbol_info)

                if not await StrategyBase.check_min_notional(trade_obj['enter'][enter_type]['price'], trade_obj['enter'][enter_type]['quantity'], self.symbol_info):
                    # TODO: Notification about min_notional
                    continue
                trade_objects.append(trade_obj)

            else:
                self.logger.info(f"{ao_pair}: NO SIGNAL")

        return trade_objects


class OCOBackTest(StrategyBase):

    def __init__(self, _config, _symbol_info={}):
        self.name = "OCOBackTest"
        self.logger = logging.getLogger('{}.{}'.format(__name__,self.name))

        self.config = _config['strategy'][self.name]

        self.quote_currency = _config['broker']['quote_currency']
        self.scales_in_minute = _config['data_input']['scales_in_minute']
        # TODO: Make proper handling for symbol_info
        self.symbol_info = _symbol_info

        return


    async def _handle_lto(self, lto, dt_index):
        """
        This function decides what to do for the LTOs based on their 'status'
        """        
        skip_calculation = False
        
        if lto['status'] == STAT_ENTER_EXP:
            if self.config['action_mapping'][STAT_ENTER_EXP] == ACTN_CANCEL or lto['history'].count(STAT_ENTER_EXP) > 1:
                lto['action'] = ACTN_CANCEL
                lto['result']['cause'] = STAT_ENTER_EXP

            elif self.config['action_mapping'][STAT_ENTER_EXP] == ACTN_POSTPONE and lto['history'].count(STAT_ENTER_EXP) <= 1:
                # NOTE: postponed_candles = 1 means 2 candle
                #       If only 1 candle is desired to be postponed, then it means we will wait for newly started candle to close so postponed_candles will be 0
                postponed_candles = 1
                lto = await StrategyBase._postpone(lto,'enter', self.config['enter']['type'], StrategyBase._eval_future_candle_time(dt_index,postponed_candles,self.scales_in_minute[0])) 
                skip_calculation = True
            else: pass

        elif lto['status'] == STAT_EXIT_EXP:
            if self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_MARKET_EXIT or lto['history'].count(STAT_EXIT_EXP) > 1:
                lto = await StrategyBase._config_market_exit(lto, self.config['exit']['type'])
                self.logger.info(f'LTO: market exit configured') # TODO: Add orderId

            elif self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_POSTPONE and lto['history'].count(STAT_EXIT_EXP) <= 1:
                postponed_candles = 1
                lto = await StrategyBase._postpone(lto,'exit', self.config['exit']['type'], StrategyBase._eval_future_candle_time(dt_index,postponed_candles,self.scales_in_minute[0]))
                skip_calculation = True

            elif self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_UPDATE:
                # TODO: Give a call to methods that calculates exit point
                # NOTE: Things to change: price, limitPrice, stopLimitPrice, expire date
                lto['action'] = ACTN_UPDATE 
                lto['update_history'].append(copy.deepcopy(lto['exit'][self.config['exit']['type']])) # This is good for debugging and will be good for perf. evaluation in future
                if self.config['exit']['type'] == TYPE_LIMIT:
                    lto['exit'][TYPE_LIMIT]['price'] *= 0.99
                    lto['exit'][TYPE_LIMIT]['amount'] = lto['exit'][TYPE_LIMIT]['price'] * lto['exit'][TYPE_LIMIT]['quantity']
                    lto['exit'][TYPE_LIMIT]['expire'] = StrategyBase._eval_future_candle_time(dt_index,3,self.scales_in_minute[0])

                elif self.config['exit']['type'] == TYPE_OCO:
                    lto['exit'][TYPE_OCO]['limitPrice'] *= 0.99
                    lto['exit'][TYPE_OCO]['amount'] = lto['exit'][TYPE_OCO]['limitPrice'] * lto['exit'][TYPE_OCO]['quantity']
                    lto['exit'][TYPE_OCO]['expire'] = StrategyBase._eval_future_candle_time(dt_index,3,self.scales_in_minute[0])
                skip_calculation = True

                # Apply the filters
                # TODO: Add min notional fix (No need to add the check because we are not gonna do anything with that)
                lto['exit'][self.config['exit']['type']] = await StrategyBase.apply_exchange_filters('exit', 
                                                                                                    self.config['exit']['type'], 
                                                                                                    lto['exit'][self.config['exit']['type']], 
                                                                                                    self.symbol_info, 
                                                                                                    exit_qty=lto['result']['enter']['quantity'])

            else: pass

        elif lto['status'] == STAT_WAITING_EXIT:
            # LTO is entered succesfully, so exit order should be executed
            # TODO: expire of the exit_module can be calculated after the trade entered

            lto['action'] = ACTN_EXEC_EXIT
            lto['exit'][self.config['exit']['type']] = await StrategyBase.apply_exchange_filters('exit', 
                                                                                                self.config['exit']['type'], 
                                                                                                lto['exit'][self.config['exit']['type']], 
                                                                                                self.symbol_info, 
                                                                                                exit_qty=lto['result']['enter']['quantity'])
            skip_calculation = True

        elif lto['status'] != STAT_CLOSED:
            # If the status is not closed, just skip the iteration. otherwise go on to make a decision
            # NOTE: This logic contains the status: STAT_OPEN_EXIT, STAT_OPEN_ENTER, STAT_PART_CLOSED_ENTER, STAT_PART_CLOSED_EXIT
            skip_calculation = True

        return skip_calculation, lto


    async def run(self, analysis_dict, lto_list, df_balance, dt_index=None):
        """
        It requires to feed analysis_dict and lto_dict so that it may decide to:
        - not to enter a new trade if there is already an open trade
        - cancel the trade if an drawdown is detected

        Args:
            analysis_dict (dict): analysis.json
            - analysis objects contains where to buy and and where to sell

            lto_dict (dict): live-trade-objects coming from the [live-trades]

            df_balance (pd.DataFrame): live-trade-objects coming from the [live-trades]

            dt_index (int): timestamp in ms for trade_object identifier
            
        Returns:
            list: nto_list
        """
        # Preliminary condition: all of the config['pairs'] exist in analysis_dict
        if not set(self.config['pairs']).issubset(analysis_dict.keys()):
            self.logger.warn(f"Configured pair \"{self.config['pairs']}\" does not exist in analysis_dict. Skipping {self.name}.run")
            return []

        # Initialize trade_dict to be filled
        trade_objects = []

        # Create a mapping between the pair and orderId such as {'BTCUSDT':['123','456']}
        pair_key_mapping = {}
        for i, lto in enumerate(lto_list):
            pair = lto['pair']
            if pair not in pair_key_mapping.keys():
                pair_key_mapping[pair] = []

            pair_key_mapping[pair].append(i)

        # This implementation enable to check number of trades and compare the value with the one in the config file.

        # NOTE: Only iterate for the configured pairs. Do not run the strategy if any of them is missing in analysis_dict
        for ao_pair in self.config['pairs']:

            # Check if there is already an LTO that has that 'pair' item. If so skip the evaluation (one pair one LTO rule)
            if ao_pair in pair_key_mapping.keys():
                
                # NOTE: If a pair contains multiple LTO then there should be another level of iteration as well
                skip_calculation, lto_list[pair_key_mapping[ao_pair][0]] = await self._handle_lto(lto_list[pair_key_mapping[ao_pair][0]], dt_index)
                if skip_calculation: continue;

            else: pass # Make a brand new decision
            
            time_dict = analysis_dict[ao_pair]
            # Since all parameters are handled in a different way, 
            # there needs to be different handlers for each type of indicator
            # TODO: Create a list of indicator handlers: [atr_handler()]

            trange_mean5 = st.mean(time_dict['15m']['trange'][-5:])
            trange_mean20 = st.mean(time_dict['15m']['trange'][-20:])

            # Make decision to enter or not
            if trange_mean5 < trange_mean20:
                self.logger.info(f"{ao_pair}: BUY SIGNAL")
                trade_obj = copy.deepcopy(GenericObject.trade)
                trade_obj['status'] = STAT_OPEN_ENTER
                trade_obj['strategy'] = self.name
                trade_obj['pair'] = ao_pair
                trade_obj['history'].append(trade_obj['status'])
                trade_obj['decision_time'] = int(dt_index) # Set decision_time to timestamp which is the open time of the current kline
                # TODO: give proper values to limit

                # Calculate enter/exit prices
                enter_price = min(time_dict['15m']['low'][-10:])
                exit_price = max(time_dict['15m']['high'][-10:])

                # Calculate enter/exit amount value

                #TODO: Amount calculation is performed to decide how much of the 'free' amount of 
                # the base asset will be used.

                free_ref_asset = df_balance.loc[self.quote_currency,'free']

                # Example: Buy XRP with 100$ in your account
                enter_ref_amount=20
                # TODO: HIGH: Check mininum amount to trade and add this section to here
                if free_ref_asset > 10:
                    if free_ref_asset < enter_ref_amount:
                        enter_ref_amount = free_ref_asset
                else:
                    # TODO: Add error logs and send notification
                    return {}

                # TODO: HIGH: In order to not to face with an issue with dust, exit amount might be "just a bit less" then what it should be
                # Example:
                #   Buy XRP from the price XRPUSDT: 0.66 (Price of 1XRP = 0.66$), use 100$ to make the trade
                #   151,51 = 100$ / 0.66
                enter_quantity = enter_ref_amount / enter_price

                #   Sell the bought XRP from the price 0.70
                #   exit_ref_amount = 151,4 * 0.70 = 105.98
                exit_ref_amount = enter_quantity * exit_price

                # Fill enter and exit modules
                enter_type = self.config['enter']['type']
                exit_type = self.config['exit']['type']

                trade_obj['enter'] = await StrategyBase._create_enter_module(enter_type, enter_price, enter_quantity, enter_ref_amount, 
                                                                        StrategyBase._eval_future_candle_time(dt_index,2,self.scales_in_minute[0])) # NOTE: Multiple scale is not supported
                trade_obj['exit'] = await StrategyBase._create_exit_module(exit_type, enter_price, enter_quantity, exit_price, exit_ref_amount, 
                                                                        StrategyBase._eval_future_candle_time(dt_index,9,self.scales_in_minute[0])) # NOTE: Multiple scale is not supported

                # TODO: Check the free amount of quote currency
                free_ref_asset = df_balance.loc[self.quote_currency,'free']

                trade_obj['enter'][self.config['enter']['type']] = await StrategyBase.apply_exchange_filters('enter', 
                                                                                                            enter_type, 
                                                                                                            trade_obj['enter'][enter_type], 
                                                                                                            self.symbol_info)

                if not await StrategyBase.check_min_notional(trade_obj['enter'][enter_type]['price'], trade_obj['enter'][enter_type]['quantity'], self.symbol_info):
                    # TODO: Notification about min_notional
                    continue
                trade_objects.append(trade_obj)

            else:
                self.logger.info(f"{ao_pair}: NO SIGNAL")

        return trade_objects


class AlwaysEnter(StrategyBase):

    def __init__(self, _config, _symbol_info={}):
        self.name = "AlwaysEnter"
        self.logger = logging.getLogger('{}.{}'.format(__name__,self.name))

        # TODO: Find a more beautiful way to implemetn this logic
        self.config = _config['strategy'][self.name]

        self.quote_currency = _config['broker']['quote_currency']
        self.scales_in_minute = _config['data_input']['scales_in_minute']

        # TODO: Make proper handling for symbol_info
        self.symbol_info = _symbol_info
        return


    async def _handle_lto(self, lto, dt_index):
        skip_calculation = False
        
        if lto['status'] == STAT_ENTER_EXP:
            if self.config['action_mapping'][STAT_ENTER_EXP] == ACTN_CANCEL or lto['history'].count(STAT_ENTER_EXP) > 1:
                lto['action'] = ACTN_CANCEL
                lto['result']['cause'] = STAT_ENTER_EXP

            elif self.config['action_mapping'][STAT_ENTER_EXP] == ACTN_POSTPONE and lto['history'].count(STAT_ENTER_EXP) <= 1:
                # NOTE: postponed_candles = 1 means 2 candle
                #       If only 1 candle is desired to be postponed, then it means we will wait for newly started candle to close so postponed_candles will be 0
                postponed_candles = 1
                lto = await StrategyBase._postpone(lto,'enter', self.config['enter']['type'], StrategyBase._eval_future_candle_time(dt_index,postponed_candles,self.scales_in_minute[0])) 
                skip_calculation = True
            else: pass

        elif lto['status'] == STAT_EXIT_EXP:
            if self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_MARKET_EXIT or lto['history'].count(STAT_EXIT_EXP) > 1:
                lto = await StrategyBase._config_market_exit(lto, self.config['exit']['type'])
                self.logger.info(f'LTO: market exit configured') # TODO: Add orderId

            elif self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_POSTPONE and lto['history'].count(STAT_EXIT_EXP) <= 1:
                postponed_candles = 1
                lto = await StrategyBase._postpone(lto,'exit', self.config['exit']['type'], StrategyBase._eval_future_candle_time(dt_index,postponed_candles,self.scales_in_minute[0]))
                skip_calculation = True

            elif self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_UPDATE:
                # TODO: Give a call to methods that calculates exit point
                # NOTE: Things to change: price, limitPrice, stopLimitPrice, expire date
                lto['action'] = ACTN_UPDATE 
                lto['update_history'].append(copy.deepcopy(lto['exit'][self.config['exit']['type']])) # This is good for debugging and will be good for perf. evaluation in future
                if self.config['exit']['type'] == TYPE_LIMIT:
                    lto['exit'][TYPE_LIMIT]['price'] *= 0.99
                    lto['exit'][TYPE_LIMIT]['amount'] = lto['exit'][TYPE_LIMIT]['price'] * lto['exit'][TYPE_LIMIT]['quantity']
                    lto['exit'][TYPE_LIMIT]['expire'] = StrategyBase._eval_future_candle_time(dt_index,1,self.scales_in_minute[0])

                elif self.config['exit']['type'] == TYPE_OCO:
                    lto['exit'][TYPE_OCO]['limitPrice'] *= 0.99
                    lto['exit'][TYPE_OCO]['amount'] = lto['exit'][TYPE_OCO]['limitPrice'] * lto['exit'][TYPE_OCO]['quantity']
                    lto['exit'][TYPE_OCO]['expire'] = StrategyBase._eval_future_candle_time(dt_index,1,self.scales_in_minute[0])
                skip_calculation = True

                # Apply the filters
                # TODO: Add min notional fix (No need to add the check because we are not gonna do anything with that)
                lto['exit'][self.config['exit']['type']] = await StrategyBase.apply_exchange_filters('exit', 
                                                                                                    self.config['exit']['type'], 
                                                                                                    lto['exit'][self.config['exit']['type']], 
                                                                                                    self.symbol_info, 
                                                                                                    exit_qty=lto['result']['enter']['quantity'])

            else: pass

        elif lto['status'] == STAT_WAITING_EXIT:
            # LTO is entered succesfully, so exit order should be executed
            # TODO: expire of the exit_module can be calculated after the trade entered

            lto['action'] = ACTN_EXEC_EXIT
            lto['exit'][self.config['exit']['type']] = await StrategyBase.apply_exchange_filters('exit', 
                                                                                                self.config['exit']['type'], 
                                                                                                lto['exit'][self.config['exit']['type']], 
                                                                                                self.symbol_info, 
                                                                                                exit_qty=lto['result']['enter']['quantity'])
            skip_calculation = True

        elif lto['status'] != STAT_CLOSED:
            # If the status is not closed, just skip the iteration. otherwise go on to make a decision
            # NOTE: This logic contains the status: STAT_OPEN_EXIT, STAT_OPEN_ENTER, STAT_PART_CLOSED_ENTER, STAT_PART_CLOSED_EXIT
            skip_calculation = True

        return skip_calculation, lto


    async def run(self, analysis_dict, lto_list, df_balance, dt_index=None):
        """
        It requires to feed analysis_dict and lto_dict so that it may decide to:
        - not to enter a new trade if there is already an open trade
        - cancel the trade if an drawdown is detected

        Args:
            analysis_dict (dict): analysis.json
            - analysis objects contains where to buy and and where to sell

            lto_dict (dict): live-trade-objects coming from the [live-trades]

            df_balance (pd.DataFrame): live-trade-objects coming from the [live-trades]

            dt_index (int): timestamp in ms for trade_object identifier
            
        Returns:
            list: nto_list
        """
        # Preliminary condition: all of the config['pairs'] exist in analysis_dict
        if not set(self.config['pairs']).issubset(analysis_dict.keys()):
            self.logger.warn(f"Configured pair \"{self.config['pairs']}\" does not exist in analysis_dict. Skipping {self.name}.run")
            return []

        # Initialize trade_dict to be filled
        trade_objects = []

        # Create a mapping between the pair and orderId such as {'BTCUSDT':['123','456']}
        pair_key_mapping = {}
        for i, lto in enumerate(lto_list):
            pair = lto['pair']
            if pair not in pair_key_mapping.keys():
                pair_key_mapping[pair] = []

            pair_key_mapping[pair].append(i)

        # This implementation enable to check number of trades and compare the value with the one in the config file.

        # NOTE: Only iterate for the configured pairs. Do not run the strategy if any of them is missing in analysis_dict
        for ao_pair in self.config['pairs']:

            # Check if there is already an LTO that has that 'pair' item. If so skip the evaluation (one pair one LTO rule)
            if ao_pair in pair_key_mapping.keys():
                
                # NOTE: If a pair contains multiple LTO then there should be another level of iteration as well
                skip_calculation, lto_list[pair_key_mapping[ao_pair][0]] = await self._handle_lto(lto_list[pair_key_mapping[ao_pair][0]], dt_index)
                if skip_calculation: continue;

            else: pass # Make a brand new decision
            
            assert len(analysis_dict[ao_pair].keys()) == 1, "Multiple time scale is not supported"
            scale = list(analysis_dict[ao_pair].keys())[0]

            # Make decision to enter or not
            if True:
                self.logger.info(f"{ao_pair}: BUY SIGNAL")
                trade_obj = copy.deepcopy(GenericObject.trade)
                trade_obj['status'] = STAT_OPEN_ENTER
                trade_obj['strategy'] = self.name
                trade_obj['pair'] = ao_pair
                trade_obj['history'].append(trade_obj['status'])
                trade_obj['decision_time'] = int(dt_index) # Set decision_time to the the open time of the current kline not the last closed kline

                # Calculate enter/exit prices
                enter_price = float(analysis_dict[ao_pair][scale]['low'][-1])/2 # NOTE: Give half of the price to make sure it will enter
                exit_price = float(analysis_dict[ao_pair][scale]['high'][-1])*2 # NOTE: Give double of the price to make sure it will not exit
                
                # Example: Buy XRP with 100$ in your account
                enter_ref_amount=20

                #   Buy XRP from the price XRPUSDT: 0.66 (Price of 1XRP = 0.66$), use 100$ to make the trade
                #   151,51 = 100$ / 0.66
                enter_quantity = enter_ref_amount / enter_price

                #   Sell the bought XRP from the price 0.70
                #   exit_ref_amount = 151,4 * 0.70 = 105.98
                exit_ref_amount = enter_quantity * exit_price

                # Fill enter and exit modules
                enter_type = self.config['enter']['type']
                exit_type = self.config['exit']['type']

                trade_obj['enter'] = await StrategyBase._create_enter_module(enter_type, enter_price, enter_quantity, enter_ref_amount, 
                                                                        StrategyBase._eval_future_candle_time(dt_index,0,self.scales_in_minute[0])) # NOTE: Multiple scale is not supported
                trade_obj['exit'] = await StrategyBase._create_exit_module(exit_type, enter_price, enter_quantity, exit_price, exit_ref_amount, 
                                                                        StrategyBase._eval_future_candle_time(dt_index,0,self.scales_in_minute[0])) # NOTE: Multiple scale is not supported

                # TODO: Check the free amount of quote currency
                free_ref_asset = df_balance.loc[self.quote_currency,'free']

                trade_obj['enter'][self.config['enter']['type']] = await StrategyBase.apply_exchange_filters('enter', 
                                                                                                            enter_type, 
                                                                                                            trade_obj['enter'][enter_type], 
                                                                                                            self.symbol_info)

                if not await StrategyBase.check_min_notional(trade_obj['enter'][enter_type]['price'], trade_obj['enter'][enter_type]['quantity'], self.symbol_info):
                    # TODO: Notification about min_notional
                    continue

                # Normally trade id should be unique and be given by the broker. Until it is executed assign the current ts. It will be updated at execution anyway
                trade_objects.append(trade_obj)

            else:
                self.logger.info(f"{ao_pair}: NO SIGNAL")

        return trade_objects


class AlwaysEnter90(StrategyBase):

    def __init__(self, _config, _symbol_info={}):
        self.name = "AlwaysEnter90"
        self.logger = logging.getLogger('{}.{}'.format(__name__,self.name))

        self.config = _config['strategy'][self.name]
        self.quote_currency = _config['broker']['quote_currency']
        self.scales_in_minute = _config['data_input']['scales_in_minute']

        # TODO: Make proper handling for symbol_info
        self.symbol_info = _symbol_info
        return


    async def _handle_lto(self, lto, dt_index):
        skip_calculation = False
        
        if lto['status'] == STAT_ENTER_EXP:
            if self.config['action_mapping'][STAT_ENTER_EXP] == ACTN_CANCEL or lto['history'].count(STAT_ENTER_EXP) > 1:
                lto['action'] = ACTN_CANCEL
                lto['result']['cause'] = STAT_ENTER_EXP

            elif self.config['action_mapping'][STAT_ENTER_EXP] == ACTN_POSTPONE and lto['history'].count(STAT_ENTER_EXP) <= 1:
                # NOTE: postponed_candles = 1 means 2 candle
                #       If only 1 candle is desired to be postponed, then it means we will wait for newly started candle to close so postponed_candles will be 0
                postponed_candles = 1
                lto = await StrategyBase._postpone(lto,'enter', self.config['enter']['type'], StrategyBase._eval_future_candle_time(dt_index,postponed_candles,self.scales_in_minute[0])) 
                skip_calculation = True
            else: pass

        elif lto['status'] == STAT_EXIT_EXP:
            if self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_MARKET_EXIT or lto['history'].count(STAT_EXIT_EXP) > 1:
                lto = await StrategyBase._config_market_exit(lto, self.config['exit']['type'])
                self.logger.info(f'LTO: market exit configured') # TODO: Add orderId

            elif self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_POSTPONE and lto['history'].count(STAT_EXIT_EXP) <= 1:
                postponed_candles = 1
                lto = await StrategyBase._postpone(lto,'exit', self.config['exit']['type'], StrategyBase._eval_future_candle_time(dt_index,postponed_candles,self.scales_in_minute[0]))
                skip_calculation = True

            elif self.config['action_mapping'][STAT_EXIT_EXP] == ACTN_UPDATE:
                # TODO: Give a call to methods that calculates exit point
                # NOTE: Things to change: price, limitPrice, stopLimitPrice, expire date
                lto['action'] = ACTN_UPDATE 
                lto['update_history'].append(copy.deepcopy(lto['exit'][self.config['exit']['type']])) # This is good for debugging and will be good for perf. evaluation in future
                if self.config['exit']['type'] == TYPE_LIMIT:
                    lto['exit'][TYPE_LIMIT]['price'] *= 0.99
                    lto['exit'][TYPE_LIMIT]['amount'] = lto['exit'][TYPE_LIMIT]['price'] * lto['exit'][TYPE_LIMIT]['quantity']
                    lto['exit'][TYPE_LIMIT]['expire'] = StrategyBase._eval_future_candle_time(dt_index,1,self.scales_in_minute[0])

                elif self.config['exit']['type'] == TYPE_OCO:
                    lto['exit'][TYPE_OCO]['limitPrice'] *= 0.99
                    lto['exit'][TYPE_OCO]['amount'] = lto['exit'][TYPE_OCO]['limitPrice'] * lto['exit'][TYPE_OCO]['quantity']
                    lto['exit'][TYPE_OCO]['expire'] = StrategyBase._eval_future_candle_time(dt_index,1,self.scales_in_minute[0])
                skip_calculation = True

                # Apply the filters
                # TODO: Add min notional fix (No need to add the check because we are not gonna do anything with that)
                lto['exit'][self.config['exit']['type']] = await StrategyBase.apply_exchange_filters('exit', 
                                                                                                    self.config['exit']['type'], 
                                                                                                    lto['exit'][self.config['exit']['type']], 
                                                                                                    self.symbol_info, 
                                                                                                    exit_qty=lto['result']['enter']['quantity'])
            else: pass

        elif lto['status'] == STAT_WAITING_EXIT:
            # LTO is entered succesfully, so exit order should be executed
            # TODO: expire of the exit_module can be calculated after the trade entered

            lto['action'] = ACTN_EXEC_EXIT
            lto['exit'][self.config['exit']['type']] = await StrategyBase.apply_exchange_filters('exit', 
                                                                                                self.config['exit']['type'], 
                                                                                                lto['exit'][self.config['exit']['type']], 
                                                                                                self.symbol_info, 
                                                                                                exit_qty=lto['result']['enter']['quantity'])
            skip_calculation = True

        elif lto['status'] != STAT_CLOSED:
            # If the status is not closed, just skip the iteration. otherwise go on to make a decision
            # NOTE: This logic contains the status: STAT_OPEN_EXIT, STAT_OPEN_ENTER, STAT_PART_CLOSED_ENTER, STAT_PART_CLOSED_EXIT
            skip_calculation = True

        return skip_calculation, lto


    async def run(self, analysis_dict, lto_list, df_balance, dt_index=None):
        """
        It requires to feed analysis_dict and lto_dict so that it may decide to:
        - not to enter a new trade if there is already an open trade
        - cancel the trade if an drawdown is detected

        Args:
            analysis_dict (dict): analysis.json
            - analysis objects contains where to buy and and where to sell

            lto_dict (dict): live-trade-objects coming from the [live-trades]

            df_balance (pd.DataFrame): live-trade-objects coming from the [live-trades]

            dt_index (int): timestamp in ms for trade_object identifier
            
        Returns:
            list: nto_list
        """
        # Preliminary condition: all of the config['pairs'] exist in analysis_dict
        if not set(self.config['pairs']).issubset(analysis_dict.keys()):
            self.logger.warn(f"Configured pair \"{self.config['pairs']}\" does not exist in analysis_dict. Skipping {self.name}.run")
            return []

        # Initialize trade_dict to be filled
        trade_objects = []

        # Create a mapping between the pair and orderId such as {'BTCUSDT':['123','456']}
        pair_key_mapping = {}
        for i, lto in enumerate(lto_list):
            pair = lto['pair']
            if pair not in pair_key_mapping.keys():
                pair_key_mapping[pair] = []

            pair_key_mapping[pair].append(i)

        # This implementation enable to check number of trades and compare the value with the one in the config file.

        # NOTE: Only iterate for the configured pairs. Do not run the strategy if any of them is missing in analysis_dict
        for ao_pair in self.config['pairs']:

            # Check if there is already an LTO that has that 'pair' item. If so skip the evaluation (one pair one LTO rule)
            if ao_pair in pair_key_mapping.keys():
                
                # NOTE: If a pair contains multiple LTO then there should be another level of iteration as well
                skip_calculation, lto_list[pair_key_mapping[ao_pair][0]] = await self._handle_lto(lto_list[pair_key_mapping[ao_pair][0]], dt_index)
                if skip_calculation: continue;

            else: pass # Make a brand new decision
            
            assert len(analysis_dict[ao_pair].keys()) == 1, "Multiple time scale is not supported"
            scale = list(analysis_dict[ao_pair].keys())[0]

            # Make decision to enter or not
            if True:
                self.logger.info(f"{ao_pair}: BUY SIGNAL")
                trade_obj = copy.deepcopy(GenericObject.trade)
                trade_obj['status'] = STAT_OPEN_ENTER
                trade_obj['strategy'] = self.name
                trade_obj['pair'] = ao_pair
                trade_obj['history'].append(trade_obj['status'])
                trade_obj['decision_time'] = int(dt_index) # Set decision_time to the the open time of the current kline not the last closed kline

                # Calculate enter/exit prices
                enter_price = float(analysis_dict[ao_pair][scale]['low'][-1])*0.90 # NOTE: Give half of the price to make sure it will enter
                exit_price = float(analysis_dict[ao_pair][scale]['high'][-1])*2
                
                # Example: Buy XRP with 100$ in your account
                enter_ref_amount=20
                # TODO: NEXT: Resource allocation

                #   Buy XRP from the price XRPUSDT: 0.66 (Price of 1XRP = 0.66$), use 100$ to make the trade
                #   151,51 = 100$ / 0.66
                enter_quantity = enter_ref_amount / enter_price

                #   Sell the bought XRP from the price 0.70
                #   exit_ref_amount = 151,4 * 0.70 = 105.98
                exit_ref_amount = enter_quantity * exit_price

                enter_type = self.config['enter']['type']
                exit_type = self.config['exit']['type']

                # Fill enter and exit modules
                # TODO: Expire calculation should be based on the 'scale'. It should not be hardcoded '15'
                trade_obj['enter'] = await StrategyBase._create_enter_module(enter_type, enter_price, enter_quantity, enter_ref_amount, 
                                                                        StrategyBase._eval_future_candle_time(dt_index,0,self.scales_in_minute[0])) # NOTE: Multiple scale is not supported
                trade_obj['exit'] = await StrategyBase._create_exit_module(exit_type, enter_price, enter_quantity, exit_price, exit_ref_amount, 
                                                                        StrategyBase._eval_future_candle_time(dt_index,0,self.scales_in_minute[0])) # NOTE: Multiple scale is not supported
                # TODO: Check the free amount of quote currency
                free_ref_asset = df_balance.loc[self.quote_currency,'free']

                trade_obj['enter'][enter_type] = await StrategyBase.apply_exchange_filters('enter', 
                                                                                            enter_type, 
                                                                                            trade_obj['enter'][enter_type], 
                                                                                            self.symbol_info)

                if not await StrategyBase.check_min_notional(trade_obj['enter'][enter_type]['price'], trade_obj['enter'][enter_type]['quantity'], self.symbol_info):
                    # TODO: Notification about min_notional
                    continue

                # Normally trade id should be unique and be given by the broker. Until it is executed assign the current ts. It will be updated at execution anyway
                trade_objects.append(trade_obj)

            else:
                self.logger.info(f"{ao_pair}: NO SIGNAL")

        return trade_objects