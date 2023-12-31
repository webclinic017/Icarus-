from exceptions import NotImplementedException
from binance.enums import *
import asyncio
import pandas as pd
import logging
import json
import bson
import sys
from utils import time_scale_to_second, get_min_scale, \
    safe_multiply, safe_divide, round_to_period, safe_sum
from objects import Trade, OCO, ECause, ECommand, EState, Limit, Market, TradeResult, Result, trade_to_dict
from utils import setup_logger
from dataclasses import asdict, dataclass
from binance import AsyncClient
from connectivity.telegram_wrapper import TelegramBot
import itertools
from typing import Dict, List, Any


logger = logging.getLogger('app')

# This variable added as deus ex machina
symbol_info = None

@dataclass
class OrderInfo():
    order: dict
    trade_list: list


class BinanceWrapper():

    kline_column_names = ["open_time", "open", "high", "low", "close", "volume", "close_time","quote_asset_volume", 
                        "num_of_trades", "taker_buy_base_ast_vol", "taker_buy_quote_ast_vol", "ignore"]

    def __init__(self, _client: AsyncClient, _config: dict):
        # TODO: Think about the binance.exceptions.BinanceAPIException: APIError(code=-1021): Timestamp for this request was 1000ms ahead of the server's time.
        #       The alternative slution (the wrapper for the binane client can be added to here):
        #       https://github.com/sammchardy/python-binance/issues/249

        self.client = _client
        self.config = _config

        # Set reference currencies
        self.quote_currency = _config['broker']['quote_currency']
        self.pricePrecision = 8
        # TODO: get the precision from symbol info and use it in the orders 


    async def close_connection(self):
        return await self.client.close_connection()


    async def get_status(self):
        broker_status = await self.client.get_system_status()
        return broker_status['status']


    async def get_time(self):
        '''
        { 
            "status": 0,              // 0: normal, 1: system maintenance
            "msg": "normal"           // "normal", "system_maintenance"
        }
        '''
        server_time = await self.client.get_server_time()
        return int(server_time['serverTime']/1000)


    async def get_info(self) -> pd.DataFrame:

        info = await self.client.get_account()
        balance = [{'asset':b['asset'], 'free':b['free'], 'locked':b['locked']}
                   for b in info['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]

        df_balance = pd.DataFrame(balance)
        df_balance.set_index(['asset'], inplace=True)
        df_balance = df_balance.astype(float)
        df_balance['total'] = df_balance['free'] + df_balance['locked']

        return df_balance


    async def get_all_tickers(self) -> pd.DataFrame:   
        df = pd.DataFrame(await self.client.get_all_tickers())
        df.set_index('symbol', inplace=True)
        df.astype(float)
        return df


    async def get_all_symbol_info(self, all_pairs) -> Dict[str, str]:
        all_info = await self.client.get_exchange_info()

        selected_info = {}
        for item in all_info['symbols']:
            if item['symbol'] in all_pairs:
                selected_info[item['symbol']] = item

        return selected_info


    async def get_current_balance(self, *args) -> pd.DataFrame:

        df_balance = await self.get_info()
        return df_balance


    async def get_data_dict(self, meta_data_pool, icarus_time_ms: int) -> Dict[str, Dict[str, Any]]:
        """
        meta_do = [('1m', 'BTCUSDT'), ('15m', 'BTCUSDT'), ('15m', 'XRPUSDT')]
        (time_scale, pair)
        length = meta_do['time_scale']
        """
        tasks_klines_scales = []
        for meta_data in meta_data_pool:

            if type(icarus_time_ms) == int:
                # NOTE: If you need exactly 720 candles not 719 (unclosed (last) candle removed) then push hist_data_start_time back 1 step
                # NOTE: The cause of +1 comes from the desire to not the make hist_data_start_time an exact minute, Because when it is an exact 1m or 15m, 1 extra hist_kline is retrived addi
                hist_data_start_time = icarus_time_ms - time_scale_to_second(meta_data[0]) * (self.config['time_scales'][meta_data[0]]) * 1000 + 1 # ms = start_time + x sec * y times * 1000 + 1
            else:
                raise NotImplementedException('start_time is not integer')

            tasks_klines_scales.append(asyncio.create_task(self.client.get_historical_klines(meta_data[1], meta_data[0], start_str=hist_data_start_time, end_str=icarus_time_ms )))

        composit_klines = list(await asyncio.gather(*tasks_klines_scales, return_exceptions=True))

        for result in composit_klines:
            if type(result) != list:
                logger.error(f'Invalid kline receiived: {result}')
                raise result

        # TODO: NEXT: BUG: If a pair does not exist in the given time, then composit kline comes empty: [[], [[...], [...]]
        data_dict = await self.decompose(meta_data_pool, composit_klines)

        # TODO: Length check for Data dict items could help a lot to detect anomalies
        return data_dict


    async def decompose(self, meta_data_pool, composit_klines):

        do_dict = dict()
        for idx, meta_data in enumerate(meta_data_pool):
            
            if not meta_data[1] in do_dict.keys():
                do_dict[meta_data[1]] = dict()
            
            df = pd.DataFrame(composit_klines[idx], columns=BinanceWrapper.kline_column_names)
            df = df.set_index(['open_time'])
            # NOTE: WARNING: Be aware that the last line is removed to not to affect analysis, since it requires closed candles.
            df.drop(df.index[-1], inplace=True)
            df = df.astype(float)

            if len(df) != int(self.config['time_scales'][meta_data[0]]-1):
                logger.warn(f"Missing historical candle for {meta_data[1]} {meta_data[0]}: {len(df)} instead of {self.config['time_scales'][meta_data[0]][1]-1}")

            do_dict[meta_data[1]][meta_data[0]] = df

        return do_dict


    async def get_order_info(self, trades: List[Trade]) -> Dict[str, OrderInfo]:
        
        if len(trades) == 0:
            return {}

        # Check the status of LTOs:
        order_coroutines = []
        trade_coroutines = []
        for trade in trades:
            if trade.status in [EState.OPEN_ENTER, EState.WAITING_EXIT]:
                order_coroutines.append(self.client.get_order(symbol=trade.pair, orderId=trade.enter.orderId))
                trade_coroutines.append(self.client.get_my_trades(symbol=trade.pair, orderId=trade.enter.orderId))

            elif trade.status in [EState.OPEN_EXIT, EState.CLOSED]:
                order_coroutines.append(self.client.get_order(symbol=trade.pair, orderId=trade.exit.orderId))
                trade_coroutines.append(self.client.get_my_trades(symbol=trade.pair, orderId=trade.exit.orderId))
                    
                if type(trade.exit) == OCO:
                    order_coroutines.append(self.client.get_order(symbol=trade.pair, orderId=trade.exit.stop_limit_orderId))
                    trade_coroutines.append(self.client.get_my_trades(symbol=trade.pair, orderId=trade.exit.stop_limit_orderId))

            else: pass

        if len(order_coroutines) == 0:
            return {}
        
        order_results = list(await asyncio.gather(*order_coroutines))
        trade_results = list(await asyncio.gather(*trade_coroutines))

        order_info = {}
        for order_result, trade_res in zip(order_results,trade_results):
            order_info[order_result['orderId']] = OrderInfo(order_result, trade_res)

        return order_info


    async def cancel_all_open_orders(self):

        # Get current balance
        df = await self.get_current_balance()

        # Obtain the symbols that has locked asset
        assets = list(df[df['locked'] != 0].index)
        symbols = list(map(lambda a: a+'USDT', assets))

        # Iterate over the symbols and get alive orders, extract orderIds
        coroutines = []
        for symbol in symbols:
            coroutines.append(self.client.get_open_orders(symbol=symbol))

        orders = await asyncio.gather(*coroutines)
        orders = list(itertools.chain(*orders))

        # Cancel orders
        for order in orders:
            try:
                response = await self.client.cancel_order(
                    symbol=order['symbol'],
                    orderId=order['orderId'])
                logger.info(f'Order canceled: {response}')
            except Exception as e:
                logger.error(f'Order could not be canceled: {e}\n{order}')


    async def _execute_oco_sell(self, trade: Trade) -> bool:
        try:
            logger.debug(f'trade.exit: {asdict(trade.exit)}')
            response = await self.client.order_oco_sell(
                symbol=trade.pair,
                quantity=trade.exit.quantity,
                price=f'%.{self.pricePrecision}f' % trade.exit.price,
                stopPrice=f'%.{self.pricePrecision}f' % trade.exit.stop_price,
                stopLimitPrice=f'%.{self.pricePrecision}f' % trade.exit.stop_limit_price,
                stopLimitTimeInForce=TIME_IN_FORCE_GTC)
            logger.debug(json.dumps(response, indent=4))

        except Exception as e:
            logger.error(f"{e}")
            return False

        else:
            response_stoploss, response_limit_maker = response["orderReports"][0], response["orderReports"][1]
            logger.info(f'{trade._id}: "{response_limit_maker["side"]}" "{response_limit_maker["type"]}" order placed: {response_limit_maker["orderId"]}')
            logger.info(f'{trade._id}: "{response_stoploss["side"]}" "{response_stoploss["type"]}" order placed: {response_stoploss["orderId"]}')

            trade.exit.orderId = response_limit_maker['orderId']
            trade.exit.stop_limit_orderId = response_stoploss['orderId']
            trade.status = EState.OPEN_EXIT

            TelegramBot.send_formatted_message('order_executed', asdict(trade.exit), ['SELL', trade.strategy, trade.pair], [trade._id])

        return True


    async def _execute_limit_sell(self, trade: Trade) -> bool:
        try:
            logger.debug(f'trade.exit: {asdict(trade.exit)}')
            logger.debug(f'trade.exit.price: %.{self.pricePrecision}f' % trade.exit.price)
            response = await self.client.order_limit_sell(
                symbol=trade.pair,
                quantity=trade.exit.quantity,
                price=f'%.{self.pricePrecision}f' % trade.exit.price)
            logger.debug(json.dumps(response, indent=4))

        except Exception as e:
            logger.error(f"{e}")
            return False

        else:
            logger.info(f'{trade._id}: "{response["side"]}" "{response["type"]}" order placed: {response["orderId"]}')
            trade.exit.orderId = response['orderId']
            trade.status = EState.OPEN_EXIT
            TelegramBot.send_formatted_message('order_executed', asdict(trade.exit), [response["side"], trade.strategy, trade.pair], [trade._id])

        return True


    async def _execute_limit_buy(self, trade: Trade) -> bool:
        try:
            logger.debug(f'trade.enter: {asdict(trade.enter)}')
            logger.debug(f'trade.enter.price: %.{self.pricePrecision}f' % trade.enter.price)
            response = await self.client.order_limit_buy(
                symbol=trade.pair,
                quantity=trade.enter.quantity,
                price=f'%.{self.pricePrecision}f' % trade.enter.price)
            logger.debug(json.dumps(response, indent=4))

        except Exception as e:
            logger.error(f"{e}")
            return False

        else:
            logger.info(f'{trade._id}: "{response["side"]}" "{response["type"]}" order placed: {response["orderId"]}')
            trade.enter.orderId = response['orderId']
            trade.status = EState.OPEN_ENTER
            TelegramBot.send_formatted_message('order_executed', asdict(trade.enter), [response["side"], trade.strategy, trade.pair], [trade._id])

        return True


    async def _execute_cancel(self, trade: Trade) -> bool:
        
        try:
            # Obtain which order to cancel
            if trade.status in [EState.OPEN_ENTER, EState.ENTER_EXP]:
                orderId_to_cancel = trade.enter.orderId
                logger.debug(f'trade.enter: {asdict(trade.enter)}')

            elif trade.status in [EState.OPEN_EXIT, EState.EXIT_EXP]:
                orderId_to_cancel = trade.order_stash[-1].orderId
                logger.debug(f'trade.exit: {asdict(trade.exit)}')
                if type(trade.exit) == OCO:
                    stop_limit_orderId = trade.order_stash[-1].stop_limit_orderId
                    logger.debug(f'Evaluated stop_limit_orderId that will be canceled: {stop_limit_orderId}')
                    pass
            else:
                raise Exception(f'Unexpected trade.status: {trade.status}')
            logger.debug(f'Evaluated orderId to cancel: {orderId_to_cancel}')

            response = await self.client.cancel_order(
                symbol=trade.pair,
                orderId=orderId_to_cancel)
            logger.debug(json.dumps(response, indent=4))

        except Exception as e:
            logger.error(f"{e}")
            return False

        else:
            # Delete the orderIds from the exit orders
            if response.get('contingencyType', '') == 'OCO':
                response_stoploss, response_limit_maker = response['orderReports'][0], response['orderReports'][1]

                if response_stoploss['status'] == ORDER_STATUS_CANCELED:
                    logger.info(f'{trade._id}: "{response_stoploss["side"]}" "{response_stoploss["type"]}" order canceled: {response_stoploss["orderId"]}')
                
                if response_limit_maker['status'] == ORDER_STATUS_CANCELED:
                    logger.info(f'{trade._id}: "{response_limit_maker["side"]}" "{response_limit_maker["type"]}" order canceled: {response_limit_maker["orderId"]}')

            else:
                logger.info(f'{trade._id}: "{response["side"]}" "{response["type"]}" order canceled: {response["orderId"]}')
            return True


    async def _execute_market_buy(self, trade: Trade) -> bool:
        try:
            logger.debug(f'trade.enter: {asdict(trade.enter)}')
            response = await self.client.order_market_buy(
                symbol=trade.pair,
                quantity=trade.enter.quantity)
            logger.debug(json.dumps(response, indent=4))

        except Exception as e:
            logger.error(f"{e}")
            return False

        else:
            logger.info(f'{trade._id}: "{response["side"]}" "{response["type"]}" order placed: {response["orderId"]}')
            trade.enter.orderId = response['orderId']
            trade.status = EState.OPEN_ENTER
            TelegramBot.send_formatted_message('order_executed', asdict(trade.enter), [response["side"], trade.strategy, trade.pair], [trade._id])

        return True


    async def _execute_market_sell(self, trade: Trade) -> bool:
        try:
            logger.debug(f'trade.exit: {asdict(trade.exit)}')
            response = await self.client.order_market_sell(
                symbol=trade.pair,
                quantity=trade.exit.quantity)
            logger.debug(json.dumps(response, indent=4))

        except Exception as e:
            logger.error(f"{e}")
            return False

        else:
            logger.info(f'{trade._id}: "{response["side"]}" "{response["type"]}" order placed: {response["orderId"]}')
            trade.exit.orderId = response['orderId']
            trade.status = EState.OPEN_EXIT
            TelegramBot.send_formatted_message('order_executed', asdict(trade.exit), [response["side"], trade.strategy, trade.pair], [trade._id])
        
        return True


    async def execute_decision(self, trades:'list[Trade]') -> None:
        for trade in trades:

            if trade.command == ECommand.NONE:
                continue

            elif trade.command == ECommand.EXEC_ENTER:
                buy_method = f'_execute_{type(trade.enter).__name__.lower()}_buy'
                if await getattr(self, buy_method)(trade): 
                    trade.reset_command()
                else:
                    trade = None

            elif trade.command == ECommand.CANCEL:
                if await self._execute_cancel(trade):
                    trade.status = EState.CLOSED
                    trade.reset_command()

                
            elif trade.command == ECommand.UPDATE:
                # First cancel, then place the new order
                if await self._execute_cancel(trade):
   
                    sell_method = f'_execute_{type(trade.exit).__name__.lower()}_sell'
                    if await getattr(self, sell_method)(trade):
                        trade.reset_command()

                else:
                    '''
                    If the cancel failed, then the exit orders are still there.
                    So do not create new order and keep the status as exit_expired
                    '''
                    pass
                pass

            elif trade.command == ECommand.EXEC_EXIT:
                sell_method = f'_execute_{type(trade.exit).__name__.lower()}_sell'
                if await getattr(self, sell_method)(trade):
                    trade.reset_command()


async def sync_trades_with_orders(icarus_time_sec: int, trades: List[Trade], data_dict: dict, strategy_period_mapping: dict, order_info_mapping: Dict[int, OrderInfo]) -> None:
    quote_cur = 'USDT'
    for trade in trades:
        base_cur = trade.pair.replace(quote_cur,'') # How to get this info?
        strategy_min_scale = strategy_period_mapping[trade.strategy]

        if trade.status == EState.OPEN_ENTER:
            # Check order
            order_info = order_info_mapping[trade.enter.orderId]
            if order_info.order['status'] == ORDER_STATUS_FILLED:
                logger.debug(json.dumps(order_info.order, indent=4))
                logger.debug(json.dumps(order_info.trade_list, indent=4))

                sum_fee = 0
                for single_trade in order_info.trade_list:
                    if single_trade['commissionAsset'] == base_cur:
                        sum_fee = safe_sum(sum_fee, single_trade['commission'])
                    else:
                        logger.error('Commission asset is not base asset: "{}" != "{}"'.format(single_trade['commissionAsset'],base_cur))

                avg_price = safe_divide(order_info.order['cummulativeQuoteQty'], order_info.order['executedQty'])

                strategy_cycle_period_in_sec = time_scale_to_second(strategy_min_scale)
                time_value = int(order_info.order['updateTime']/1000)
                # Get the start time of the current candle
                execution_time = round_to_period(time_value, strategy_cycle_period_in_sec, direction='floor')

                trade.set_result_enter(execution_time, #int(order['transactTime']),
                    price=avg_price,
                    quantity=float(order_info.order['executedQty']),
                    fee=sum_fee)
                TelegramBot.send_formatted_message('order_filled', asdict(trade.result.enter), [order_info.order["side"], trade.strategy, trade.pair], [trade._id])
                
            elif hasattr(trade.enter, 'expire') and trade.enter.expire <= icarus_time_sec:
                trade.status = EState.ENTER_EXP

        elif trade.status == EState.OPEN_EXIT:
            order_info = order_info_mapping[trade.exit.orderId]

            if order_info.order['status'] == ORDER_STATUS_FILLED:
                logger.debug(json.dumps(order_info.order, indent=4))
                logger.debug(json.dumps(order_info.trade_list, indent=4))

                sum_fee = 0
                for single_trade in order_info.trade_list:
                    if single_trade['commissionAsset'] == quote_cur:
                        sum_fee = safe_sum(sum_fee, single_trade['commission'])
                    else:
                        logger.error('Commission asset is not quote asset: "{}" != "{}"'.format(single_trade['commissionAsset'], quote_cur))
                
                avg_price = safe_divide(order_info.order['cummulativeQuoteQty'], order_info.order['executedQty'])

                strategy_cycle_period_in_sec = time_scale_to_second(strategy_min_scale)
                time_value = int(order_info.order['updateTime']/1000)
                # Get the start time of the current candle
                execution_time = round_to_period(time_value, strategy_cycle_period_in_sec, direction='floor')

                # Evaluate cause
                if order_info.order['type'] == 'MARKET':
                    trade_cause = ECause.MARKET
                else:
                    trade_cause = ECause.LIMIT

                trade.set_result_exit(execution_time, #int(order['transactTime']),
                    price=avg_price,
                    quantity=float(order_info.order['executedQty']),
                    fee=sum_fee,
                    cause=trade_cause)
                TelegramBot.send_formatted_message('trade_closed', asdict(trade.result), [trade.strategy, trade.pair], [trade._id])

                
            elif order_info.order['status'] == ORDER_STATUS_EXPIRED:
                # NOTE: Here is OCO order
                stop_limit_info = order_info_mapping[trade.exit.stop_limit_orderId]
                logger.debug(json.dumps(order_info.order, indent=4))
                logger.debug(json.dumps(stop_limit_info.order, indent=4))

                sum_fee = 0
                for single_trade in stop_limit_info.trade_list:
                    if single_trade['commissionAsset'] == quote_cur:
                        sum_fee = safe_sum(sum_fee, single_trade['commission'])
                    else:
                        logger.error('Commission asset is not quote asset: "{}" != "{}"'.format(single_trade['commissionAsset'], quote_cur))
                
                strategy_cycle_period_in_sec = time_scale_to_second(strategy_min_scale)
                time_value = int(stop_limit_info.order['updateTime']/1000)
                # Get the start time of the current candle
                execution_time = round_to_period(time_value, strategy_cycle_period_in_sec, direction='floor')

                trade.set_result_exit(execution_time,
                    price=float(stop_limit_info.order['price']),            # Price cannot change
                    quantity=float(stop_limit_info.order['executedQty']),
                    fee=sum_fee,
                    cause=ECause.STOP_LIMIT)
                TelegramBot.send_formatted_message('trade_closed', asdict(trade.result), [trade.strategy, trade.pair], [trade._id])

                
            elif hasattr(trade.exit, 'expire') and trade.exit.expire <= icarus_time_sec:
                trade.status = EState.EXIT_EXP

