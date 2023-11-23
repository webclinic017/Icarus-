import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, Legend, HoverTool
from bokeh.layouts import gridplot
from utils import time_scale_to_milisecond
import json
import sys
import asyncio
from binance import AsyncClient
from brokers import backtest_wrapper
from analyzers import Analyzer
from itertools import chain
import itertools
import datetime
from collections import defaultdict
from PIL import Image
from dashboard import analyzer_plot, trade_plot, observer_plot
import mongo_utils
from utils import get_pair_min_period_mapping
from objects import ECause
from sshtunnel import SSHTunnelForwarder


@st.cache_data
def get_config():
    f = open(str(sys.argv[1]),'r')
    return json.load(f)

@st.cache_data
def get_symbols(config):
    symbols = [strategy['pairs'] for strategy in config['strategy'].values()]
    return list(set(chain(*symbols)))

@st.cache_data
def get_time_scales(config):
    time_scales = [strategy['time_scales'] for strategy in config['strategy'].values()]   
    return list(set(chain(*time_scales)))

@st.cache_data
def get_analyzer_names(config):
    return list(set(chain(*[list(layer.keys()) for layer in config['analysis']])))

@st.cache_data
def get_observer_names(config):
    return list(set([obs_config['type'] for obs_config in config['observers']]))

@st.cache_data
def get_credentials(config):
    with open(config['credential_file'], 'r') as cred_file:
        cred_info = json.load(cred_file)
        return cred_info

def async_to_sync(async_function):
    def sync_function(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_function(*args, **kwargs))
        return result
    return sync_function

@st.cache_data
@async_to_sync
async def get_data_dict(config, credentials):
    client = await AsyncClient.create(**credentials['Binance']['Production'])
    bwrapper = backtest_wrapper.BacktestWrapper(client, config)
    start_time = datetime.datetime.strptime(config['backtest']['start_time'], "%Y-%m-%d %H:%M:%S")
    start_timestamp = int(datetime.datetime.timestamp(start_time))*1000
    end_time = datetime.datetime.strptime(config['backtest']['end_time'], "%Y-%m-%d %H:%M:%S")
    end_timestamp = int(datetime.datetime.timestamp(end_time))*1000

    # Create pools for pair-scales
    time_scale_pool = []
    pair_pool = []
    for strategy in config['strategy'].values():
        time_scale_pool.append(strategy['time_scales'])
        pair_pool.append(strategy['pairs'])

    time_scale_pool = list(set(chain(*time_scale_pool)))
    pair_pool = list(set(chain(*pair_pool)))
    meta_data_pool = list(itertools.product(time_scale_pool, pair_pool))
    await bwrapper.obtain_candlesticks(meta_data_pool, start_timestamp, end_timestamp)
    # Function to recursively convert defaultdict to a normal dictionary
    def convert_nested_defaultdict(d):
        if isinstance(d, defaultdict):
            return {k: convert_nested_defaultdict(v) for k, v in d.items()}
        else:
            return d

    return convert_nested_defaultdict(bwrapper.downloaded_data)

def merge_dicts(dict1, dict2):
    result = {}
    for key in dict1.keys() | dict2.keys():
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                result[key] = merge_dicts(dict1[key], dict2[key])
            else:
                result[key] = dict1[key]
        elif key in dict1:
            result[key] = dict1[key]
        else:
            result[key] = dict2[key]
    return result

async def get_trades(config):
    mongo_client = mongo_utils.MongoClient(**config['mongodb'])
    pair_scale_mapping = await get_pair_min_period_mapping(config)

    trade_ids = []
    trades_dict = {}

    for idx, (pair, scale) in enumerate(pair_scale_mapping.items()):
        canceled = await mongo_utils.do_find_trades(mongo_client, 'hist-trades', {'result.cause':ECause.ENTER_EXP, 'pair':pair})
        closed = await mongo_utils.do_find_trades(mongo_client, 'hist-trades', {'result.cause':{'$in':[ECause.MARKET, ECause.STOP_LIMIT, ECause.LIMIT]}, 'pair':pair})
        trade_ids += [str(closed_trade._id) for closed_trade in closed]
        
        if len(trade_ids) == 0:
            continue

        if pair not in trades_dict:
            trades_dict[pair] = {}

        if scale not in trades_dict[pair]:
            trades_dict[pair][scale] = {}
        trades_dict[pair][scale]['trades'] = closed + canceled

    return trades_dict

@st.cache_data
@async_to_sync
async def get_observer_dict(config):
    mongo_client = mongo_utils.MongoClient(**config['mongodb'])
    observer_dict = {}
    # Get observer objects
    for obs_config in config.get('observers', []):
        if not hasattr(observer_plot, obs_config['type']) or obs_config['type'] in observer_dict:
            continue

        observers = list(await mongo_client.do_find('observer',{'type':obs_config['type']}))
        df_observers = pd.DataFrame(observers)
        
        if df_observers.empty:
            continue

        observer_dtype = observers[0].get('dtype','')
        if observer_dtype != '':
            # TODO: Process the data packs acording to their dtype
            observer_dict[obs_config['type']] = observers
            continue
        
        df_obs_data = pd.DataFrame(df_observers['data'].to_list())
        df_obs_data.set_index(df_observers['ts']*1000, inplace=True)
        #df_obs_data = df_obs_data[obs_list]
        observer_dict[obs_config['type']] = df_obs_data

    return observer_dict

@st.cache_data
@async_to_sync
async def get_analysis_dict(config, data_dict):
    analyzer = Analyzer(config)
    analysis_dict = await analyzer.analyze(data_dict)
    trades_dict = await get_trades(config)
    if len(trades_dict) == 0:
        return analysis_dict
    return merge_dicts(analysis_dict, trades_dict)


@st.cache_data
def filter_observations(trades, _observations, selected_trades):
    df = pd.DataFrame(trades)
    df_trades =  df[df['_id'].isin(selected_trades)]

    observations_filtered = []
    for idx, trade_row in df_trades.iterrows():
        observation_dates = [trade_row['decision_time'], trade_row['result']['exit']['time']]
        for observation in _observations:
            if observation['ts'] not in observation_dates:
                continue
            observations_filtered.append(observation)
    return observations_filtered


@st.cache_data
def get_strategies(trades):
    return list(set([trade.strategy for trade in trades]))


@st.cache_data
def filter_trades(trades, selected_strategies):
    return [trade for trade in trades if trade.strategy in selected_strategies]