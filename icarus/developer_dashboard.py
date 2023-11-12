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


st.set_page_config(layout="wide")

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
def get_filtered_observations(trades, observations):
    df = pd.DataFrame(trades)
    df_trades =  df[df['_id'].isin(selected_trades)]

    observations_filtered = []
    for idx, trade_row in df_trades.iterrows():
        observation_dates = [trade_row['decision_time'], trade_row['result']['exit']['time']]
        for observation in observations:
            if observation['ts'] not in observation_dates:
                continue
            observations_filtered.append(observation)
    return observations_filtered


# Obtain data to visualize
config = get_config()

if 'ssh_tunnel' in config:
    #tunnel_server = SSHTunnelForwarder(**config['ssh_tunnel'])
    tunnel_server = SSHTunnelForwarder(
        tuple(config['ssh_tunnel']['ssh_address_or_host']),
        ssh_username=config['ssh_tunnel']['ssh_username'],
        ssh_pkey=config['ssh_tunnel']['ssh_pkey'],
        remote_bind_address=tuple(config['ssh_tunnel']['remote_bind_address']),
        local_bind_address=tuple(config['ssh_tunnel']['local_bind_address'])
    )
    tunnel_server.start()

config['mongodb']['clean'] = False
symbols = get_symbols(config)
time_scales = get_time_scales(config)
credentials = get_credentials(config)
analyzer_names = get_analyzer_names(config)
observer_names = get_observer_names(config)
data_dict = get_data_dict(config, credentials)
analysis_dict = get_analysis_dict(config, data_dict)
observer_dict = get_observer_dict(config)

# Configure dashboard
st.sidebar.title("Icarus Developer Dashboard")


image = Image.open('resources/icarus-visual.jpg')
st.sidebar.image(image)
column1, column2 = st.sidebar.columns(2)
with column1:
    symbol = st.sidebar.selectbox("Select Symbol", symbols)
with column2:
    timeframe = st.sidebar.selectbox("Select Timeframe", time_scales)
st.sidebar.markdown("----")

if 'trades' in analysis_dict[symbol][timeframe]:
    analyzer_names.append('trades')

analyzer_names.sort()
selected_analyzers = st.sidebar.multiselect(
    "Select analyzers:",
    analyzer_names,
    max_selections=5,
)

trade_ids = [trade._id for trade in analysis_dict[symbol][timeframe].get('trades', [])]
selected_trades = st.sidebar.multiselect(
    "Select trades:",
    trade_ids
)

selected_observers = st.sidebar.multiselect(
    "Select observers:",
    observer_names,
    max_selections=5,
)


# Visualize Data
candle_width = time_scale_to_milisecond(timeframe)/2
df = data_dict[symbol][timeframe]

df_bearish = df.loc[(df['open'] >= df['close'])]
df_bullish = df.loc[(df['close'] >= df['open'])]

inc = df['close'] > df['open']
dec = df['open'] > df['close']
# Create a Bokeh candlestick chart
source = ColumnDataSource(df)
source_bearish = ColumnDataSource(df_bearish)
source_bullish = ColumnDataSource(df_bullish)

p = figure(title=f"{symbol} Candlestick Chart ({timeframe})", x_axis_label="Date", x_axis_type="datetime")
p.add_layout(Legend(click_policy="hide", orientation='horizontal', spacing=20), 'center')
low, high = source.data['open'].min(), source.data['close'].max()
diff = high - low
p.y_range = Range1d(low - 0.1 * diff, high + 0.1 * diff)

p.segment('open_time', 'high', 'open_time', 'low', color="#26a69a", source=source_bullish, legend_label='Candlesticks')
p.segment('open_time', 'high', 'open_time', 'low', color="#ef5350", source=source_bearish, legend_label='Candlesticks')
vbars_bullish = p.vbar('open_time', candle_width, 'open', 'close', source=source_bullish, fill_color="#26a69a", line_color="#26a69a", legend_label='Candlesticks')
vbars_bearish = p.vbar('open_time', candle_width, 'close', 'open', source=source_bearish, fill_color="#ef5350", line_color="#ef5350", legend_label='Candlesticks')

# Add HoverTool to display open, high, close data
tooltips=[
    ("Open", "@open{0.00}"),
    ("High", "@high{0.00}"),
    ("Low", "@low{0.00}"),
    ("Close", "@close{0.00}"),
    ("Date", "@open_time{%F %T}")
]
formatters={
    "@open_time": "datetime"  # Format the date and time
}

hover = HoverTool(renderers=[vbars_bearish, vbars_bullish], tooltips=tooltips, formatters=formatters)
p.add_tools(hover)

# Add new overlay
p.extra_y_ranges = {"volume": Range1d(start=0, end=df['volume'].max() * 4)}
alpha = 0.5
p.vbar('open_time', candle_width/2, 'volume', 0, source=source_bullish, fill_color="green", line_color="green", legend_label='Volume', y_range_name="volume", fill_alpha=alpha)
p.vbar('open_time', candle_width/2, 'volume', 0, source=source_bearish, fill_color="red", line_color="red", legend_label='Volume', y_range_name="volume", fill_alpha=alpha)
p.add_layout(LinearAxis(y_range_name="volume", axis_label="Volume"), 'right')

grid_list = [[p]]

p_analyzer = figure(title=f"Analyzer", x_axis_label="Date", x_axis_type="datetime", x_range=p.x_range, plot_height=200)
for analyzer in selected_analyzers:
    # Evaluate plotter function name
    if hasattr(analyzer_plot, analyzer):
        plotter = getattr(analyzer_plot, analyzer)
    elif hasattr(trade_plot, analyzer):
        plotter = getattr(trade_plot, analyzer)
    elif analyzer[:3] == 'cdl':
        plotter_name = 'pattern_visualizer'
        plotter = getattr(analyzer_plot, plotter_name)
    elif 'market_regime' in analyzer and analyzer != 'market_regime_index':
        plotter_name = 'market_regime_handler'
        plotter = getattr(analyzer_plot, plotter_name)
    else:
        continue

    analysis = analysis_dict[symbol][timeframe][analyzer]
    plotter(p, p_analyzer, source, analysis, analyzer)

if len(selected_trades) > 0 and 'trades' not in selected_analyzers:
    analyzer = 'trades'
    plotter = getattr(trade_plot, 'individual_trades')
    analysis = analysis_dict[symbol][timeframe][analyzer]
    plotter(p, p_analyzer, source, analysis, selected_trades)

for observer in selected_observers:
    # Evaluate plotter function name
    if hasattr(observer_plot, observer):
        plotter = getattr(observer_plot, observer)
    else:
        continue

    observations = observer_dict[observer]
    if len(selected_trades) == 0:
        plotter(p, p_analyzer, source, observations, observer)
        continue

    observations_filtered = get_filtered_observations(analysis_dict[symbol][timeframe]['trades'], observations)
    plotter(p, p_analyzer, source, observations_filtered, observer, enable_details=True)

grid_list.append([p_analyzer])

# Create a grid layout with the two plots
grid = gridplot(grid_list, sizing_mode='stretch_width', toolbar_location='below')

# Streamlit Bokeh chart
st.bokeh_chart(grid, use_container_width=True)

if 'ssh_tunnel' in config:
    tunnel_server.stop()