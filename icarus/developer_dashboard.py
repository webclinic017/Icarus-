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
    print("backtest wrapper")
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

@st.cache_data
@async_to_sync
async def get_analysis_dict(config, data_dict):
    analyzer = Analyzer(config)
    analysis_dict = await analyzer.analyze(data_dict)
    return analysis_dict


# Obtain data to visualize
config = get_config()
symbols = get_symbols(config)
time_scales = get_time_scales(config)
credentials = get_credentials(config)
analyzer_names = get_analyzer_names(config)
data_dict = get_data_dict(config, credentials)
analysis_dict = get_analysis_dict(config, data_dict)

# Configure dashboard
st.title("Icarus Developer Dashboard")


image = Image.open('resources/icarus-visual.jpg')
st.sidebar.image(image)
column1, column2 = st.sidebar.columns(2)
with column1:
    symbol = st.sidebar.selectbox("Select Symbol", symbols)
with column2:
    timeframe = st.sidebar.selectbox("Select Timeframe", time_scales)
st.sidebar.markdown("----")
analyzer = st.sidebar.selectbox("Select Analyzer", analyzer_names)
on = st.sidebar.toggle('Overwrite Indicator')

# Visualize Data
candle_width = time_scale_to_milisecond(timeframe)/2
df = data_dict[symbol][timeframe]
inc = df['close'] > df['open']
dec = df['open'] > df['close']
# Create a Bokeh candlestick chart
source = ColumnDataSource(df)
p = figure(title=f"{symbol} Candlestick Chart ({timeframe})", x_axis_label="Date", x_axis_type="datetime")
p.add_layout(Legend(click_policy="hide", orientation='horizontal', spacing=20), 'above')
low, high = source.data['open'].min(), source.data['close'].max()
diff = high - low
p.y_range = Range1d(low - 0.1 * diff, high + 0.1 * diff)
p.segment(x0="open_time", y0="high", x1="open_time", y1="low", source=source, line_color="black", legend_label='Candlestick')
p.vbar(df.index[inc], candle_width, df.open[inc], df.close[inc], fill_color="#26a69a", line_color="black", legend_label='Candlestick')
p.vbar(df.index[dec], candle_width, df.open[dec], df.close[dec], fill_color="#ef5350", line_color="black", legend_label='Candlestick')

# Add HoverTool to display open, high, close data
hover = HoverTool(
    tooltips=[
        ("Open", "@open{0}"),
        ("High", "@high{0}"),
        ("Low", "@low{0}"),
        ("Close", "@close{0}"),
        ("Date", "@open_time{%F %T}")
    ],
    formatters={
        "@open_time": "datetime",  # Format the date and time
    }
)
p.add_tools(hover)


# Add new overlay
p.extra_y_ranges = {"volume": Range1d(start=0, end=df['volume'].max() * 4)}
alpha = 0.5
p.vbar(df.index[inc], candle_width/2, df['volume'][inc], 0, fill_color="green", line_color="black", legend_label='Volume', y_range_name="volume", fill_alpha=alpha)
p.vbar(df.index[dec], candle_width/2, df['volume'][dec], 0, fill_color="red", line_color="black", legend_label='Volume', y_range_name="volume", fill_alpha=alpha)
p.add_layout(LinearAxis(y_range_name="volume", axis_label="Volume"), 'right')

# Access RSI data
rsi_data = analysis_dict[symbol][timeframe]['close']

# Create a new plot for RSI
p_rsi = figure(title=f"{symbol} RSI Indicator ({timeframe})", x_axis_label="Date", x_axis_type="datetime", x_range=p.x_range, plot_height=200)
p_rsi.line(df.index, rsi_data, line_color="blue", legend_label='RSI')

# Create a grid layout with the two plots
grid = gridplot([[p], [p_rsi]], sizing_mode='stretch_width')

# Streamlit Bokeh chart
st.bokeh_chart(grid, use_container_width=True)