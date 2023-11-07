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
from dashboard import analyzer_plot

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
st.sidebar.title("Icarus Developer Dashboard")


image = Image.open('resources/icarus-visual.jpg')
st.sidebar.image(image)
column1, column2 = st.sidebar.columns(2)
with column1:
    symbol = st.sidebar.selectbox("Select Symbol", symbols)
with column2:
    timeframe = st.sidebar.selectbox("Select Timeframe", time_scales)
st.sidebar.markdown("----")
selected_analyzers = st.sidebar.multiselect(
    "Select analyzers:",
    analyzer_names,
    max_selections=5,
)

# Visualize Data
candle_width = time_scale_to_milisecond(timeframe)/2
df = data_dict[symbol][timeframe]

df_red = df.loc[(df['open'] >= df['close'])]
df_green = df.loc[(df['close'] >= df['open'])]

inc = df['close'] > df['open']
dec = df['open'] > df['close']
# Create a Bokeh candlestick chart
source = ColumnDataSource(df)
source_red = ColumnDataSource(df_red)
source_green = ColumnDataSource(df_green)

p = figure(title=f"{symbol} Candlestick Chart ({timeframe})", x_axis_label="Date", x_axis_type="datetime", toolbar_location='left')
p.add_layout(Legend(click_policy="hide", orientation='horizontal', spacing=20), 'below')
low, high = source.data['open'].min(), source.data['close'].max()
diff = high - low
p.y_range = Range1d(low - 0.1 * diff, high + 0.1 * diff)

segmnts_green = p.segment('open_time', 'high', 'open_time', 'low', color="#26a69a", source=source_green, legend_label='Candlesticks')
segmnts_red = p.segment('open_time', 'high', 'open_time', 'low', color="#ef5350", source=source_red, legend_label='Candlesticks')
vbars_green = p.vbar('open_time', candle_width, 'open', 'close', source=source_green, fill_color="#26a69a", line_color="#26a69a", legend_label='Candlesticks')
vbars_red = p.vbar('open_time', candle_width, 'close', 'open', source=source_red, fill_color="#ef5350", line_color="#ef5350", legend_label='Candlesticks')

# Add HoverTool to display open, high, close data
tooltips=[
    ("Open", "@open"),
    ("High", "@high"),
    ("Low", "@low"),
    ("Close", "@close"),
    ("Date", "@open_time{%F %T}")
]
formatters={
    "@open_time": "datetime"  # Format the date and time
}

red_hover = HoverTool(renderers=[vbars_red], tooltips=tooltips, formatters=formatters)
p.add_tools(red_hover)
green_hover = HoverTool(renderers=[vbars_green], tooltips=tooltips, formatters=formatters)
p.add_tools(green_hover)

# Add new overlay
p.extra_y_ranges = {"volume": Range1d(start=0, end=df['volume'].max() * 4)}
alpha = 0.5
p.vbar('open_time', candle_width/2, 'volume', 0, source=source_green, fill_color="green", line_color="green", legend_label='Volume', y_range_name="volume", fill_alpha=alpha)
p.vbar('open_time', candle_width/2, 'volume', 0, source=source_red, fill_color="red", line_color="red", legend_label='Volume', y_range_name="volume", fill_alpha=alpha)
p.add_layout(LinearAxis(y_range_name="volume", axis_label="Volume"), 'right')

grid_list = [[p]]

p_analyzer = figure(title=f"Analyzer", x_axis_label="Date", x_axis_type="datetime", x_range=p.x_range, plot_height=200, toolbar_location='left')

print(selected_analyzers)
for analyzer in selected_analyzers:
    if not hasattr(analyzer_plot, analyzer):
        continue

    analysis = analysis_dict[symbol][timeframe][analyzer]
    analysis_plotter = getattr(analyzer_plot, analyzer)
    analysis_plotter(p, p_analyzer, source, analysis)

grid_list.append([p_analyzer])

# Create a grid layout with the two plots
grid = gridplot(grid_list, sizing_mode='stretch_width')

# Streamlit Bokeh chart
st.bokeh_chart(grid, use_container_width=True)