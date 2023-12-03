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
from analyzers import support_resistance
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
from dashboard.cache_functions import *

st.set_page_config(layout="wide", page_title="Icarus Developer Dashboard")

# Obtain data to visualize
config = get_config()
credentials = get_credentials(config)

if config.get('ssh_tunnel', False):
    tunnel_server = SSHTunnelForwarder(
        tuple(credentials['ssh_tunnel']['ssh_address_or_host']),
        ssh_username=credentials['ssh_tunnel']['ssh_username'],
        ssh_pkey=credentials['ssh_tunnel']['ssh_pkey'],
        remote_bind_address=tuple(credentials['ssh_tunnel']['remote_bind_address']),
        local_bind_address=tuple(credentials['ssh_tunnel']['local_bind_address'])
    )
    tunnel_server.start()
    tunnel_server.check_tunnels()
    print(tunnel_server.tunnel_is_up, flush=True)

config['mongodb']['clean'] = False
symbols = get_symbols(config)
time_scales = get_time_scales(config)
analyzer_names = get_analyzer_names(config)
observer_names = get_observer_names(config)
observer_dict = get_observer_dict(config)
candle_start, candle_end = get_start_end_times(observer_dict)
data_dict = get_data_dict(config, credentials, candle_start, candle_end)
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
visualize_analysis_plot = st.sidebar.toggle('Visualize analysis plot')

if 'trades' in analysis_dict[symbol][timeframe]:
    analyzer_names.append('trades')

analyzer_names.sort()
selected_analyzers = st.sidebar.multiselect(
    "Select analyzers:",
    analyzer_names,
    max_selections=5,
)

current_trades = analysis_dict[symbol][timeframe].get('trades', [])

strategy_names = get_strategies(analysis_dict[symbol][timeframe].get('trades', []))
selected_strategies = st.sidebar.multiselect(
    "Select strategies:",
    strategy_names,
    disabled=('trades' not in selected_analyzers)
)

if len(selected_strategies) != 0:
    current_trades = filter_trades(analysis_dict[symbol][timeframe].get('trades', []), selected_strategies)

trade_ids = [trade._id for trade in current_trades]
selected_trades = st.sidebar.multiselect(
    "Select trades:",
    trade_ids
)

selected_observers = st.sidebar.multiselect(
    "Select observers:",
    observer_names,
    max_selections=5,
)

# Check Support Resistance Filters
is_there_sr_analzer = any(['support' in analysis or 'resistance' in analysis for analysis in selected_analyzers])
is_there_sr_observer = any(['support' in observer or 'resistance' in observer for observer in selected_observers])

if is_there_sr_observer or is_there_sr_analzer:
    st.sidebar.markdown("----")
    filter_score_dist = st.sidebar.slider('Select a range for distribution_score', 0, 1000, (0, 1000))
    filter_count_bounce = st.sidebar.slider('Number of Bounce', 0, 50, (0, 50))
    filter_count_break = st.sidebar.slider('Number of Break', 0, 50, (0, 50))
    filter_count_pass_horizontal = st.sidebar.slider('Number of Pass Horizontal', 0, 50, (0, 50))
    filter_count_pass_vertical = st.sidebar.slider('Number of Pass Vertical', 0, 50, (0, 50))
    filter_count_in_zone = st.sidebar.slider('Number of In Zone', 0, 10, (0, 10))

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
    ("Open", "@open{0.0000}"),
    ("High", "@high{0.0000}"),
    ("Low", "@low{0.0000}"),
    ("Close", "@close{0.0000}"),
    ("Timestamp", "@open_time"),
    ("Datetime", "@open_time{%F %T}")
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

    if analyzer == 'trades':
        analysis = current_trades
    else:
        analysis = analysis_dict[symbol][timeframe][analyzer]

    # Apply sr cluster filters
    if 'support' in analyzer or 'resistance' in analyzer and len(analysis) > 0:
        filter_dict = {
            'distribution_score': filter_score_dist,
            'count_bounce': filter_count_bounce,
            'count_break': filter_count_break,
            'count_pass_horizontal': filter_count_pass_horizontal,
            'count_pass_vertical': filter_count_pass_vertical,
            'count_in_zone': filter_count_in_zone,
        }
        analysis = support_resistance.multi_filter_by(filter_dict, analysis)

    plotter(p, p_analyzer, source, analysis, analyzer)

if len(selected_trades) > 0 and 'trades' not in selected_analyzers:
    plotter = getattr(trade_plot, 'individual_trades')
    plotter(p, p_analyzer, source, current_trades, selected_trades)

for observer in selected_observers:
    # Evaluate plotter function name
    if hasattr(observer_plot, observer):
        plotter = getattr(observer_plot, observer)
    else:
        continue

    observations = observer_dict[observer]
    if len(selected_trades) == 0:
        enable_details = False
        observations_filtered = observations
    else:
        enable_details = True
        observations_filtered = filter_observations(current_trades, observations, selected_trades)

    # Apply sr cluster filters
    if 'support' in observer or 'resistance' in observer:
        filter_dict = {
            'distribution_score': filter_score_dist,
            'count_bounce': filter_count_bounce,
            'count_break': filter_count_break,
            'count_pass_horizontal': filter_count_pass_horizontal,
            'count_pass_vertical': filter_count_pass_vertical,
            'count_in_zone': filter_count_in_zone,
        }
        analysis = support_resistance.multi_filter_by(filter_dict, observations_filtered)

    plotter(p, p_analyzer, source, observations_filtered, observer, enable_details=enable_details)

if visualize_analysis_plot:
    grid_list.append([p_analyzer])

# Create a grid layout with the two plots
grid = gridplot(grid_list, sizing_mode='stretch_width', toolbar_location='below')

# Streamlit Bokeh chart
st.bokeh_chart(grid, use_container_width=True)

if config.get('ssh_tunnel', False):
    tunnel_server.stop()