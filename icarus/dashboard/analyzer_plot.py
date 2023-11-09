from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Category10, Category20, Category20b, Category20c
from bokeh.models import Legend, HoverTool
from bokeh.models.glyphs import Scatter
from analyzers.support_resistance import SRCluster, SREvent, SREventType, count_srevent
import numpy as np
from typing import List
from dataclasses import asdict
import pandas as pd

BLUE='#0000FF'
PALE_BLUE='#CCCCFF'
GREEN='#00FF00'
PALE_GREEN='#CCFFCC'
RED='#FF0000'
PALE_RED='#FFCCCC'
CYAN='#00FFFF'
PALE_CYAN='#CCFFFF'
MAGENTA='#FF00FF'
PALE_MAGENTA='#FFCCFF'
YELLOW='#FFFB00'
PALE_YELLOW='#FFFBCC'

color_map_support_basic = [
    (BLUE, PALE_BLUE),
    (BLUE, PALE_BLUE)
]

color_map_resistance_basic = [
    (MAGENTA, PALE_MAGENTA),
    (MAGENTA, PALE_MAGENTA)
]

color_map_support = [
    (BLUE, PALE_BLUE),
    (GREEN, PALE_GREEN)
]

color_map_resistance = [
    (RED, PALE_RED),
    (YELLOW, PALE_YELLOW)
]

color_map_cluster = [
    (MAGENTA, PALE_MAGENTA),
    (CYAN, PALE_CYAN)
]


sr_event_colors = {
    SREventType.BOUNCE: BLUE,
    SREventType.BREAK: MAGENTA,
    SREventType.IN_ZONE: CYAN,
    SREventType.PASS_HORIZONTAL: YELLOW,
    SREventType.PASS_VERTICAL: PALE_RED
}

sr_event_marker = {
    SREventType.BOUNCE: 'circle',
    SREventType.BREAK: 'diamond',
    SREventType.IN_ZONE: 'star',
    SREventType.PASS_HORIZONTAL: 'square',
    SREventType.PASS_VERTICAL: 'square'
}


def line_plotter(p: figure, source: ColumnDataSource, analysis, **kwargs):
    number_of_lines = max(len(analysis),3)
    print(number_of_lines)
    if type(analysis) == dict:
        for i, (param, analysis_data) in enumerate(analysis.items()):
            p.line(source.data['open_time'], analysis_data, 
                legend_label=param,
                line_color=Category10[number_of_lines][i])
    elif type(analysis) == list:
        if all(isinstance(el, list) for el in analysis): 
            for sub_list in analysis:
                p.line(source.data['open_time'], sub_list)
        else:
            p.line(source.data['open_time'], analysis, 
                line_color=kwargs.get('line_color','blue'),
                legend_label=kwargs.get('legend_label',''))
    if 'y_range' in kwargs:
        p.y_range.start, p.y_range.end = kwargs['y_range']


def scatter_plotter(p: figure, source: ColumnDataSource, analysis, **kwargs):
    style = kwargs.get('style','diamond')
    if type(analysis) == list:
        is_not_nan = ~np.isnan(analysis)
        getattr(p, style)(source.data['open_time'][is_not_nan], np.array(analysis)[is_not_nan], size=20, color=kwargs.get('color',Category10[3][0]))
    elif type(analysis) == dict:
        num_of_class = max(len(analysis),3)
        for i, (key, value) in enumerate(analysis.items()):
            is_not_nan = ~np.isnan(value)
            getattr(p, style)(source.data['open_time'][is_not_nan], np.array(value)[is_not_nan], size=20, color=Category10[num_of_class][i])


def support_resistance_plotter(p: figure, source: ColumnDataSource, analysis: List[SRCluster], **kwargs):
    sr_details = kwargs.get('details', True)
    sr_type = kwargs.get('type', '')
    sr_cmap = kwargs.get('cmap')

    raw_data = {
        'x': [],
        'y': [],
        'line_color': []
    }

    chunk_start_indexes = list(set([sr_cluster.chunk_start_index for sr_cluster in analysis]))
    sorted(chunk_start_indexes)
    colormap_idx = 0
    color_start_index_mapping = dict()
    for index in chunk_start_indexes:
        colormap_idx += 1
        colormap_idx = colormap_idx % len(sr_cmap)
        color_start_index_mapping[index] = sr_cmap[colormap_idx]

    for sr_cluster in analysis:
        # NOTE: Normally this is used to filter sr clusters
        if sr_cluster.distribution_score < 0:
            continue

        if sr_details:
            xyxy = [source.data['open_time'][sr_cluster.validation_index], sr_cluster.price_min, source.data['open_time'][sr_cluster.chunk_end_index], sr_cluster.price_max]
            rect_center = ((xyxy[2] + xyxy[0])/2, (xyxy[3] + xyxy[1])/2)
            rect_width = xyxy[2] - xyxy[0]
            rect_height = xyxy[3] - xyxy[1]
            p.rect(x=rect_center[0], y=rect_center[1], width=rect_width, height=rect_height,
                color=color_start_index_mapping[sr_cluster.chunk_start_index][1], alpha=0.5,
                legend_label=sr_type+'Cluster')
            sr_event_plotter(p, source, sr_cluster)
        x_data = list(source.data['open_time'][sr_cluster.chunk_start_index:sr_cluster.chunk_end_index+1])
        y_data = [sr_cluster.price_mean]*len(x_data)
        raw_data['x'].append(x_data)
        raw_data['y'].append(y_data)
        raw_data['line_color'].append(color_start_index_mapping[sr_cluster.chunk_start_index][0])

    raw_data['line_dash'] = [[1,2]] * len(raw_data['y'])
    raw_data['distribution_score'] = [sr_cluster.distribution_score for sr_cluster in analysis]
    raw_data['price_mean'] = [sr_cluster.price_mean for sr_cluster in analysis]
    raw_data['number_of_members'] = [sr_cluster.number_of_members for sr_cluster in analysis]
    raw_data['distribution_efficiency'] = [sr_cluster.distribution_efficiency for sr_cluster in analysis]
    raw_data['bounce'] = [count_srevent(sr_cluster, SREventType.BOUNCE) for sr_cluster in analysis]
    raw_data['break'] = [count_srevent(sr_cluster, SREventType.BREAK) for sr_cluster in analysis]
    raw_data['pass_horizontal'] = [count_srevent(sr_cluster, SREventType.PASS_HORIZONTAL) for sr_cluster in analysis]
    raw_data['pass_vertical'] = [count_srevent(sr_cluster, SREventType.PASS_VERTICAL) for sr_cluster in analysis]

    data_source = ColumnDataSource(data=raw_data)

    # Plot center line of cluster
    sr_cluster_lines = p.multi_line(xs='x', ys='y', source=data_source, line_color='line_color', line_width=4, line_dash='line_dash', legend_label=sr_type+'Line')

    # Hover Tool
    hover = HoverTool()
    hover.tooltips = [
        ("PriceMean", "@price_mean"),
        ("# Members", "@number_of_members"),
        ("DistScore", "@distribution_score"),
        ("DistEff", "@distribution_efficiency"),
        ("# Bounce", "@bounce"),
        ("#  Break", "@break"),
        ("# Pass H", "@pass_horizontal"),
        ("# Pass V", "@pass_vertical")
    ]
    hover.renderers = [sr_cluster_lines]
    p.add_tools(hover)


def sr_event_plotter(p: figure, source: ColumnDataSource, sr_cluster: SRCluster, **kwargs):
    sr_events = sr_cluster.events
    raw_data = {'x': [], 'y': [], 'markers': [], 'colors': []}
    for sr_event in sr_events:
        raw_data['x'].append(source.data['open_time'][sr_cluster.chunk_start_index + sr_event.start_index])
        raw_data['y'].append(sr_cluster.price_mean)
        raw_data['markers'].append(sr_event_marker[sr_event.type])
        raw_data['colors'].append(sr_event_colors[sr_event.type])
    
    sr_event_data_source = ColumnDataSource(data=raw_data)

    glyph = Scatter(x='x', y='y', size=20, marker='markers', fill_color='colors', fill_alpha=0.7)
    p.add_glyph(sr_event_data_source, glyph)
    # TODO: Add legend item


def support_birch(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Support', 'cmap':color_map_support}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def resistance_birch(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Resistance', 'cmap':color_map_resistance}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def support_optics(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Support', 'cmap':color_map_support}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def resistance_optics(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Resistance', 'cmap':color_map_resistance}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def support_meanshift(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Support', 'cmap':color_map_support}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def resistance_meanshift(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Resistance', 'cmap':color_map_resistance}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def support_dbscan(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Support', 'cmap':color_map_support}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def resistance_dbscan(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Resistance', 'cmap':color_map_resistance}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def support_kmeans(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Support', 'cmap':color_map_support}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def resistance_kmeans(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Resistance', 'cmap':color_map_resistance}; 
    support_resistance_plotter(p_candlesticks, source, analysis, **kwargs)

def fractal_line_3(p_candlesticks, p_analyzer, source, analysis): kwargs = {}; line_plotter(p_candlesticks, source, analysis, **kwargs)
def fractal_aroon(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range': (0, 100)}; line_plotter(p_analyzer, source, analysis, **kwargs)
def fractal_aroonosc(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range': (-100, 100)}; line_plotter(p_analyzer, source, analysis, **kwargs)

def close(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_candlesticks, source, analysis)
def bullish_fractal_5(p_candlesticks, p_analyzer, source, analysis): kwargs = {'color':BLUE}; scatter_plotter(p_candlesticks, source, analysis, **kwargs)
def bearish_fractal_3(p_candlesticks, p_analyzer, source, analysis): kwargs = {'color':MAGENTA}; scatter_plotter(p_candlesticks, source, analysis, **kwargs)
def bullish_fractal_5(p_candlesticks, p_analyzer, source, analysis): kwargs = {'color':BLUE}; scatter_plotter(p_candlesticks, source, analysis, **kwargs)
def bearish_fractal_3(p_candlesticks, p_analyzer, source, analysis): kwargs = {'color':MAGENTA}; scatter_plotter(p_candlesticks, source, analysis, **kwargs)
def bullish_aroon_break(p_candlesticks, p_analyzer, source, analysis): kwargs = {'color':BLUE}; scatter_plotter(p_candlesticks, source, analysis, **kwargs)
def bearish_aroon_break(p_candlesticks, p_analyzer, source, analysis): kwargs = {'color':MAGENTA}; scatter_plotter(p_candlesticks, source, analysis, **kwargs)

def kaufman_efficiency_ratio(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range': (np.nanmin(analysis),np.nanmax(analysis))}; line_plotter(p_analyzer, source, analysis, **kwargs)
def price_density(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range': (np.nanmin(analysis),np.nanmax(analysis))}; line_plotter(p_analyzer, source, analysis, **kwargs)
def dmi(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_analyzer, source, analysis)
def supertrend_band(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_candlesticks, source, analysis)

####################################  TA-LIB Indicators Visualization ####################################

def ma(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_candlesticks, source, analysis)

def rsi(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range': (0, 100)}; line_plotter(p_analyzer, source, analysis, **kwargs)
def stoch(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range': (0, 100)}; line_plotter(p_analyzer, source, analysis, **kwargs)
def stochf(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range': (0, 100)}; line_plotter(p_analyzer, source, analysis, **kwargs)
def bband(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range': (0, 100)}; line_plotter(p_candlesticks, source, analysis, **kwargs)
def macd(p_candlesticks, p_analyzer, source, analysis): kwargs = {}; line_plotter(p_analyzer, source, analysis, **kwargs)

# Momentum Indicators
def adx(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range':(0, 100), 'band':(25,50)}; line_plotter(p_analyzer, source, analysis, **kwargs)
def adxr(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range':(0, 100), 'band':(25,50)}; line_plotter(p_analyzer, source, analysis, **kwargs)
def aroon(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range':(0, 100), 'band':(20,80)}; line_plotter(p_analyzer, source, analysis, **kwargs)
def aroonosc(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range':(-100, 100), 'band':(-50,50)}; line_plotter(p_analyzer, source, analysis, **kwargs)
def mfi(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range':(0, 100), 'band':(20,80)}; line_plotter(p_analyzer, source, analysis, **kwargs)
def roc(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_analyzer, source, analysis)
def rocp(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_analyzer, source, analysis)
def rocr(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_analyzer, source, analysis)
def rocr100(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_analyzer, source, analysis)

# Volume indicators
def obv(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range':(min(analysis),max(analysis))}; line_plotter(p_analyzer, source, analysis, **kwargs)
def ad(p_candlesticks, p_analyzer, source, analysis): kwargs = {'y_range':(min(analysis),max(analysis))}; line_plotter(p_analyzer, source, analysis, **kwargs)

# Volatility Indicators
def atr(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_analyzer, source, analysis)
def natr(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_analyzer, source, analysis)
def trange(p_candlesticks, p_analyzer, source, analysis): line_plotter(p_analyzer, source, analysis)

# TA-LIB Patterns
def pattern_visualizer(p_candlesticks, p_analyzer, source, analysis): scatter_plotter(p_candlesticks, source, analysis)
def direction_macd(p_candlesticks, p_analyzer, source, analysis): scatter_plotter(p_candlesticks, source, analysis)