from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Category10, Category20, Category20b, Category20c
from bokeh.models import Legend, HoverTool
from analyzers.support_resistance import SRCluster, SREvent, SREventType
import numpy as np
from typing import List

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
    if type(analysis) == list:
        is_not_nan = ~np.isnan(analysis)
        p.diamond(source.data['open_time'][is_not_nan], np.array(analysis)[is_not_nan], size=20, color=kwargs.get('color',Category10[3][0]))
    elif type(analysis) == dict:
        num_of_class = max(len(analysis),3)
        for i, (key, value) in enumerate(analysis.items()):
            is_not_nan = ~np.isnan(value)
            p.diamond(source.data['open_time'][is_not_nan], np.array(value)[is_not_nan], size=20, color=Category10[num_of_class][i])

def support_resistance_handler(p: figure, source: ColumnDataSource, analysis: List[SRCluster], **kwargs):
    sr_details = kwargs.get('details', True)
    sr_type = kwargs.get('type', '')
    sr_cmap = kwargs.get('cmap')

    # Color Map params
    colormap_idx = 0
    start_idx = None

    for sr_cluster in analysis:

        if sr_cluster.distribution_score < 0:
            continue

        if start_idx == None:
            start_idx = sr_cluster.chunk_start_index

        # Change color of cluster based on the start_index of cluster
        if start_idx != sr_cluster.chunk_start_index:
            start_idx = sr_cluster.chunk_start_index
            colormap_idx += 1
            colormap_idx = colormap_idx % len(color_map_support)

        if sr_details:
            text_bot = "HorDist:{}, VerDist:{}, Dist:{}".format(
                sr_cluster.horizontal_distribution_score, 
                sr_cluster.vertical_distribution_score, 
                sr_cluster.distribution_score)

            text_top_left = "#MinMember: {}, #NumOfRetest:{}".format(sr_cluster.min_cluster_members,sr_cluster.number_of_retest)
            text_top_right = "#Frame:{}".format(sr_cluster.chunk_end_index-sr_cluster.chunk_start_index)

            xyxy = [source.data['open_time'][sr_cluster.validation_index], sr_cluster.price_min, source.data['open_time'][sr_cluster.chunk_end_index], sr_cluster.price_max]
            rect_center = ((xyxy[2] + xyxy[0])/2, (xyxy[3] + xyxy[1])/2)
            rect_width = xyxy[2] - xyxy[0]
            rect_height = xyxy[3] - xyxy[1]
            p.rect(x=rect_center[0], y=rect_center[1], width=rect_width, height=rect_height,
                color=sr_cmap[colormap_idx][1], alpha=0.5)
            
        p.line(source.data['open_time'][sr_cluster.chunk_start_index:sr_cluster.chunk_end_index+1], 
               sr_cluster.price_mean, line_color=sr_cmap[colormap_idx][0], line_width=4, line_dash = [1, 2])

    hover = HoverTool()
    hover.tooltips = [
        ("Horizontal Distance", "@cluster_data.horizontal_distribution_score"),
        ("Vertical Distance", "@cluster_data.vertical_distribution_score"),
        ("Total Distance", "@cluster_data.distribution_score"),
        ("Min Members", "@cluster_data.min_cluster_members"),
        ("Number of Retests", "@cluster_data.number_of_retest"),
        ("Frame Count", "@cluster_data.chunk_end_index - @cluster_data.chunk_start_index")
    ]

    p.add_tools(hover)

        #fplt.add_line((x[sr_cluster.chunk_start_index], sr_cluster.price_mean), 
        #    (x[sr_cluster.chunk_end_index], sr_cluster.price_mean), style='.', color=sr_cmap[colormap_idx][0], width=2, interactive=False)

#def support_birch(x, y, axes): disable_ax_bot(axes); support_resistance_handler(x, y, axes, **{'type':'Support', 'cmap':color_map_support})
#def resistance_birch(x, y, axes): disable_ax_bot(axes); support_resistance_handler(x, y, axes, **{'type':'Resistance', 'cmap':color_map_resistance})
#def support_optics(x, y, axes): disable_ax_bot(axes); support_resistance_handler(x, y, axes, **{'type':'Support', 'cmap':color_map_support})
#def resistance_optics(x, y, axes): disable_ax_bot(axes); support_resistance_handler(x, y, axes, **{'type':'Resistance', 'cmap':color_map_resistance})
def support_meanshift(p_candlesticks, p_analyzer, source, analysis): 
    kwargs = {'type':'Support', 'cmap':color_map_support}; 
    support_resistance_handler(p_candlesticks, source, analysis, **kwargs)
#def resistance_meanshift(x, y, axes): disable_ax_bot(axes); support_resistance_handler(x, y, axes, **{'type':'Resistance', 'cmap':color_map_resistance})
#def support_dbscan(x, y, axes): disable_ax_bot(axes); support_resistance_handler(x, y, axes, **{'type':'Support', 'cmap':color_map_support})
#def resistance_dbscan(x, y, axes): disable_ax_bot(axes); support_resistance_handler(x, y, axes, **{'type':'Resistance', 'cmap':color_map_resistance})
#def support_kmeans(x, y, axes): disable_ax_bot(axes); support_resistance_handler(x, y, axes, **{'type':'Support', 'cmap':color_map_support})
#def resistance_kmeans(x, y, axes): disable_ax_bot(axes); support_resistance_handler(x, y, axes, **{'type':'Resistance', 'cmap':color_map_resistance})


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