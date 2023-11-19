from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Category10, RdYlGn
import numpy as np


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
    legend_label = kwargs.get('analyzer','scatter')
    if type(analysis) == list:
        is_not_nan = ~np.isnan(analysis)
        getattr(p, style)(source.data['open_time'][is_not_nan], np.array(analysis)[is_not_nan], size=20, color=kwargs.get('color',Category10[3][0]), legend_label=legend_label)
    elif type(analysis) == dict:
        num_of_class = max(len(analysis),3)
        for i, (key, value) in enumerate(analysis.items()):
            is_not_nan = ~np.isnan(value)
            getattr(p, style)(source.data['open_time'][is_not_nan], np.array(value)[is_not_nan], size=20, color=Category10[num_of_class][i])
