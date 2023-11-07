from bokeh.plotting import figure

def line_plotter(p: figure, source, analysis, **kwargs):
    p.line(source.data['open_time'], analysis, 
           line_color=kwargs.get('line_color','blue'),
           legend_label=kwargs.get('legend_label',''))
    
    if 'y_range' in kwargs:
        p.y_range.start, p.y_range.end = kwargs['y_range']


def aroon(p_candlesticks: figure, p_analyzer: figure, source, analysis, **kwargs):
    kwargs['y_range'] = (0, 100)
    kwargs['line_color'] = 'green'
    line_plotter(p_analyzer, source, analysis['aroonup'], **kwargs)
    kwargs['line_color'] = 'red'
    line_plotter(p_analyzer, source, analysis['aroondown'], **kwargs)
    print("aaaaaaaa")

def rsi(p_candlesticks: figure, p_analyzer: figure, source, analysis, **kwargs):
    kwargs['y_range'] = (0, 100)
    line_plotter(p_analyzer, source, analysis, **kwargs)
    print("aaaaaaaa")

def close(p_candlesticks: figure, p_analyzer: figure, source, analysis, **kwargs):
    line_plotter(p_candlesticks, source, analysis, **kwargs)
    print("bbbb")