from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Category10, RdYlGn
from bokeh.models import Legend, HoverTool
from bokeh.models.glyphs import Scatter
from analyzers.support_resistance import SRCluster, SREvent, SREventType, count_srevent
from objects import Trade, trade_to_dict
import numpy as np
from typing import List
from dataclasses import asdict
import pandas as pd

def trades(p: figure, p_analyzer: figure, source: ColumnDataSource, analysis: List[Trade], analyzer: str):

    raw_data = {
        'x': [],
        'y': [],
        'w': [],
        'h': [],
        'color': []
    }

    df = pd.DataFrame(analysis)
    df_canceled = df[df['exit'] == None]
    df_closed = df[df['exit'] != None]
    df_results = df_closed['result'].apply(pd.Series)
    df_result_exits = df_results['exit'].apply(pd.Series).add_prefix('result_exit_')
    df_result_enters = df_results['enter'].apply(pd.Series).add_prefix('result_enter_')

    df_source = pd.concat([df_closed[['_id','decision_time','strategy','order_stash']], df_result_enters, df_result_exits], axis=1)
    df_source['x'] = (df_source['decision_time'] + df_source['result_exit_time']).div(2).mul(1000)
    df_source['y'] = (df_source['result_enter_price'] + df_source['result_exit_price']).div(2).mul(1000)
    df_source['h'] = (df_source['result_enter_price'] - df_source['result_exit_price']).abs()
    df_source['w'] = (df_source['result_exit_time'] - df_source['decision_time']).mul(1000)

    data_source = ColumnDataSource(data=df_source)

    p.rect(x='x', y='y', width='w', height='h', source=data_source,
            legend_label=analyzer)
