from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import RdYlGn, Greys
from bokeh.models import HoverTool
from objects import Trade
import numpy as np
from typing import List
import pandas as pd

def individual_trades(p: figure, p_analyzer: figure, source: ColumnDataSource, analysis: List[Trade], trade_ids: List[str]):

    if len(analysis) < 0:
        return

    df = pd.DataFrame(analysis)
    df_trades =  df[df['_id'].isin(trade_ids)]
    trades(p, p_analyzer, source, df_trades, 'trades')


def trades(p: figure, p_analyzer: figure, source: ColumnDataSource, analysis: List[Trade], analyzer: str):

    if len(analysis) < 0:
        return

    if type(analysis) == pd.DataFrame:
        df = analysis
    elif type(analysis) == list:
        df = pd.DataFrame(analysis)
    
    df_results = df['result'].apply(pd.Series).add_prefix('result_')

    df_closed = df[df_results['result_cause'] != 'enter_expire']
    if not df_closed.empty:
        plot_closed_trades(p, source, df_closed)

    df_canceled = df[df_results['result_cause'] == 'enter_expire']
    if not df_canceled.empty:
        plot_expired_trades(p, source, df_canceled)


def plot_closed_trades(p, source, df_closed):
    df_results = df_closed['result'].apply(pd.Series).add_prefix('result_')
    df_enter = df_closed['enter'].apply(pd.Series).add_prefix('enter_')
    df_exit = df_closed['exit'].apply(pd.Series).add_prefix('exit_')

    df_result_exits = df_results['result_exit'].apply(pd.Series).add_prefix('result_exit_')
    df_result_enters = df_results['result_enter'].apply(pd.Series).add_prefix('result_enter_')
    df_source = pd.concat([df_closed[['_id','decision_time','strategy','order_stash', 'exit']], df_enter, df_exit, df_results[['result_cause','result_profit','result_live_time']], df_result_enters, df_result_exits], axis=1)
    df_source['decision_time'] = df_source['decision_time'].mul(1000).astype(np.int64)
    df_source['result_exit_time'] = df_source['result_exit_time'].mul(1000).astype(np.int64)
    df_source['result_enter_time'] = df_source['result_enter_time'].mul(1000).astype(np.int64)
    df_source['x'] = (df_source['result_enter_time'] + df_source['result_exit_time']).div(2)
    df_source['y'] = (df_source['result_enter_price'] + df_source['result_exit_price']).div(2)
    df_source['h'] = (df_source['result_enter_price'] - df_source['result_exit_price']).abs()
    df_source['w'] = (df_source['result_enter_time'] - df_source['result_exit_time'])
    df_source['color'] = df_source['result_profit'].apply(lambda x: RdYlGn[3][2] if x <= 0 else RdYlGn[3][0])

    df_source = add_enter_lines(source, df_source, 'result_exit_time', 'result_enter_price')
    df_source = add_exit_order_frames(df_source)

    data_source = ColumnDataSource(data=df_source)

    profit_rects = p.rect(x='x', y='y', width='w', height='h', color='color', source=data_source, alpha=0.5,
            legend_label='Closed Trades')
    exit_lines = p.multi_line(xs='xs_stashed', ys='ys_stashed', source=data_source, line_color='black', line_width=2, legend_label='Exit Order Frames')
    enter_lines = p.multi_line(xs='xs', ys='ys', source=data_source, line_color='blue', line_width=4, legend_label='Closed Trades')

    hover_closed = HoverTool()
    hover_closed.tooltips = [
        ("ID", "@_id"),
        ("Strategy", "@strategy"),
        ("Profit", "% @result_profit{0.0}"),
        ("Target Enter Price", "@enter_price{0.00}"),
        ("Enter Price", "@result_enter_price{0.00}"),
        ("Target Exit Price", "@exit_price{0.00}"),
        ("Exit Price", "@result_exit_price{0.00}"),
    ]
    hover_closed.renderers = [profit_rects, enter_lines, exit_lines]
    p.add_tools(hover_closed)


def plot_expired_trades(p, source, df_canceled):
    df_enter = df_canceled['enter'].apply(pd.Series).add_prefix('enter_')
    df_canceled_source = pd.concat([df_canceled[['_id','decision_time','strategy','order_stash']], df_enter[['enter_price','enter_expire']]], axis=1)
    df_canceled_source['decision_time'] = df_canceled_source['decision_time'].mul(1000).astype(np.int64)
    df_canceled_source['enter_expire'] = df_canceled_source['enter_expire'].mul(1000).astype(np.int64)
    df_canceled_source = add_enter_lines(source, df_canceled_source, 'enter_expire', 'enter_price')
    data_source_expire = ColumnDataSource(data=df_canceled_source)
    
    enter_expire_lines = p.multi_line(xs='xs', ys='ys', source=data_source_expire, line_color=Greys[3][1], line_width=4, legend_label='Enter Expire Trades')

    hover_expired = HoverTool()
    hover_expired.tooltips = [
        ("ID", "@_id"),
        ("Strategy", "@strategy"),
        ("Target Enter Price", "@enter_price{0.00}"),
    ]
    hover_expired.renderers = [enter_expire_lines]
    p.add_tools(hover_expired)


def add_enter_lines(source, df, end_of_line, price_level):
    raw_lines = {
        'xs': [],
        'ys': []
    }

    for index, row in df.iterrows():
        mask = (source.data['open_time'] >= row['decision_time']) & (source.data['open_time'] <= row[end_of_line])
        x_data = list(source.data['open_time'][mask])
        y_data = [row[price_level]]*len(x_data)
        raw_lines['xs'].append(x_data)
        raw_lines['ys'].append(y_data)

    df = df.assign(xs=raw_lines['xs'])
    df = df.assign(ys=raw_lines['ys'])
    return df

def add_exit_order_frames(df):
    raw_lines = {
        'xs': [],
        'ys': []
    }
    
    for index, row in df.iterrows():
        x_data = []
        y_data = []
        for stashed_order in row['order_stash']:
            stashed_order_expire = stashed_order['expire']*1000
            if 'stop_price' in stashed_order: # OCO
                x_data += [row['result_enter_time'], stashed_order_expire, stashed_order_expire, row['result_enter_time'], row['result_enter_time'], stashed_order_expire, stashed_order_expire]
                y_data += [row['result_enter_price'], row['result_enter_price'], row['exit_price'], row['exit_price'], stashed_order['stop_limit_price'], stashed_order['stop_limit_price'], row['result_enter_price']]
            elif 'expire' in stashed_order: # LIMIT
                x_data += [row['result_enter_time'], stashed_order_expire, stashed_order_expire, row['result_enter_time'], row['result_enter_time']]
                y_data += [row['result_enter_price'], row['result_enter_price'], stashed_order['price'], stashed_order['price'], row['result_enter_price']]
            else:  # MARKET
                # NOTE: It does not make sense the market orders to be stashed
                pass
        
        # Filled Exit orders
        if 'stop_price' in row['exit']: # OCO
            # NOTE: not tested
            x_data += [row['result_enter_time'], row['result_exit_time'], row['result_exit_time'], row['result_enter_time'], row['result_enter_time'], row['result_exit_time'], row['result_exit_time']]
            y_data += [row['result_enter_price'], row['result_enter_price'], row['exit_price'], row['exit_price'], row['exit_stop_limit_price'], row['exit_stop_limit_price'], row['result_enter_price']]
        elif 'expire' in row['exit']: # LIMIT
            x_data += [row['result_enter_time'], row['result_exit_time'], row['result_exit_time'], row['result_enter_time'], row['result_enter_time']]
            y_data += [row['result_enter_price'], row['result_enter_price'], row['exit_price'], row['exit_price'], row['result_enter_price']]
        else:  # MARKET
            x_data += [row['result_enter_time'], row['result_exit_time'], row['result_exit_time']]
            y_data += [row['result_enter_price'], row['result_enter_price'], row['result_exit_price']]

        raw_lines['xs'].append(x_data)
        raw_lines['ys'].append(y_data)

    df = df.assign(xs_stashed=raw_lines['xs'])
    df = df.assign(ys_stashed=raw_lines['ys'])

    return df
