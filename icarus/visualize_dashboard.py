
import finplot as fplt
import json
from PyQt5.QtWidgets import QComboBox, QCheckBox, QWidget
from pyqtgraph import QtGui
import pyqtgraph as pg
import sys
import asyncio
from binance import AsyncClient
from brokers import backtest_wrapper
import datetime
from itertools import chain
import itertools
from analyzers import Analyzer
from visualization import indicator_plot
from functools import partial

from utils import get_pair_min_period_mapping
import mongo_utils
from objects import ECause, EState
from copy import deepcopy
from visualization import trade_plot, observer_plot
import pandas as pd

def change_asset(*args, **kwargs):
    '''Resets and recalculates everything, and plots for the first time.'''
    # save window zoom position before resetting  
    #fplt._savewindata(fplt.windows[0])

    symbol = ctrl_panel.symbol.currentText()
    interval = ctrl_panel.interval.currentText()
    indicator = ctrl_panel.indicators.currentText().lower()

    # remove any previous plots
    if ctrl_panel.autoclear.isChecked() or "symbol" in args:
        ax.reset()
        axo.reset()
        ax_bot.reset()
        axo_bot.reset()

    fplt.candlestick_ochl(data_dict[symbol][interval]['open close high low'.split()], ax=ax, colorfunc=fplt.strength_colorfilter)
    fplt.volume_ocv(data_dict[symbol][interval]['open close volume'.split()], ax=axo)

    # Visualize indicators
    if indicator != 'clean':
        if indicator == 'trades':
            trades = analysis_dict[symbol][interval]['trades']
            if trades['canceled']:
                trade_plot.plot_canceled_orders(deepcopy(trades['canceled']))
            if trades['closed']:
                trade_plot.plot_closed_orders(ax, deepcopy(trades['closed']))
        elif indicator[:3] == 'obs':
            observer = indicator.split('_', 1)[1]
            handler = getattr(observer_plot, observer)
            handler(data_dict[symbol][interval].index, analysis_dict['obs_'+observer], 
                {'ax':ax, 'axo':axo, 'ax_bot':ax_bot, 'axo_bot':axo_bot})

        elif hasattr(indicator_plot, indicator):
            plotter_name = indicator
        elif indicator[:3] == 'cdl':
            plotter_name = 'cdl_handler'
        elif 'market_regime' in indicator:
            plotter_name = 'market_regime_handler'
        else:
            ax.set_visible(xaxis=True)
            # restores saved zoom position, if in range
            fplt.refresh()
            return

        if indicator != 'trades' and indicator[:3] != 'obs':
            handler = getattr(indicator_plot, plotter_name)
            if indicator not in analysis_dict[symbol][interval]:
                indicator = indicator.rsplit('_', 1)[0]
            handler(data_dict[symbol][interval].index, analysis_dict[symbol][interval][indicator], 
                {'ax':ax, 'axo':axo, 'ax_bot':ax_bot, 'axo_bot':axo_bot})

    ax.set_visible(xaxis=True)
    # restores saved zoom position, if in range
    fplt.refresh()


def dark_mode_toggle(dark):

    # first set the colors we'll be using
    if dark:
        fplt.foreground = '#777'
        fplt.background = '#090c0e'
        fplt.candle_bull_color = fplt.candle_bull_body_color = '#0b0'
        fplt.candle_bear_color = '#a23'
        volume_transparency = '6'
    else:
        fplt.foreground = '#444'
        fplt.background = fplt.candle_bull_body_color = '#fff'
        fplt.candle_bull_color = '#380'
        fplt.candle_bear_color = '#c50'
        volume_transparency = 'c'
    fplt.volume_bull_color = fplt.volume_bull_body_color = fplt.candle_bull_color + volume_transparency
    fplt.volume_bear_color = fplt.candle_bear_color + volume_transparency
    fplt.cross_hair_color = fplt.foreground+'8'
    fplt.draw_line_color = '#888'
    fplt.draw_done_color = '#555'

    pg.setConfigOptions(foreground=fplt.foreground, background=fplt.background)
    # control panel color
    if ctrl_panel is not None:
        p = ctrl_panel.palette()
        p.setColor(ctrl_panel.darkmode.foregroundRole(), pg.mkColor(fplt.foreground))
        ctrl_panel.darkmode.setPalette(p)

    # window background
    for win in fplt.windows:
        win.setBackground(fplt.background)

    # axis, crosshair, candlesticks, volumes
    axs = [ax for win in fplt.windows for ax in win.axs]
    vbs = set([ax.vb for ax in axs])
    axs += fplt.overlay_axs
    axis_pen = fplt._makepen(color=fplt.foreground)
    for ax in axs:
        ax.axes['left']['item'].setPen(axis_pen)
        ax.axes['left']['item'].setTextPen(axis_pen)
        ax.axes['bottom']['item'].setPen(axis_pen)
        ax.axes['bottom']['item'].setTextPen(axis_pen)
        if ax.crosshair is not None:
            ax.crosshair.vline.pen.setColor(pg.mkColor(fplt.foreground))
            ax.crosshair.hline.pen.setColor(pg.mkColor(fplt.foreground))
            ax.crosshair.xtext.setColor(fplt.foreground)
            ax.crosshair.ytext.setColor(fplt.foreground)
        for item in ax.items:
            if isinstance(item, fplt.FinPlotItem):
                isvolume = ax in fplt.overlay_axs
                if not isvolume:
                    item.colors.update(
                        dict(bull_shadow      = fplt.candle_bull_color,
                             bull_frame       = fplt.candle_bull_color,
                             bull_body        = fplt.candle_bull_body_color,
                             bear_shadow      = fplt.candle_bear_color,
                             bear_frame       = fplt.candle_bear_color,
                             bear_body        = fplt.candle_bear_color))
                else:
                    item.colors.update(
                        dict(bull_frame       = fplt.volume_bull_color,
                             bull_body        = fplt.volume_bull_body_color,
                             bear_frame       = fplt.volume_bear_color,
                             bear_body        = fplt.volume_bear_color))
                item.repaint()


def create_ctrl_panel(win, pairs, time_scales, indicators):
    panel = QWidget(win)
    panel.move(150, 0)
    win.scene().addWidget(panel)
    layout = QtGui.QGridLayout(panel)

    panel.symbol = QComboBox(panel)
    [panel.symbol.addItem(pair) for pair in pairs]
    panel.symbol.setCurrentIndex(0)
    layout.addWidget(panel.symbol, 0, 0)
    panel.symbol.currentTextChanged.connect(partial(change_asset, "symbol"))

    layout.setColumnMinimumWidth(1, 30)

    panel.interval = QComboBox(panel)
    [panel.interval.addItem(scale) for scale in time_scales]
    panel.interval.setCurrentIndex(0)
    layout.addWidget(panel.interval, 0, 2)
    panel.interval.currentTextChanged.connect(partial(change_asset, "interval"))

    layout.setColumnMinimumWidth(3, 30)

    panel.indicators = QComboBox(panel)
    panel.indicators.addItem('clean')
    [panel.indicators.addItem(ind) for ind in indicators]

    panel.indicators.setCurrentIndex(0)
    layout.addWidget(panel.indicators, 0, 4)
    panel.indicators.currentTextChanged.connect(partial(change_asset, "indicators"))

    layout.setColumnMinimumWidth(5, 30)

    panel.darkmode = QCheckBox(panel)
    panel.darkmode.setText('Haxxor mode')
    panel.darkmode.setCheckState(2)
    panel.darkmode.toggled.connect(dark_mode_toggle)
    layout.addWidget(panel.darkmode, 0, 6)

    layout.setColumnMinimumWidth(5, 30)

    panel.autoclear = QCheckBox(panel)
    panel.autoclear.setText('Autoclear')
    panel.autoclear.setCheckState(2)
    layout.addWidget(panel.autoclear, 0, 8)

    return panel


'''
async def get_trade_data(bwrapper, mongocli, config):
    start_time = datetime.datetime.strptime(config['backtest']['start_time'], "%Y-%m-%d %H:%M:%S")
    start_timestamp = int(datetime.datetime.timestamp(start_time))*1000
    end_time = datetime.datetime.strptime(config['backtest']['end_time'], "%Y-%m-%d %H:%M:%S")
    end_timestamp = int(datetime.datetime.timestamp(end_time))*1000

    # Create pools for pair-scales
    meta_data_pool = []
    for strategy_tag, strategy in config['strategy'].items():
        meta_data_pool.append(list(itertools.product([strategy_tag], strategy['time_scales'], strategy['pairs'])))

    meta_data_pool = list(set(chain(*meta_data_pool)))

    dashboard_data_pack = {}
    
    for strategy_tag, timeframe, symbol in meta_data_pool:
        canceled = await mongo_utils.do_find_trades(mongocli, 'hist-trades', {'result.cause':ECause.ENTER_EXP, 'pair':symbol, 'strategy':strategy_tag})
        closed = await mongo_utils.do_find_trades(mongocli, 'hist-trades', {'result.cause':{'$in':[ECause.MARKET, ECause.STOP_LIMIT, ECause.LIMIT]}, 'pair':symbol, 'strategy':strategy_tag})

        if symbol not in dashboard_data_pack:
            dashboard_data_pack[symbol] = {}
          
        if timeframe not in dashboard_data_pack[symbol]:
            dashboard_data_pack[symbol][timeframe] = {}

        dashboard_data_pack[symbol][timeframe]['trades'] = {
            'canceled':canceled,
            'closed':closed
        }

    return dashboard_data_pack

'''
async def get_trade_data(bwrapper, mongocli, config):
    start_time = datetime.datetime.strptime(config['backtest']['start_time'], "%Y-%m-%d %H:%M:%S")
    start_timestamp = int(datetime.datetime.timestamp(start_time))*1000
    end_time = datetime.datetime.strptime(config['backtest']['end_time'], "%Y-%m-%d %H:%M:%S")
    end_timestamp = int(datetime.datetime.timestamp(end_time))*1000

    pair_scale_mapping = await get_pair_min_period_mapping(config)

    df_list, dashboard_data_pack = [], {}
    for pair,scale in pair_scale_mapping.items(): 
        df_list.append(bwrapper.get_historical_klines(start_timestamp, end_timestamp, pair, scale))
        dashboard_data_pack[pair]={}
    
    df_pair_list = list(await asyncio.gather(*df_list))

    for idx, item in enumerate(pair_scale_mapping.items()):
        canceled = await mongo_utils.do_find_trades(mongocli, 'hist-trades', {'result.cause':ECause.ENTER_EXP, 'pair':item[0]})
        closed = await mongo_utils.do_find_trades(mongocli, 'hist-trades', {'result.cause':{'$in':[ECause.MARKET, ECause.STOP_LIMIT, ECause.LIMIT]}, 'pair':item[0]})
        
        if item[1] not in dashboard_data_pack[item[0]]:
            dashboard_data_pack[item[0]][item[1]] = {}

        dashboard_data_pack[item[0]][item[1]]['trades'] = {
            'df':df_pair_list[idx],
            'canceled':canceled,
            'closed':closed
        }

    return dashboard_data_pack


async def get_observer_data(mongocli, config):
    dashboard_data_pack = {}
    
    # Get observer objects
    for obs_type, obs_list in config['visualization']['observers'].items():
        if not hasattr(observer_plot, obs_type):
            continue
        df_observers = pd.DataFrame(list(await mongocli.do_find('observer',{'type':obs_type})))
        df_obs_data = pd.DataFrame(df_observers['data'].to_list())
        df_obs_data.set_index(df_observers['ts']*1000, inplace=True)
        df_obs_data = df_obs_data[obs_list]
        dashboard_data_pack['obs_'+obs_type] = df_obs_data

    return dashboard_data_pack


def analysis_dashboard(pair_pool, time_scale_pool, indicator_pool, title='Buy/Sell Plot'):

    global ctrl_panel, ax, axo, ax_bot, axo_bot, pair_data, pair_analysis
    pair_data = data_dict
    pair_analysis = analysis_dict

    print("buy sell dahsboard")
    
    # Set dashboard specifics
    fplt.display_timezone = datetime.timezone.utc
    fplt.y_pad = 0.07 # pad some extra (for control panel)
    fplt.max_zoom_points = 7
    fplt.autoviewrestore()
    ax,ax_bot = fplt.create_plot(title, rows=2)
    axo = ax.overlay()
    axo_bot = ax_bot.overlay()
    ax_bot.hide()
    ax_bot.vb.setBackgroundColor(None) # don't use odd background color
    ax.set_visible(xaxis=True)

    ctrl_panel = create_ctrl_panel(ax.vb.win, pair_pool, time_scale_pool, indicator_pool)
    dark_mode_toggle(False)
    change_asset()

    fplt.show()


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

async def visualize_dashboard(bwrapper: backtest_wrapper.BacktestWrapper, config):

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

    global data_dict, analysis_dict
    await bwrapper.obtain_candlesticks(meta_data_pool, start_timestamp, end_timestamp)
    data_dict = bwrapper.downloaded_data
    analyzer = Analyzer(config)
    analysis_dict = await analyzer.analyze(data_dict)

    config['mongodb']['clean'] = False
    mongo_client = mongo_utils.MongoClient(**config['mongodb'])
    trade_pair_dict = await get_trade_data(bwrapper, mongo_client, config)   
    analysis_dict = merge_dicts(trade_pair_dict, analysis_dict)

    observer_dict = await get_observer_data(mongo_client, config)  
    analysis_dict = merge_dicts(observer_dict, analysis_dict)

    analyzer_names = []
    analyzer_config = list(set(chain(*[list(layer.keys()) for layer in config['analysis']])))

    for key in analyzer_config:
        # NOTE: Following 2 lines are about the feature of generate_report tool
        #if 'plot' in config['analysis'][key].keys():
        #    analyzer_names = analyzer_names + [key+'_'+name for name in config['analysis'][key]['plot']]
        if hasattr(indicator_plot, key) or key[:3] == 'cdl' or 'market_regime' in key:
            analyzer_names.append(key)
    analyzer_names.append('trades')
    [analyzer_names.append(obs) for obs in observer_dict.keys()]
    analyzer_names.sort()
    analysis_dashboard(pair_pool, time_scale_pool, analyzer_names, title=f'Visualizing Time Frame: {config["backtest"]["start_time"]} - {config["backtest"]["end_time"]}')

async def main():

    client = await AsyncClient.create(**cred_info['Binance']['Production'])
    bwrapper = backtest_wrapper.BacktestWrapper(client, config)

    await visualize_dashboard(bwrapper, config)


if __name__ == '__main__':
    print(sys.argv)
    f = open(str(sys.argv[1]),'r')
    config = json.load(f)
    
    if len(sys.argv) >=3:
        config['credential_file'] = str(sys.argv[2])

    with open(config['credential_file'], 'r') as cred_file:
        cred_info = json.load(cred_file)
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

