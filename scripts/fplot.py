import argparse
import backtrader as bt
from tables import file
import talib
import collections
import finplot as fplt
import pandas as pd
import numpy as np
from collections import defaultdict

#kline_column_names = ["open_time", "open", "high", "low", "close", "volume", "close_time","quote_asset_volume", 
#                    "nbum_of_trades", "taker_buy_base_ast_vol", "taker_buy_quote_ast_vol", "ignore"]

def calc_volume_profile(df, period, bins):
    '''
    Calculate a poor man's volume distribution/profile by "pinpointing" each kline volume to a certain
    price and placing them, into N buckets. (IRL volume would be something like "trade-bins" per candle.)
    The output format is a matrix, where each [period] time is a row index, and even columns contain
    start (low) price and odd columns contain volume (for that price and time interval). See
    finplot.horiz_time_volume() for more info.
    '''
    data = []
    df['hlc3'] = (df.high + df.low + df.close) / 3 # assume this is volume center per each 1m candle
    _,all_bins = pd.cut(df.hlc3, bins, right=False, retbins=True)
    for _,g in df.groupby(pd.Grouper(key='open_time', freq=period)):
        t = g['open_time'].iloc[0]
        volbins = pd.cut(g.hlc3, all_bins, right=False)
        price2vol = defaultdict(float)
        for iv,vol in zip(volbins, g.volume):
            price2vol[iv.left] += vol
        data.append([t, sorted(price2vol.items())])
    return data


def fplot_volume_profile(filename=None):

    if filename == None:
        filename=args.filename

    df = pd.read_csv(filename)
    df['open_time'] = df['open_time'].astype('datetime64[ms]')
    #df = df.set_index(['open_time'])
    time_volume_profile = calc_volume_profile(df, period='d', bins=100) # try fewer/more horizontal bars (graphical resolution only)

    fplt.plot(df['open_time'], df['close'], legend='Price')
    fplt.candlestick_ochl(df[['open_time','open','close','high','low']], colorfunc=fplt.strength_colorfilter)

    fplt.horiz_time_volume(time_volume_profile, draw_va=0.7, draw_poc=1.0)
    fplt.show()



def fplot_2row(filename=None):

    if filename == None:
        filename=args.filename

    df = pd.read_csv(filename)
    df = df.set_index(['open_time'])
    print(df)
    ax, ax2 = fplt.create_plot('S&P 500 MACD', rows=2)

    # plot macd with standard colors first
    macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    df['macd_diff'] = macd - signal
    fplt.volume_ocv(df[['open','close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
    fplt.plot(macd, ax=ax2, legend='MACD')
    fplt.plot(signal, ax=ax2, legend='Signal')

    fplt.candlestick_ochl(df[['open','close','high','low']], ax=ax, colorfunc=fplt.strength_colorfilter)
    hover_label = fplt.add_legend('', ax=ax)
    axo = ax.overlay()
    fplt.volume_ocv(df[['open','close','volume']], ax=axo)
    fplt.plot(df['volume'].ewm(span=24).mean(), ax=axo, color=1)
    fplt.show()


def fplot_2row_scatter(filename=None):

    if filename == None:
        filename=args.filename

    df = pd.read_csv(filename)
    df = df.set_index(['open_time'])
    print(df)
    ax, ax2 = fplt.create_plot('S&P 500 MACD', rows=2)

    # plot macd with standard colors first
    macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    df['macd_diff'] = macd - signal
    fplt.volume_ocv(df[['open','close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
    fplt.plot(macd, ax=ax2, legend='MACD')
    fplt.plot(signal, ax=ax2, legend='Signal')

    fplt.candlestick_ochl(df[['open','close','high','low']], ax=ax, colorfunc=fplt.strength_colorfilter)
    hover_label = fplt.add_legend('', ax=ax)
    axo = ax.overlay()
    fplt.volume_ocv(df[['open','close','volume']], ax=axo)
    fplt.plot(df['volume'].ewm(span=24).mean(), ax=axo, color=1)



    #dft.plot(kind='labels', ax=ax) #TODO: Add labels for buy and sell prices


    fplt.show()


def fplot_volume(filename=None):

    if filename == None:
        filename=args.filename

    df = pd.read_csv(filename)
    df = df.set_index(['open_time'])

    ax = fplt.create_plot('S&P 500 MACD', rows=1)
    fplt.candlestick_ochl(df[['open','close','high','low']], ax=ax, colorfunc=fplt.strength_colorfilter)
    hover_label = fplt.add_legend('', ax=ax)

    # Add ax overlay
    axo = ax.overlay()
    fplt.volume_ocv(df[['open','close','volume']], ax=axo)
    fplt.plot(df['volume'].ewm(span=24).mean(), ax=axo, color=1)
    fplt.show()


def fplot(filename=None):

    if filename == None:
        filename=args.filename

    df = pd.read_csv(filename)
    df = df.set_index(['open_time'])
    fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']], colorfunc=fplt.strength_colorfilter)

    # Add period separator lines
    periods = pd.to_datetime(df.index, unit='ms').strftime('%H')
    last_period = ''
    for x,(period,price) in enumerate(zip(periods, df.close)):
        if period != last_period:
            fplt.add_line((x-0.5, price*0.5), (x-0.5, price*2), color='#bbb', style='--')
        last_period = period
    fplt.show()


def buy_sell(df):


    ax = fplt.create_plot('Buy/Sell')
    fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']], ax=ax, colorfunc=fplt.strength_colorfilter)

    # Add period separator lines
    periods = pd.to_datetime(df.index, unit='ms').strftime('%H')
    last_period = ''
    for x,(period,price) in enumerate(zip(periods, df.close)):
        if period != last_period:
            fplt.add_line((x-0.5, price*0.5), (x-0.5, price*2), color='#bbb', style='--')
        last_period = period

    point_buy = df['buy']-10
    point_buy.plot(kind='scatter', color=-2, width=2, ax=ax, zoomscale=False, style='^')

    point_sell = df['sell']+10
    point_sell.plot(kind='scatter', color=-1, width=2, ax=ax, zoomscale=False, style='v')
    
    fplt.show()


# Helper Functions

if __name__ == '__main__':

    # Instantiate the parser
    # python .\scripts\fplot.py --filename .\test\data\btcusdt_15m_202005121212_202005131213.csv
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--filename', type=str,)
    args = parser.parse_args()

    #fplot()
    buy_sell()
    #fplot_volume()
    #fplot_2row()
    #fplot_2row_scatter()
    #fplot_volume_profile()


