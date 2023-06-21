import talib as ta
from exceptions import NotImplementedException
import numpy as np

    
class TALibIndicators():
    async def _bband(self, candlesticks, **kwargs):
        upperband, middleband, lowerband = ta.BBANDS(candlesticks['close'], **kwargs)
        return {'upper':list(upperband), 'middle': list(middleband), 'lower':list(lowerband)}
    async def _dema(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _ema(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _ht_trendline(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _kama(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _ma(self, candlesticks, **kwargs): return list(ta.MA(candlesticks['close'], **kwargs))
    async def _mama(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _mavp(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _midpoint(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _midprice(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _sar(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _sarext(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _sma(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _t3(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _tema(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _trima(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _wma(self, candlesticks, **kwargs): raise NotImplementedException('indicator')


    # Momentum Indicators
    async def _adx(self, candlesticks, **kwargs): return list(ta.ADX(candlesticks['high'], candlesticks['low'], candlesticks['close'], **kwargs))
    async def _adxr(self, candlesticks, **kwargs): return list(ta.ADXR(candlesticks['high'], candlesticks['low'], candlesticks['close'], **kwargs))
    async def _apo(self, candlesticks, **kwargs): return list(ta.APO(candlesticks['high'], **kwargs))
    async def _aroon(self, candlesticks, **kwargs): 
        aroondown, aroonup = ta.AROON(candlesticks['high'], candlesticks['low'], **kwargs)
        return {'aroonup':list(aroonup), 'aroondown': list(aroondown)}
    async def _aroonosc(self, candlesticks, **kwargs): return list(ta.AROONOSC(candlesticks['high'], candlesticks['low'], **kwargs))
    async def _bop(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _cci(self, candlesticks, **kwargs): return list(ta.CCI(candlesticks['high'], candlesticks['low'], candlesticks['close'], **kwargs))
    async def _cmo(self, candlesticks, **kwargs): return list(ta.CMO(candlesticks['close'], **kwargs))
    async def _dx(self, candlesticks, **kwargs): return list(ta.DX(candlesticks['high'], candlesticks['low'], candlesticks['close'], **kwargs))
    async def _macd(self, candlesticks, **kwargs):
        macd, macdsignal, macdhist = ta.MACD(candlesticks['close'], **kwargs)
        return {'macd':list(macd), 'macdsignal': list(macdsignal), 'macdhist':list(macdhist)}
    async def _macdext(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _macdfix(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _mfi(self, candlesticks, **kwargs): return list(ta.MFI(candlesticks['high'], candlesticks['low'], candlesticks['close'], candlesticks['volume'], **kwargs))
    async def _minus_di(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _minus_dm(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _mom(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _plus_di(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _plus_dm(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _ppo(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _roc(self, candlesticks, **kwargs): return list(ta.ROC(candlesticks['close'], **kwargs))
    async def _rocp(self, candlesticks, **kwargs): return list(ta.ROCP(candlesticks['close'], **kwargs))
    async def _rocr(self, candlesticks, **kwargs): return list(ta.ROCR(candlesticks['close'], **kwargs))
    async def _rocr100(self, candlesticks, **kwargs): return list(ta.ROCR100(candlesticks['close'], **kwargs))
    async def _rsi(self, candlesticks, **kwargs): 
        return list(ta.RSI(candlesticks['close'], **kwargs))
    async def _stoch(self, candlesticks, **kwargs):
        slowk, slowd = ta.STOCH(candlesticks['high'], candlesticks['low'], candlesticks['close'], **kwargs)
        return {'slowk':list(slowk), 'slowd': list(slowd)}
    async def _stochf(self, candlesticks, **kwargs):
        fastk, fastd = ta.STOCHF(candlesticks['high'], candlesticks['low'], candlesticks['close'], **kwargs)
        return {'fastk':list(fastk), 'fastd': list(fastd)}
    async def _stochrsi(self, candlesticks, **kwargs):
        fastk, fastd = ta.STOCHF(candlesticks['close'], **kwargs)
        return {'fastk':list(fastk), 'fastd': list(fastd)}
    async def _trix(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _ultosc(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _willr(self, candlesticks, **kwargs):
        return list(ta.WILLR(candlesticks['high'], candlesticks['low'], candlesticks['close'], **kwargs))

    # Volume indicators
    async def _ad(self, candlesticks, **kwargs): return list(ta.AD(candlesticks['high'], candlesticks['low'], candlesticks['close'], candlesticks['volume']))
    async def _adosc(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _obv(self, candlesticks, **kwargs): return list(ta.OBV(candlesticks['close'], candlesticks['volume']))


    # Volatility indicators
    async def _atr(self, candlesticks, **kwargs): return list(ta.ATR( candlesticks['high'],  candlesticks['low'],  candlesticks['close']))
    async def _natr(self, candlesticks, **kwargs): return list(ta.NATR( candlesticks['high'],  candlesticks['low'],  candlesticks['close']))
    async def _trange(self, candlesticks, **kwargs): return list(ta.TRANGE( candlesticks['high'],  candlesticks['low'],  candlesticks['close']))


    # Price Transform
    async def _avgprice(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _medprice(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _typprice(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _wclprice(self, candlesticks, **kwargs): raise NotImplementedException('indicator')


    # Cycle Indicators
    async def _ht_dcperiod(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _ht_dcphase(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _ht_phasor(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _sine(self, candlesticks, **kwargs): raise NotImplementedException('indicator')
    async def _trendmode(self, candlesticks, **kwargs): raise NotImplementedException('indicator')


    # Pattern Recognition
    async def _trendmode(self, candlesticks, **kwargs): raise NotImplementedException('indicator')

    async def _cdl_handler(self, candlesticks, pattern, is_raw=False):
        pattern_name = pattern[1:].upper()
        flags = list(getattr(ta, pattern_name)(candlesticks['open'], candlesticks['high'], candlesticks['low'], candlesticks['close']))
        
        if is_raw:
            return flags
        
        return self._multiclass_pattern_handler(flags, candlesticks)

    def _singleclass_pattern_handler(self, flags, candlesticks, key='low', offset=0):
        indices = np.where(np.array(flags) != 0)[0]
        result = [None]*len(flags)
        for idx in indices:
            result[idx+offset] = candlesticks[key].iloc[idx+offset]
        return result


    def _multiclass_pattern_handler(self, flags, candlesticks):
        bearish_indices = np.where(np.array(flags) < 0)[0]
        bullish_indices = np.where(np.array(flags) > 0)[0]
        
        bearish_prices = [None]*len(flags)
        for idx in bearish_indices:
            bearish_prices[idx] = candlesticks['high'].iloc[idx]

        bullish_prices = [None]*len(flags)
        for idx in bullish_indices:
            bullish_prices[idx] = candlesticks['low'].iloc[idx]

        return {'bearish':bearish_prices, 'bullish':bullish_prices}
