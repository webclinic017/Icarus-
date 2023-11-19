import numpy as np
import pandas as pd

class Patterns():
    async def _bearish_fractal_5(self, analysis, **kwargs): return list(np.roll(analysis['candlesticks']['high'].rolling(5).apply(Patterns.is_resistance), -1))
    async def _bullish_fractal_5(self, analysis, **kwargs): return list(np.roll(analysis['candlesticks']['low'].rolling(5).apply(Patterns.is_support), -1))
    async def _bearish_fractal_3(self, analysis, **kwargs): return list(np.roll(analysis['candlesticks']['high'].rolling(3).apply(Patterns.is_resistance), -1))
    async def _bullish_fractal_3(self, analysis, **kwargs): return list(np.roll(analysis['candlesticks']['low'].rolling(3).apply(Patterns.is_support), -1))

    def is_resistance(serie):
        if len(serie) == 3 and serie.iloc[0] < serie.iloc[1] > serie.iloc[2]:
            return serie.iloc[1]
        elif len(serie) == 5 and serie.iloc[0] < serie.iloc[1] < serie.iloc[2] > serie.iloc[3] > serie.iloc[4]:
            return serie.iloc[2]
        return np.NaN

    def is_support(serie):
        if len(serie) == 3 and serie.iloc[0] > serie.iloc[1] < serie.iloc[2]:
            return serie.iloc[1]
        elif len(serie) == 5 and serie.iloc[0] > serie.iloc[1] > serie.iloc[2] < serie.iloc[3] < serie.iloc[4]:
            return serie.iloc[2]
        return np.NaN

    async def _bullish_aroon_break(self, analysis, **kwargs):
        shift = kwargs.get('shift',-1)
        marker_level = kwargs.get('marker_level', 'low')

        mask = np.roll(pd.Series(analysis['aroon']['aroondown']).rolling(2).apply(Patterns.is_aroon_break), shift)
        levels = analysis['candlesticks'][marker_level].copy()
        levels[mask != 1] = np.nan
        return levels.to_list()
    
    async def _bearish_aroon_break(self, analysis, **kwargs):
        shift = kwargs.get('shift',-1)
        marker_level = kwargs.get('marker_level', 'high')

        mask = np.roll(pd.Series(analysis['aroon']['aroonup']).rolling(2).apply(Patterns.is_aroon_break), shift)
        levels = analysis['candlesticks'][marker_level].copy()
        levels[mask != 1] = np.nan
        return levels.to_list()
        
    def is_aroon_break(serie):
        if serie.iloc[0] == 100 and serie.iloc[1] != 100:
            return True
        return False
