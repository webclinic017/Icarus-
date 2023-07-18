from dataclasses import dataclass
import numpy as np
from itertools import groupby
from operator import itemgetter
from hmmlearn.hmm import GaussianHMM
from enum import Enum
import pandas as pd
import pandas_ta as pd_ta

class Direction(Enum):
    UP = 0
    DOWN = 1
    SIDE = 2

@dataclass
class MarketRegime():
    label: str
    start_ts: int
    end_ts: int
    start_price: float
    end_price: float
    duration_in_candle: int
    validation_price: float
    validation_ts: int
    time_scale: str = ''
    symbol: str = ''
    def __post_init__(self):
        self.perc_price_change = round(100 * (self.end_price - self.start_price) / self.start_price, 2)

        # TODO: Investıgate thıs logic
        if self.end_ts < self.validation_ts:
            self.perc_val_price_change = None
        else:
            self.perc_val_price_change = round(100 * (self.end_price - self.validation_price) / self.validation_price, 2)
        # NOTE: If the market_regime has 1 candle then perc_val_price_change value is

    def set_attribute(self, name, value):
        self.__setattr__(name, value)

@dataclass
class PredefinedMarketRegime(MarketRegime):
    pass

@dataclass
class UndefinedMarketRegime(MarketRegime):
    pass


class MarketClassification():

    async def _hmm(self, candlesticks, **kwargs):
        # TODO: It works fıne but what does it tell???
        close = np.array(candlesticks['close']).reshape(-1,1)
        daily_return = (1 - candlesticks['close'].div(candlesticks['close'].shift())).fillna(0)
        volatility_indicator = await self._atr(candlesticks)


        data_source = np.array(daily_return).reshape(-1,1)

        hmm_model = GaussianHMM(
            n_components=3, covariance_type="full", n_iter=1000
        ).fit(data_source)
        print("Model Score:", hmm_model.score(data_source))
        hidden_states = hmm_model.predict(data_source)
        #print(hmm_model.n_components)

        unique_states = np.unique(hidden_states)
        class_indexes = {}

        for state in unique_states:
            state_name = f'state_{state}'
            class_indexes[state_name] = np.where(hidden_states == state)[0]
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, class_indexes, kwargs.get('validation_threshold', 0))

        return detected_market_regimes

    async def _market_class_aroonosc(self, candlesticks, **kwargs):
        analyzer = '_aroonosc'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks, timeperiod=kwargs.get('timeperiod',14))

        classification = np.where(np.round(np.nan_to_num(analysis_output) <= 0,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification

        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_class_fractal_aroon(self, candlesticks, **kwargs):
        analyzer = '_fractal_aroon'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks, timeperiod=kwargs.get('timeperiod',14))

        classification = np.where(np.round(np.nan_to_num(analysis_output['aroondown']) > 80,2), Direction.DOWN, Direction.SIDE)
        classification = np.where(np.round(np.nan_to_num(analysis_output['aroonup']) > 80,2), Direction.UP, classification)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['aroonup']))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification

        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_aroon(self, candlesticks, **kwargs):
        analyzer = '_aroon'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks, timeperiod=kwargs.get('timeperiod',14))
        
        classification = np.where(np.round(np.nan_to_num(analysis_output['aroondown']) > 80,2), Direction.DOWN, Direction.SIDE)
        classification = np.where(np.round(np.nan_to_num(analysis_output['aroonup']) > 80,2), Direction.UP, classification)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['aroonup']))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification

        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_macd(self, candlesticks, **kwargs):
        analyzer = '_macd'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(np.round(np.nan_to_num(analysis_output['macdhist']) <= 0,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['macdhist']))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification

        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_rsi(self, candlesticks, **kwargs):
        analyzer = '_rsi'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks, timeperiod=kwargs.get('timeperiod',14))

        classification = np.where(np.round(np.nan_to_num(analysis_output) <= 50,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_stoch(self, candlesticks, **kwargs):
        analyzer = '_stoch'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(np.round(np.nan_to_num(analysis_output['slowk']),2) <= 50, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['slowk']))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_stochf(self, candlesticks, **kwargs):
        analyzer = '_stochf'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(np.round(np.nan_to_num(analysis_output['fastk']) <= 50,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['fastk']))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_stochrsi(self, candlesticks, **kwargs):
        analyzer = '_stochrsi'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(np.round(np.nan_to_num(analysis_output['fastk']) <= 50,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['fastk']))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_willr(self, candlesticks, **kwargs):
        analyzer = '_willr'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(np.round(np.nan_to_num(analysis_output) <= -50,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_cci(self, candlesticks, **kwargs):
        analyzer = '_cci'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(np.round(np.nan_to_num(analysis_output) <= 0,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_cmo(self, candlesticks, **kwargs):
        analyzer = '_cmo'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(np.round(np.nan_to_num(analysis_output) <= 0,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_mfi(self, candlesticks, **kwargs):
        analyzer = '_mfi'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(np.round(np.nan_to_num(analysis_output) <= 50,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_ultosc(self, candlesticks, **kwargs):
        analyzer = '_ultosc'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(np.round(np.nan_to_num(analysis_output) <= 50,2), Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_dmi_3(self, candlesticks, **kwargs):
        analyzer = '_dmi'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(
            np.logical_and(
                np.round(np.nan_to_num(analysis_output['plus_di']),2) > np.round(np.nan_to_num(analysis_output['minus_di']),2),
                np.round(np.nan_to_num(analysis_output['adx']),2) > 20
            ), Direction.UP, Direction.SIDE)
        classification = np.where(
            np.logical_and(
                np.round(np.nan_to_num(analysis_output['minus_di']),2) > np.round(np.nan_to_num(analysis_output['plus_di']),2),
                np.round(np.nan_to_num(analysis_output['adx']),2) > 20
            ), Direction.DOWN, classification)

        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['plus_di']))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_dmi(self, candlesticks, **kwargs):
        analyzer = '_dmi'
        output_format = kwargs.get('format','direction')

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(candlesticks)

        classification = np.where(
            np.round(np.nan_to_num(analysis_output['plus_di']),2) > np.round(np.nan_to_num(analysis_output['minus_di']),2),
            Direction.UP, Direction.DOWN)

        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['plus_di']))
        classification[:nan_value_offset] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def _market_class_supertrend(self, candlesticks, **kwargs):

        output_format = kwargs.get('format','direction')

        analysis_output = pd_ta.supertrend(candlesticks['high'], candlesticks['low'], candlesticks['close'], **kwargs)
        direction_col = analysis_output.iloc[:,1]
        classification = np.where(direction_col == -1, Direction.DOWN, Direction.UP)

        #nan_value_offset = np.count_nonzero(np.isnan(direction_col))
        classification[:kwargs.get('length',7)] = None

        if output_format == 'direction':
            return classification
        
        detected_market_regimes = await MarketClassification.detect_regime_instances(candlesticks, classification, kwargs.get('validation_threshold', 0))
        return detected_market_regimes


    async def detect_regime_instances(candlesticks, classification, validation_threshold):
        '''
        The essential features of classes is start and end timestamps. The rest is evaluated using these initial points
        '''
        # NOTE: Since some indicators are lagging ones with a timeperiod, no market class should be detected until this timeperiod completed
        # for the first time.

        class_indexes = {}
        class_indexes['downtrend'] = np.where(classification == Direction.DOWN)[0]
        class_indexes['ranging'] = np.where(classification == Direction.SIDE)[0]
        class_indexes['uptrend'] = np.where(classification == Direction.UP)[0]

        ts_index = candlesticks.index
        result = {}

        for class_name, filter_idx in class_indexes.items():
            class_item_list = []
            for k, g in groupby(enumerate(filter_idx), lambda ix: ix[0] - ix[1]):
                seq_idx = list(map(itemgetter(1), g))
                # NOTE: If the sq. length is 1 than it will not be displayed. Apply "seq_idx[-1]+1" if you need to

                if len(seq_idx) == 0 or len(seq_idx) < validation_threshold:
                    continue # Continue if the validation is performed and the class instance is not valid

                if seq_idx[0]+validation_threshold >= len(ts_index):
                    continue # Continue since the market regime is not validated during the current chart
                
                pmr = PredefinedMarketRegime(
                    label=class_name,
                    start_ts=ts_index[seq_idx[0]],
                    end_ts=ts_index[seq_idx[-1]],
                    duration_in_candle=len(seq_idx),
                    start_price=candlesticks['open'][ts_index[seq_idx[0]]],
                    end_price=candlesticks['close'][ts_index[seq_idx[-1]]],
                    validation_ts=ts_index[seq_idx[0]+validation_threshold],
                    validation_price=candlesticks['open'][ts_index[seq_idx[0]+validation_threshold]],
                )
                class_item_list.append(pmr)
            result[class_name] = class_item_list
            # TODO: No need to have a seperation between the calss instances since they are all labeled and self sufficient to be alice in visu.
        
        # TODO: Drop unused class
        return result

    async def _market_regime_index(self, candlesticks, **kwargs):

        market_regime_dict = {}
        for key, value in kwargs.items():
            market_regime_dict[key] = await getattr(self, '_'+key)(candlesticks, **value)

        return market_regime_dict

    async def _market_class_index(self, candlesticks, **kwargs):

        market_class_table = {}
        for key, value in kwargs.items():
            market_class_table[key] = await getattr(self, '_'+key)(candlesticks, **value)

        df = pd.DataFrame(market_class_table)
        return df