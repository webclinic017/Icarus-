from dataclasses import dataclass
import numpy as np
from itertools import groupby
from operator import itemgetter
from hmmlearn.hmm import GaussianHMM
from enum import Enum

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

        classification = np.where(np.nan_to_num(analysis_output) <= 0, Direction.DOWN, Direction.UP)
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

        classification = np.where(np.nan_to_num(analysis_output['aroondown']) > 80, Direction.DOWN, Direction.SIDE)
        classification = np.where(np.nan_to_num(analysis_output['aroonup']) > 80, Direction.UP, classification)
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
        
        classification = np.where(np.nan_to_num(analysis_output['aroondown']) > 80, Direction.DOWN, Direction.SIDE)
        classification = np.where(np.nan_to_num(analysis_output['aroonup']) > 80, Direction.UP, classification)
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

        classification = np.where(np.nan_to_num(analysis_output['macdhist']) <= 0, Direction.DOWN, Direction.UP)
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

        classification = np.where(np.nan_to_num(analysis_output) <= 50, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

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


    async def _market_class_index(self, candlesticks, **kwargs):

        market_class_table = {}
        for key, value in kwargs.items():
            market_class_table[key] = await getattr(self, '_'+key)(candlesticks, **value)

        return market_class_table