from dataclasses import dataclass
import numpy as np
from itertools import groupby
from operator import itemgetter
from enum import Enum
import pandas as pd
import pandas_ta as pd_ta
import joblib

def enum_to_value(enum_element):
    return int(enum_element.value) if enum_element is not None else None

def value_to_enum(value, EnumClass):
    # Find the Enum element that corresponds to the given value
    try:
        enum_element = EnumClass(value)
        return enum_element
    except ValueError:
        return None

def array_to_enum(array, EnumClass):
    # Convert each value in the array to the corresponding Enum element
    enum_array = np.array([value_to_enum(value, EnumClass) for value in array], dtype=EnumClass)
    return enum_array

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

    async def _market_direction_aroonosc(self, analysis, **kwargs):
        analyzer = '_aroonosc'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'], timeperiod=kwargs.get('timeperiod',14))

        classification = np.where(np.round(np.nan_to_num(analysis_output),2) <= 0, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        return classification

    async def _market_direction_fractal_aroon(self, analysis, **kwargs):
        analyzer = '_fractal_aroon'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'], timeperiod=kwargs.get('timeperiod',14))

        classification = np.where(np.round(np.nan_to_num(analysis_output['aroondown']),2) > 80, Direction.DOWN, Direction.SIDE)
        classification = np.where(np.round(np.nan_to_num(analysis_output['aroonup']),2) > 80, Direction.UP, classification)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['aroonup']))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_aroon(self, analysis, **kwargs):
        analyzer = '_aroon'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'], timeperiod=kwargs.get('timeperiod',14))
        
        classification = np.where(np.round(np.nan_to_num(analysis_output['aroondown']),2) > 80, Direction.DOWN, Direction.SIDE)
        classification = np.where(np.round(np.nan_to_num(analysis_output['aroonup']),2) > 80, Direction.UP, classification)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['aroonup']))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_macd(self, analysis, **kwargs):
        analyzer = '_macd'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(np.round(np.nan_to_num(analysis_output['macdhist']),2) <= 0, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['macdhist']))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_rsi(self, analysis, **kwargs):
        analyzer = '_rsi'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'], timeperiod=kwargs.get('timeperiod',14))

        classification = np.where(np.round(np.nan_to_num(analysis_output),2) <= 50, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_stoch(self, analysis, **kwargs):
        analyzer = '_stoch'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(np.round(np.nan_to_num(analysis_output['slowk']),2) <= 50, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['slowk']))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_stochf(self, analysis, **kwargs):
        analyzer = '_stochf'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(np.round(np.nan_to_num(analysis_output['fastk']),2) <= 50, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['fastk']))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_stochrsi(self, analysis, **kwargs):
        analyzer = '_stochrsi'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(np.round(np.nan_to_num(analysis_output['fastk']),2) <= 50, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['fastk']))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_willr(self, analysis, **kwargs):
        analyzer = '_willr'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(np.round(np.nan_to_num(analysis_output),2) <= -50, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_cci(self, analysis, **kwargs):
        analyzer = '_cci'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(np.round(np.nan_to_num(analysis_output),2) <= 0, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_cmo(self, analysis, **kwargs):
        analyzer = '_cmo'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(np.round(np.nan_to_num(analysis_output),2) <= 0, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_mfi(self, analysis, **kwargs):
        analyzer = '_mfi'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(np.round(np.nan_to_num(analysis_output),2) <= 50, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_ultosc(self, analysis, **kwargs):
        analyzer = '_ultosc'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(np.round(np.nan_to_num(analysis_output),2) <= 50, Direction.DOWN, Direction.UP)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_dmi_3(self, analysis, **kwargs):
        analyzer = '_dmi'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

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

        return classification


    async def _market_direction_dmi(self, analysis, **kwargs):
        analyzer = '_dmi'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'])

        classification = np.where(
            np.round(np.nan_to_num(analysis_output['plus_di']),2) > np.round(np.nan_to_num(analysis_output['minus_di']),2),
            Direction.UP, Direction.DOWN)

        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['plus_di']))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_supertrend(self, analysis, **kwargs):
        # TODO:configure length

        analysis_output = pd_ta.supertrend(analysis['candlesticks']['high'], analysis['candlesticks']['low'], analysis['candlesticks']['close'], **kwargs)
        direction_col = analysis_output.iloc[:,1]
        classification = np.where(direction_col == -1, Direction.DOWN, Direction.UP)

        #nan_value_offset = np.count_nonzero(np.isnan(direction_col))
        classification[:kwargs.get('length',7)] = None

        return classification


    async def _market_direction_good_entry_2(self, analysis, **kwargs):
        analyzer = '_percentage_possible_change'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'], **kwargs)

        uptrend_top_th = kwargs.get('uptrend_top_th', 0.01) 
        uptrend_bot_th = kwargs.get('uptrend_bot_th', -0.005)

        classification = np.where((analysis_output['pos_change'] > uptrend_top_th) & (analysis_output['neg_change'] > uptrend_bot_th), Direction.UP, Direction.DOWN)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['pos_change']))
        classification[:nan_value_offset] = None

        return classification


    async def _market_direction_good_entry_3(self, analysis, **kwargs):
        analyzer = '_percentage_possible_change'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis['candlesticks'], **kwargs)

        uptrend_top_th = kwargs.get('uptrend_top_th', 0.01) 
        uptrend_bot_th = kwargs.get('uptrend_bot_th', -0.005)

        downtrend_top_th = kwargs.get('downtrend_top_th', 0.005)
        downtrend_bot_th = kwargs.get('downtrend_bot_th', -0.01)

        classification = np.where((analysis_output['pos_change'] > uptrend_top_th) & (analysis_output['neg_change'] > uptrend_bot_th), Direction.UP, Direction.SIDE)
        classification = np.where((analysis_output['pos_change'] <= downtrend_top_th) & (analysis_output['neg_change'] <= downtrend_bot_th), Direction.DOWN, classification)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output['pos_change']))
        classification[:nan_value_offset] = None # NOTE: Should be pad from the future
        # FIX ME
        return classification


    async def _market_direction_open2close_change(self, analysis, **kwargs):
        analyzer = '_open2close_change'

        if hasattr(self, analyzer):
            analysis_output = await getattr(self, analyzer)(analysis, **kwargs)

        classification = np.where(analysis_output == 1, Direction.UP, Direction.DOWN)
        nan_value_offset = np.count_nonzero(np.isnan(analysis_output))

        if nan_value_offset > 0:
            classification[-nan_value_offset:] = None # Pad from future

        return classification


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


    async def _market_regime_index(self, analysis, **kwargs):

        market_regime_dict = {}
        for indicator_name in kwargs['indicators']:
           market_regime_dict[indicator_name] = analysis[indicator_name]

        return market_regime_dict


    async def _market_direction_index(self, analysis, **kwargs):

        market_class_table = {}
        for indicator_name in kwargs['indicators']:
            market_class_table[indicator_name] = analysis[indicator_name]

        df = pd.DataFrame(market_class_table)

        window_size = kwargs.get('window',0)
        if window_size <= 0:
            return df
                
        exclude_last_column = kwargs.get('exclude_last_column', True)
        if exclude_last_column:
            df_to_shift = df.iloc[:,:-1]
        else:
            df_to_shift = df.iloc[:,:]

        final_df = df
        for i in range(1, window_size + 1):
            lag_df = df_to_shift.shift(i) # lag_df = df.iloc[:,:-1].shift(i)
            lag_df.columns = [col + '_lag_' + str(i)  for col in lag_df.columns]
 
            final_df = pd.concat([lag_df, final_df], axis=1)
        final_df.dropna(inplace=True)

        return final_df
    
    
    async def _market_regime_rsi(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_rsi'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_macd(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_macd'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_stoch(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_stoch'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_stochf(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_stochf'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_aroon(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_aroon'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_aroonosc(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_aroonosc'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_willr(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_willr'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_cci(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_cci'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_mfi(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_mfi'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_ultosc(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_ultosc'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_dmi(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_dmi'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_supertrend(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_supertrend'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_good_entry_3(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_good_entry_3'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_open2close_change(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_open2close_change'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_logisticregression(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_logisticregression'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes

    async def _market_regime_lstm(self, analysis, **kwargs):
        detected_market_regimes = await MarketClassification.detect_regime_instances(analysis['candlesticks'], analysis['market_direction_lstm'], kwargs.get('validation_threshold', 0))
        return detected_market_regimes