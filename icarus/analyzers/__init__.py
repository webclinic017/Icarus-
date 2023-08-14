import asyncio
from analyzers.talib_indicators import TALibIndicators
from analyzers.indicators import Indicators
from analyzers.market_classification import MarketClassification
from analyzers.patterns import Patterns
from analyzers.support_resistance import SupportResistance
from analyzers.machine_learning import MachineLearning
from typing import Dict

class Analyzer(Indicators, TALibIndicators, Patterns, SupportResistance, MarketClassification, MachineLearning):

    def __init__(self, config: Dict):
        #self.analysis_config = config.get('analysis')
        self.analysis_layers = config.get('analysis')
        self.time_scales_config = config.get('time_scales')

        return


    async def analyze(self, data_dict: Dict[str, Dict]):
        layer_0 = self.analysis_layers[0]
        analysis_dict=dict()
        for pair, data_obj in data_dict.items():
            analysis_obj = dict()

            for time_scale, candlesticks in data_obj.items():

                # Generate coroutines
                indicator_coroutines = []
                header = '_'
                indicator_method_names = list(map(lambda orig_string: header + orig_string, layer_0.keys()))
                indicator_names = list(layer_0.keys())

                for ind_method, ind_name in zip(indicator_method_names,indicator_names):
                    if hasattr(self, ind_method): 
                        indicator_coroutines.append(getattr(self, ind_method)(candlesticks, **layer_0.get(ind_name,{})))
                    elif ind_method[:4] == '_cdl':
                        indicator_coroutines.append(getattr(self, '_cdl_handler')(candlesticks, ind_method, **layer_0.get(ind_name,{})))

                    else: raise RuntimeError(f'Unknown Analyzer: "{ind_method}"')

                analysis_output = list(await asyncio.gather(*indicator_coroutines))


                # TODO: REFACTORUNG: Sample analyzer does not do patterns
                #       Just combine everyting. you can get rid of the prefixes if youwish
                # NOTE: Since coroutines are not reuseable, they require to be created in each cycle
                # NOTE: pd.Series needs to be casted to list
                stats = dict()
                for key, value in zip(indicator_names, analysis_output):
                    stats[key] = value
                # Assign "stats" to each "time_scale"
                analysis_obj[time_scale] = stats

            analysis_dict[pair] = analysis_obj

        for layer in self.analysis_layers[1:]:
            await self.analyze_layer(layer, analysis_dict)

        return analysis_dict


    async def analyze_layer(self, layer: Dict, analysis_dict: Dict[str, Dict]):

        for symbol, symbol_analysis in analysis_dict.items():
            for time_scale, analyzer_result in symbol_analysis.items():
                # Generate coroutines
                indicator_coroutines = []
                header = '_'
                indicator_method_names = list(map(lambda orig_string: header + orig_string, layer.keys()))
                indicator_names = list(layer.keys())

                for ind_method, ind_name in zip(indicator_method_names, indicator_names):
                    if hasattr(self, ind_method): 
                        indicator_coroutines.append(getattr(self, ind_method)(analyzer_result, **layer.get(ind_name,{})))
                    elif ind_method[:4] == '_cdl':
                        indicator_coroutines.append(getattr(self, '_cdl_handler')(analyzer_result, ind_method, **layer.get(ind_name,{})))

                    else: raise RuntimeError(f'Unknown Analyzer: "{ind_method}"')

                analysis_output = list(await asyncio.gather(*indicator_coroutines))
                for key, value in zip(indicator_names, analysis_output):
                    analysis_dict[symbol][time_scale][key] = value

        return analysis_dict
