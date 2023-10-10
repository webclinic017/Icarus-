from objects import Observation, Trade
from typing import Dict, List
from utils import eval_total_capital_in_lto
import pandas as pd
import logging
from dataclasses import asdict, is_dataclass
from analyzers.support_resistance import SRCluster

logger = logging.getLogger('app')

def _serialize_analysis(analysis):
    if type(analysis) == list:

        if len(analysis) == 0:
            return analysis
        
        if is_dataclass(analysis[0]):
            return [asdict(obj) for obj in analysis]
    
    elif type(analysis) == dict:
        pass
    return analysis

def quote_asset(obs_config: Dict, ikarus_time_sec: int, config: Dict, df_balance: pd.DataFrame, live_trade_list: List[Trade], new_trade_list: List[Trade]) -> Observation:
    observation_obj = {}
    observation_obj['free'] = df_balance.loc[config['broker']['quote_currency'],'free']
    observation_obj['in_trade'] = eval_total_capital_in_lto(live_trade_list+new_trade_list)
    observation_obj['total'] = observation_obj['free'] + observation_obj['in_trade']
    return Observation(obs_config['type'], ikarus_time_sec, observation_obj)


def analyzer(obs_config: Dict, ikarus_time_sec: int, analysis: List[Trade]) -> Observation:
    kwargs = obs_config['kwargs']
    data = analysis[kwargs['symbol']][kwargs['timeframe']][kwargs['analyzer']]
    x = _serialize_analysis(data)
    return Observation(obs_config['type'], ikarus_time_sec, _serialize_analysis(data), dtype=kwargs['dtype'])


def balance(obs_config: Dict, ikarus_time_sec: int, df_balance: pd.DataFrame) -> Observation:
    observation_obj = list(df_balance.reset_index(level=0).T.to_dict().values())
    return Observation(obs_config['type'], ikarus_time_sec, observation_obj)


def strategy_capitals(obs_config: Dict, ikarus_time_sec: int, strategy_res_allocator) -> Observation:
    return Observation(obs_config['type'], ikarus_time_sec, strategy_res_allocator.strategy_capitals)