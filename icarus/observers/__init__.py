from objects import Observation, Trade
from typing import Dict, List
from utils import eval_total_capital_in_lto
import pandas as pd


def quote_asset(obs_config: Dict, ikarus_time_sec: int, config: Dict, df_balance: pd.DataFrame, live_trade_list: List[Trade], new_trade_list: List[Trade]) -> Observation:
    observation_obj = {}
    observation_obj['free'] = df_balance.loc[config['broker']['quote_currency'],'free']
    observation_obj['in_trade'] = eval_total_capital_in_lto(live_trade_list+new_trade_list)
    observation_obj['total'] = observation_obj['free'] + observation_obj['in_trade']
    return Observation(obs_config['type'], ikarus_time_sec, observation_obj)


def analyzer(obs_config: Dict, ikarus_time_sec: int, live_trade_list: List[Trade], analysis: List[Trade]) -> Observation:
    observation_obj = {}
    return Observation(obs_config['type'], ikarus_time_sec, observation_obj)


def balance(obs_config: Dict, ikarus_time_sec: int, df_balance: pd.DataFrame) -> Observation:
    observation_obj = list(df_balance.reset_index(level=0).T.to_dict().values())
    return Observation(obs_config['type'], ikarus_time_sec, observation_obj)


def strategy_capitals(obs_config: Dict, ikarus_time_sec: int, strategy_res_allocator) -> Observation:
    return Observation(obs_config['type'], ikarus_time_sec, strategy_res_allocator.strategy_capitals)