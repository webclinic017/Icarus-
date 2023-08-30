from statistics import mean, stdev
import numpy as np
import pandas as pd
from objects import ECause, EState, Report, ReportMeta
from safe_operators import safe_divide, safe_substract
import copy
import asyncio
from dataclasses import asdict
import itertools
from report.report_writer import evaluate_filename

accuracy_conditions_for_ppc = {
    'downtrend': lambda a,count : (np.array(a) < -1 ).sum() / count * 100,
    'uptrend': lambda a,count : (np.array(a) > 1 ).sum() / count * 100,
    'ranging': lambda a,count : ((np.array(a) > -1) & (np.array(a) < 1)).sum() / count * 100,
}

async def ohlcv(indices, analysis):
    df = analysis[0][['open', 'high', 'low', 'close', 'volume']]
    df.set_index(np.array(df.index).astype('datetime64[ms]'), inplace=True)
    return df


async def perc_pos_change_raw(indices, analysis):
    filename = 'perc_pos_change_raw_{}_{}'.format(indices[0][0], indices[0][1])
    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=analysis[0].set_index(np.array(analysis[0].index).astype('datetime64[ms]')))


async def perc_pos_change_occurence(indices, analysis):
    filename = 'perc_pos_change_occurence_{}_{}'.format(indices[0][0], indices[0][1])
    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=pd.DataFrame([analysis[0]['pos_change'].value_counts(), analysis[0]['neg_change'].value_counts()]).T)


async def perc_pos_change_stats(indices, analysis):
    df = analysis[0]
    pos_change_thresholds = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

    #statistic_dict = {
    #    "count": len(df),
    #    "average_pos_change": round(df['pos_change'].mean(), 4 ),
    #    "average_neg_change": round(df['neg_change'].mean(), 4 )
    #}

    table = []
    for th in pos_change_thresholds:
        #statistic_dict[key] = round((df['pos_change'] > value).sum()/len(df), 2 )
        table.append([
            round((df['pos_change'] > th).sum()/len(df), 2 ),
            round((df['neg_change'] < -th).sum()/len(df), 2 )
        ])
    df_th = pd.DataFrame(table, columns=['pos_change','neg_change'], index=list(map(lambda x: str(x).replace('.','_'), pos_change_thresholds)))
    #statistic_dict['threshold_table'] = df_th.to_dict()

    filename = 'perc_pos_change_stats_{}_{}'.format(indices[0][0], indices[0][1])
    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=df_th.T)


async def correlation_matrix(indices, analysis):
    df = pd.DataFrame(analysis, index=[indice[0] for indice in indices]).T
    #logretdf = np.log(df.pct_change() + 1)
    pct_changedf = df.pct_change()

    report_meta = ReportMeta(
        title='correlation_matrix',
        filename='correlation_matrix'
        )
    return Report(meta=report_meta, data=pct_changedf.corr())

# *ppc: price percentage change
async def market_regime_ppc(index, detected_market_regimes):
    filename = 'ppc_' + evaluate_filename(index, special_char=False)

    tabular_dict = {}
    for regime_name, regime_instances in detected_market_regimes[0].items():
        perc_price_change_list = [instance.perc_price_change for instance in regime_instances]
        tabular_dict[regime_name] = perc_price_change_list
    
    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=tabular_dict)


async def market_regime_pvpc(index, detected_market_regimes):
    tabular_dict = {}
    for regime_name, regime_instances in detected_market_regimes[0].items():
        perc_val_price_change_list = [instance.perc_val_price_change for instance in regime_instances if instance.perc_val_price_change != None]
        tabular_dict[regime_name] = perc_val_price_change_list
    return tabular_dict


async def market_regime_table_stats(index, detected_market_regimes):
    filename = evaluate_filename(index, special_char=False)

    tabular_dict = {}
    for regime_name, regime_instances in detected_market_regimes[0].items():
        # "Exception has occurred: ValueError: not enough values to unpack..." error fixed
        if not regime_instances:
            continue
        perc_price_change_list, perc_val_price_change_list, duration_in_candle_list = list(map(np.array, zip(
            *[[instance.perc_price_change, instance.perc_val_price_change, instance.duration_in_candle] for instance in regime_instances])))
        regime_stats = {}
        regime_stats['Occurence'] = int(len(regime_instances))
        regime_stats['Average PPC'] = round(mean(perc_price_change_list),2)
        regime_stats['Average PVPC'] = round(perc_val_price_change_list[perc_val_price_change_list != None].mean(),2) # TODO: <string>:1: RuntimeWarning: Mean of empty slice.
        regime_stats['None PVPC'] = round(sum(x is None for x in perc_val_price_change_list)/len(regime_instances)*100,2)
        regime_stats['Average duration'] = int(mean(duration_in_candle_list))
        regime_stats['Coverage'] = round(sum(duration_in_candle_list) / len(index) * 100,2)
        regime_stats['PPC Accuracy'] = round(
            accuracy_conditions_for_ppc[regime_name](perc_price_change_list, len(regime_instances)),2)
        regime_stats['PVPC Accuracy'] = round(
            accuracy_conditions_for_ppc[regime_name](perc_val_price_change_list[perc_val_price_change_list != None], len(regime_instances)),2)

        tabular_dict[regime_name] = regime_stats
    
    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=tabular_dict)


async def perc_pos_change_stats_in_market_regime(index, analysis):
    filename = 'ppc_accuracy_' +evaluate_filename([index[0]], special_char=False)

    market_regimes = analysis[0]
    df_change = copy.deepcopy(analysis[1])

    coroutines = []
    results = []
    coroutines.append(perc_pos_change_stats(index, [df_change]))

    for regime_name, regime_instances in market_regimes.items():
        df_change[regime_name] = False
        for instance in regime_instances:
            df_change.loc[instance.start_ts:instance.end_ts, regime_name]=True
        coroutines.append(perc_pos_change_stats(index, [df_change[df_change[regime_name]]]))

    results = await asyncio.gather(*coroutines)
    result_dict = {key:regime_report.data for key, regime_report in zip(['all'] + list(market_regimes.keys()), results)}

    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=result_dict)


async def ppc_trigger_stats_in_market_regime(index, analysis):
    filename = 'ppc_trigger_accuracy_' +evaluate_filename([index[0]], special_char=False)

    market_regimes = analysis[0]
    df_change = copy.deepcopy(analysis[1])

    coroutines = []
    results = []
    coroutines.append(perc_pos_change_stats(index, [df_change]))

    for regime_name, regime_instances in market_regimes.items():
        df_change[regime_name] = False
        for instance in regime_instances:
            df_change.loc[instance.start_ts, regime_name]=True
        coroutines.append(perc_pos_change_stats(index, [df_change[df_change[regime_name]]]))

    results = await asyncio.gather(*coroutines)
    result_dict = {key:regime_report.data for key, regime_report in zip(['all'] + list(market_regimes.keys()), results)}

    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=result_dict)


async def supres_tables_per_metric(index, analysis_data):
    metrics = ['vertical_distribution_score', 'horizontal_distribution_score', 'distribution_score',
        'number_of_members', 'distribution_efficiency', 'number_of_retest', 'number_of_cluster']

    timeframes_x_algo = []
    for  clusters in analysis_data:
        timeframes_x_algo.append(pd.DataFrame([asdict(cluster) for cluster in clusters], columns=metrics))

    tuple_index = pd.MultiIndex.from_tuples(list(map(tuple, index)))
    df_timeframes_x_algo_mean = pd.DataFrame(index=tuple_index, columns=timeframes_x_algo[0].columns)

    for tuple_indice, df_tf_x_algo in zip(tuple_index,timeframes_x_algo):
        df_timeframes_x_algo_mean.loc[tuple_indice] = df_tf_x_algo.mean()
        df_timeframes_x_algo_mean.loc[tuple_indice, 'number_of_cluster'] = len(df_tf_x_algo)

    tables_per_metric = {}
    for metric in metrics:
        df_unstacked = df_timeframes_x_algo_mean[metric].unstack(level=2).astype(float)

        # Hack for retaining the index order after unstack operation
        level0 = df_timeframes_x_algo_mean[metric].index.get_level_values(0).unique()
        level1 = df_timeframes_x_algo_mean[metric].index.get_level_values(1).unique()
        tables_per_metric[metric] = df_unstacked.reindex(list(itertools.product(*[level0, level1])))

    return tables_per_metric


async def supres_tables_per_algo(index, analysis_data):
    metrics = ['vertical_distribution_score', 'horizontal_distribution_score', 'distribution_score',
        'number_of_members', 'distribution_efficiency', 'number_of_retest', 'number_of_cluster']

    timeframes_x_algo = []
    for  clusters in analysis_data:
        timeframes_x_algo.append(pd.DataFrame([asdict(cluster) for cluster in clusters], columns=metrics))

    tuple_index = pd.MultiIndex.from_tuples(list(map(tuple, index)))
    df_timeframes_x_algo_mean = pd.DataFrame(index=tuple_index, columns=timeframes_x_algo[0].columns)

    for tuple_indice, df_tf_x_algo in zip(tuple_index,timeframes_x_algo):
        df_timeframes_x_algo_mean.loc[tuple_indice] = df_tf_x_algo.mean()
        df_timeframes_x_algo_mean.loc[tuple_indice, 'number_of_cluster'] = len(df_tf_x_algo)

    algos = df_timeframes_x_algo_mean.index.get_level_values(2).unique()
    tables_per_algo = {algo:df_timeframes_x_algo_mean.xs(algo, level=2, drop_level=True) for algo in algos}

    return tables_per_algo


async def supres_tables_per_timeframe(index, analysis_data):
    metrics = ['vertical_distribution_score', 'horizontal_distribution_score', 'distribution_score',
        'number_of_members', 'distribution_efficiency', 'number_of_retest', 'number_of_cluster']

    timeframes_x_algo = []
    for  clusters in analysis_data:
        timeframes_x_algo.append(pd.DataFrame([asdict(cluster) for cluster in clusters], columns=metrics))

    tuple_index = pd.MultiIndex.from_tuples(list(map(tuple, index)))
    df_timeframes_x_algo_mean = pd.DataFrame(index=tuple_index, columns=timeframes_x_algo[0].columns)

    for tuple_indice, df_tf_x_algo in zip(tuple_index,timeframes_x_algo):
        df_timeframes_x_algo_mean.loc[tuple_indice] = df_tf_x_algo.mean()
        df_timeframes_x_algo_mean.loc[tuple_indice, 'number_of_cluster'] = len(df_tf_x_algo)

    timeframes = df_timeframes_x_algo_mean.index.get_level_values(1).unique()
    tables_per_algo = {tf:df_timeframes_x_algo_mean.xs(tf, level=1, drop_level=True) for tf in timeframes}

    return tables_per_algo


async def supres_distribution_per_metric(index, analysis_data):
    metrics = ['vertical_distribution_score', 'horizontal_distribution_score', 'distribution_score',
        'number_of_members', 'distribution_efficiency', 'number_of_retest']

    timeframes_x_algo = []
    for  clusters in analysis_data:
        timeframes_x_algo.append(pd.DataFrame([asdict(cluster) for cluster in clusters], columns=metrics))

    result_dict = {}
    for metric in metrics:
        metric_dict = {}
        for idx, tf_x_algo in zip(index, timeframes_x_algo):
            metric_dict[idx[-1]] = tf_x_algo[metric]
        result_dict[metric] = metric_dict 

    # result_dict: 
    # 1h timeframe
    # {
    #   number_of_retest: {sr_dbscan:number_of_retest, sr_birch:number_of_retest,...}, # Plot
    #   number_of_members: {sr_dbscan:number_of_members, sr_birch:number_of_members,...},
    #   ...
    # }
    # The results will go to box plots so for each metric
    return result_dict

async def dummy_reporter(index, analysis_data):
    return analysis_data

async def strategy_statistics(index, reporter_input):

    df = pd.DataFrame(reporter_input[0])

    if df.empty:
        return Report()

    stats = {}

    # Count
    stat_count = {}
    stat_count['live'] = (df['status'] != EState.CLOSED).sum()
    stat_count['closed'] = (df['status'] == EState.CLOSED).sum()

    count_cause = df['cause'].value_counts()
    for cause in ECause:
        stat_count[cause.value] = count_cause.get(cause,0)

    count_updated = df['is_updated'].value_counts()
    stat_count['not_updated'] = count_updated.get(False, 0)
    stat_count['updated'] = count_updated.get(True, 0)
    stat_count['win'] = (df['profit'] > 0).sum()
    stat_count['lose'] = (df['profit'] <= 0).sum()

    stat_absolute_profit = {
        'best': df['profit'].max(),
        'worst': df['profit'].min(),
        'total': df['profit'].sum(),
        'total_updated': df[df['is_updated']==True]['profit'].sum(),
        'total_not_updated': df[df['is_updated']==False]['profit'].sum(),
        'average': df['profit'].mean(),
        'average_updated': df[df['is_updated']==True]['profit'].mean(),
        'average_not_updated': df[df['is_updated']==False]['profit'].mean()
    }

    stat_percentage_profit = {
        'best': df['percentage_profit'].max(),
        'worst': df['percentage_profit'].min(),
        'total': df['percentage_profit'].sum(),
        'total_updated': df[df['is_updated']==True]['percentage_profit'].sum(),
        'total_not_updated': df[df['is_updated']==False]['percentage_profit'].sum(),
        'average': df['percentage_profit'].mean(),
        'average_updated': df[df['is_updated']==True]['percentage_profit'].mean(),
        'average_not_updated': df[df['is_updated']==False]['percentage_profit'].mean()
    }

    stat_price_change = {
        'best': df['price_change'].max(),
        'worst': df['price_change'].min(),
        'total': df['price_change'].sum(),
        'total_updated': df[df['is_updated']==True]['price_change'].sum(),
        'total_not_updated': df[df['is_updated']==False]['price_change'].sum(),
        'average': df['price_change'].mean(),
        'average_updated': df[df['is_updated']==True]['price_change'].mean(),
        'average_not_updated': df[df['is_updated']==False]['price_change'].mean()
    }

    day_in_ms = 1000*60*60*24
    hour_in_ms = 1000*60*60
    durations = df['duration']/hour_in_ms
    durations[df['is_updated']==True].mean(),
    stat_duration = {
        'max': durations.max(),
        'min': durations.min(),
        'total': durations.sum(),
        'average': durations.mean(),
        'average_not_updated': durations[df['is_updated']==False].mean(),
        'average_updated': durations[df['is_updated']==True].mean()
    }
    # TODO: Add duration for enter and exit orders separetaly

    stat_rates = {
        'win': (df['profit'] > 0).sum() / len(df['profit']),
        'lose': (df['profit'] <= 0).sum() / len(df['profit']),
        'enter': (df['cause'] != ECause.ENTER_EXP).sum() / len(df)
    }

    stat_risk = dict()
    if 'exit_type' in df.columns and any('oco' == df['exit_type']):
        df_oco = df[df['exit_type'] == 'oco']
        df_risk = pd.DataFrame(df_oco['risk_data'].to_list())
        df_r = pd.concat([df_oco,df_risk], axis=1)
        df_r.loc[df_r['cause'] == 'limit','r_value'] = (df_risk['target_price'] - df_risk['enter_price']) / (df_risk['enter_price'] - df_risk['stop_limit_price'])
        df_r.loc[df_r['cause'] == 'stop_limit','r_value'] = -1

        stat_risk['expectancy'] = df_r['r_value'].mean()

        if len(df_oco) < 100:
            sqn_coeff = len(df_oco)
        else:
            sqn_coeff = 100
        
        r_value_std = df_r['r_value'].std()
        if r_value_std != 0:
            stat_risk['SQN'] = stat_risk['expectancy']/df_r['r_value'].std() * np.sqrt(sqn_coeff)
        else:
            stat_risk['SQN'] = None
    
    stat_others = {
        'total_fee': df['fee'].sum()
    }

    # Combine Stats
    stats = {
        'strategy': df['strategy'][0],
        'count': stat_count,
        'absolute_profit': stat_absolute_profit,
        'percentage_profit': stat_percentage_profit,
        'price_change': stat_price_change,
        'duration':stat_duration,
        'rates': stat_rates,
        'risk': stat_risk,
        'others': stat_others
    }

    # Round all floats to 2
    for stat_key, stat in stats.items():
        if type(stat) != dict:
            continue
        
        for k,v in stat.items():
            if type(v) in [np.float64, float]:
                stat[k] = round(v,3)
            elif type(v) == np.int64:
                stat[k] = int(v)

    return Report(ReportMeta(title='strategy_{}'.format(df['strategy'][0])), data=stats)


async def balance_statistics(index, reporter_input):

    df = pd.DataFrame(reporter_input[0])

    if df.empty:
        return Report()
    
    mdd_percentage = (df['total'].max() - df['total'].min() ) / df['total'].max() * 100


    quote_asset_start = df['total'].iloc[0]
    quote_asset_end = df['total'].iloc[-1]
    
    stats = {
        'start':quote_asset_start,
        'end':quote_asset_end,
        'absolute_profit': safe_substract(quote_asset_end, quote_asset_start, quant='0.01')
    }
    stats['percentage_profit'] = safe_divide(stats['absolute_profit']*100, stats['start'], quant='0.01')
    stats['max_drawdown'] = round(mdd_percentage,2)

    return  Report(ReportMeta(title='balance_statistics'),data=stats)


async def trade_cause(index, reporter_input):

    df = pd.DataFrame(reporter_input[0])

    if df.empty:
        return Report()

    count_cause = df['cause'].value_counts()

    report_meta = ReportMeta(
        title='trade.cause: {}'.format(df['strategy'][0]),
        filename='trade_cause_{}'.format(df['strategy'][0])
        )
    return  Report(report_meta, data=count_cause.to_dict())


async def trade_perc_profit_duration_distribution(index, reporter_input):

    df = pd.DataFrame(reporter_input[0])

    if df.empty:
        return Report()

    df['duration'] = df['duration']/(60*60*1000)
    report_meta = ReportMeta(
        title='trade.result.percentage_profit: {}'.format(df['strategy'][0]),
        filename='trade_result_percentage_profit_{}'.format(df['strategy'][0])
        )
    return Report(report_meta, data=df[['duration', 'percentage_profit']])


async def trade_perc_profit(index, reporter_input):

    df = pd.DataFrame(reporter_input[0])

    if df.empty:
        return Report()

    df = df.set_index(df['decision_time'].astype('datetime64[ms]'))
    report_meta = ReportMeta(
        title='trade.result.percentage_profit: {}'.format(df['strategy'][0]),
        filename='trade_result_percentage_profit_{}'.format(df['strategy'][0])
        )
    return Report(report_meta, data=df[['percentage_profit']])


async def trade_profit_duration_distribution(index, reporter_input):

    df = pd.DataFrame(reporter_input[0])

    if df.empty:
        return Report()
    
    df['duration'] = df['duration']/(60*60*1000)
    report_meta = ReportMeta(
        title='trade.result.profit: {}'.format(df['strategy'][0]),
        filename='trade_result_profit_{}'.format(df['strategy'][0])
        )
    return Report(report_meta, data=df[['duration', 'profit']])


async def strategy_capitals(index, reporter_input):

    df_base = pd.DataFrame(reporter_input[0])
    df = pd.DataFrame(df_base['data'].to_list(), index=df_base['ts'].astype('datetime64[ms]'))
    #df['Total'] = df.sum(axis=1)

    report_meta = ReportMeta(
        title='Strategy Capitals',
        filename='strategy_capitals'
        )
    return Report(report_meta, data=df)


async def strategy_capital_statistics(index, reporter_input):

    capitals = [
        reporter_input[0][0]['first_capitals'],
        reporter_input[0][0]['last_capitals'],
    ]
    df = pd.DataFrame(capitals, index=['first','last']).T

    df['percentage_profit'] = ((df['last']-df['first']) / df['first'] * 100).round(2)

    report_meta = ReportMeta(
        title='Strategy Capital Statistics',
        filename='strategy_capital_statistics'
        )
    return Report(report_meta, data=df)


async def r_multiples(index, reporter_input):
    
    df = pd.DataFrame(reporter_input[0])

    if df.empty:
        return Report()

    df.loc[df['cause'] == 'limit','r_value'] = (df['target_price'] - df['enter_price']) / (df['enter_price'] - df['stop_limit_price'])
    df.loc[df['cause'] == 'stop_limit','r_value'] = -1

    hour_in_ms = 1000*60*60
    df['duration']  = df['duration'] / hour_in_ms

    report_meta = ReportMeta(
        title='R Multiple Distribution {}'.format(df['strategy'][0]),
        filename='r_multiple_distribution_{}'.format(df['strategy'][0]),
        )
    return Report(report_meta, data=df[['duration','r_value']])


async def symbol_price_change(index, reporter_input):
    price_change = {}
    for idx, rep in zip(index, reporter_input):
        if idx[0] in price_change:
            continue
        price_change[idx[0]] = round((rep[-1]-rep[0])/rep[0]*100, 2)

    return Report(ReportMeta(title='Symbol Price Change'), data=pd.DataFrame([price_change]).T)


async def pattern_counter(index, reporter_input):
    filename = evaluate_filename(index, special_char=False)
    counts = {}
    for idx, rep in zip(index, reporter_input):
        bearish_count = sum(map(lambda x : x < 0, rep))
        bullish_count = sum(map(lambda x : x > 0, rep))
        counts[idx[2]] = {
            'total':bearish_count+bullish_count, 
            'bearish':bearish_count, 
            'bullish':bullish_count, 
            'occ_ratio': round((bearish_count+bullish_count)/len(rep),2)}
    count_df = pd.DataFrame(counts).T
    count_df_sorted = count_df.sort_values(by = 'total', ascending = False)
    return Report(ReportMeta(title=filename,filename=filename), data=count_df_sorted)


async def market_direction_correlation(index, reporter_input):
    filename = evaluate_filename(index, special_char=False)
    df = reporter_input[0]
    corr_matrix = df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)

    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=corr_matrix)


async def market_regime_duration_up(index, reporter_input):
    filename = evaluate_filename(index, special_char=False)
    filename += '_up'
    tabular_stats = {}
    for classifier, regimes in reporter_input[0].items():
        tabular_stats[classifier.replace('market_regime_', '')] = [trend.duration_in_candle for trend in regimes['uptrend']]

    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=tabular_stats)

async def market_regime_duration_down(index, reporter_input):
    filename = evaluate_filename(index, special_char=False)
    filename += '_down'
    tabular_stats = {}
    for classifier, regimes in reporter_input[0].items():
        tabular_stats[classifier.replace('market_regime_', '')] = [trend.duration_in_candle for trend in regimes['downtrend']]

    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=tabular_stats)


async def market_regime_duration_up_stat(index, reporter_input):
    filename = evaluate_filename(index, special_char=False)
    filename += '_up'
    tabular_stats = {}
    for classifier, regimes in reporter_input[0].items():
        downtrend_dist = [trend.duration_in_candle for trend in regimes['uptrend']]
        tabular_stats[classifier.replace('market_regime_', '')] = {
            'q1': np.percentile(downtrend_dist, 25),
            'median': np.median(downtrend_dist),
            'q3': np.percentile(downtrend_dist, 75)
            }

    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=tabular_stats)


async def market_regime_duration_down_stat(index, reporter_input):
    filename = evaluate_filename(index, special_char=False)
    filename += '_down'
    tabular_stats = {}
    for classifier, regimes in reporter_input[0].items():
        downtrend_dist = [trend.duration_in_candle for trend in regimes['downtrend']]
        tabular_stats[classifier.replace('market_regime_', '')] = {
            'q1': np.percentile(downtrend_dist, 25),
            'median': np.median(downtrend_dist),
            'q3': np.percentile(downtrend_dist, 75)
            }

    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    return Report(meta=report_meta, data=tabular_stats)


def enum_to_value(enum_element):
    return int(enum_element.value) if enum_element is not None else None


async def dataset_market_direction(index, reporter_input):

    filename = evaluate_filename(index, special_char=False)
    df = reporter_input[0]
    df = df.applymap(enum_to_value)

    report_meta = ReportMeta(
        title=filename,
        filename=filename
        )
    
    return Report(meta=report_meta, data=df)