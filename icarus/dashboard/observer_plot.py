import finplot as fplt
from analyzers.support_resistance import SRCluster, serialize_srcluster
from typing import List
from utils import minute_to_time_scale
from visualization import indicator_plot
from functools import wraps
from dashboard.analyzer_plot import support_resistance_plotter
from dashboard import analyzer_plot

def quote_asset(p_candlesticks, p_analyzer, source, observation, observer):
    
    fplt.plot(y['total'], width=3, ax=axes['ax'], legend='Total')
    fplt.plot(y['free'], width=2, ax=axes['ax'], legend='Free')
    fplt.plot(y['in_trade'], width=2, ax=axes['ax'], legend='In Trade')
    fplt.add_line((y['total'].index[0], y['total'].iloc[0]),
        (y['total'].index[-1], y['total'].iloc[0]), color='#000000', interactive=False)


def text(x, y, axes):
    fplt.plot(x, y=[1]*len(x), ax=axes['ax_bot'])
    for index, row in y.iterrows():
        fplt.add_text((index, 0.5), str(row[0]), color='#000000',anchor=(0,0), ax=axes['ax_bot'])
    pass

def adapt_cluster_indexes(x, y, enable_details=False) -> List[SRCluster]:
    all_cluster = []
    for observation in y:
        raw_clusters = [serialize_srcluster(cluster_dict) for cluster_dict in observation['data']]
        candle_time_diff_sec = int((x[1]-x[0])/1000)
        observation_time = observation['ts']

        # NOTE: Assuming the same timeframe !
        end_candlestick_idx = (observation_time - x[1]/1000)/candle_time_diff_sec
        idx_offset = int(end_candlestick_idx - (raw_clusters[0].chunk_end_index+1))

        for srcluster in raw_clusters:
            srcluster.chunk_end_index += idx_offset
            if enable_details:
                srcluster.validation_index += idx_offset
                srcluster.chunk_start_index += idx_offset
            else:
                srcluster.validation_index = srcluster.chunk_end_index
                srcluster.chunk_start_index += srcluster.chunk_end_index - 5
        
        all_cluster += raw_clusters
    return all_cluster


def adapt_clusters_decorator(type, color_map):
    def decorator(func):
        @wraps(func)
        def wrapper(p_candlesticks, p_analyzer, source, analysis, analyzer):
            clusters = adapt_cluster_indexes(source.data['open_time'], analysis, False)
            kwargs = {'type': type, 'cmap': color_map, 'details': False}
            support_resistance_plotter(p_candlesticks, source, clusters, **kwargs)
            return func(p_candlesticks, p_analyzer, source, clusters, analyzer)
        return wrapper
    return decorator

@adapt_clusters_decorator('Support', analyzer_plot.color_map_support_basic)
def support_meanshift(p_candlesticks, p_analyzer, source, analysis, analyzer): return

@adapt_clusters_decorator('Resistance', analyzer_plot.color_map_resistance_basic)
def resistance_meanshift(p_candlesticks, p_analyzer, source, analysis, analyzer): return

@adapt_clusters_decorator('Support', analyzer_plot.color_map_support_basic)
def support_dbscan(p_candlesticks, p_analyzer, source, analysis, analyzer): return

@adapt_clusters_decorator('Resistance', analyzer_plot.color_map_resistance_basic)
def resistance_dbscan(p_candlesticks, p_analyzer, source, analysis, analyzer): return

@adapt_clusters_decorator('Support', analyzer_plot.color_map_support_basic)
def support_birch(p_candlesticks, p_analyzer, source, analysis, analyzer): return

@adapt_clusters_decorator('Resistance', analyzer_plot.color_map_resistance_basic)
def resistance_birch(p_candlesticks, p_analyzer, source, analysis, analyzer): return

@adapt_clusters_decorator('Support', analyzer_plot.color_map_support_basic)
def support_optics(p_candlesticks, p_analyzer, source, analysis, analyzer): return

@adapt_clusters_decorator('Resistance', analyzer_plot.color_map_resistance_basic)
def resistance_optics(p_candlesticks, p_analyzer, source, analysis, analyzer): return