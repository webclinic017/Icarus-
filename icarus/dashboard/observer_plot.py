from analyzers.support_resistance import SRCluster, deserialize_srcluster
from typing import List
from functools import wraps
from dashboard.analyzer_plot import support_resistance_plotter
from dashboard import analyzer_plot
from bokeh.palettes import Category10

def quote_asset(p_candlesticks, p_analyzer, source, observation, observer):

    p_analyzer.line([observation.index[0], observation.index[-1]], observation['total'].iloc[0], line_color='black')
    p_analyzer.line(observation.index, observation['total'], legend_label='total', line_color=Category10[3][0])
    #p_analyzer.line(observation.index, observation['free'], legend_label='free', line_color=Category10[3][1])
    #p_analyzer.line(observation.index, observation['in_trade'], legend_label='in_trade', line_color=Category10[3][2])


def adapt_cluster_indexes(x, y, enable_details=False) -> List[SRCluster]:
    all_cluster = []
    for observation in y:
        raw_clusters = [deserialize_srcluster(cluster_dict) for cluster_dict in observation['data']]
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
        def wrapper(p_candlesticks, p_analyzer, source, analysis, analyzer, **kwargs):
            enable_details = kwargs.get('enable_details', False)
            clusters = adapt_cluster_indexes(source.data['open_time'], analysis, enable_details)
            kwargs_sr_plotter = {'type': type, 'cmap': color_map, 'details': enable_details}
            support_resistance_plotter(p_candlesticks, source, clusters, **kwargs_sr_plotter)
            return func(p_candlesticks, p_analyzer, source, clusters, analyzer, **kwargs)
        return wrapper
    return decorator

@adapt_clusters_decorator('Support', analyzer_plot.color_map_support_basic)
def support_meanshift(p_candlesticks, p_analyzer, source, analysis, analyzer, **kwargs): return

@adapt_clusters_decorator('Resistance', analyzer_plot.color_map_resistance_basic)
def resistance_meanshift(p_candlesticks, p_analyzer, source, analysis, analyzer, **kwargs): return

@adapt_clusters_decorator('Support', analyzer_plot.color_map_support_basic)
def support_dbscan(p_candlesticks, p_analyzer, source, analysis, analyzer, **kwargs): return

@adapt_clusters_decorator('Resistance', analyzer_plot.color_map_resistance_basic)
def resistance_dbscan(p_candlesticks, p_analyzer, source, analysis, analyzer, **kwargs): return

@adapt_clusters_decorator('Support', analyzer_plot.color_map_support_basic)
def support_birch(p_candlesticks, p_analyzer, source, analysis, analyzer, **kwargs): return

@adapt_clusters_decorator('Resistance', analyzer_plot.color_map_resistance_basic)
def resistance_birch(p_candlesticks, p_analyzer, source, analysis, analyzer, **kwargs): return

@adapt_clusters_decorator('Support', analyzer_plot.color_map_support_basic)
def support_optics(p_candlesticks, p_analyzer, source, analysis, analyzer, **kwargs): return

@adapt_clusters_decorator('Resistance', analyzer_plot.color_map_resistance_basic)
def resistance_optics(p_candlesticks, p_analyzer, source, analysis, analyzer, **kwargs): return