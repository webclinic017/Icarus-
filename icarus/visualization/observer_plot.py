import finplot as fplt
from analyzers.support_resistance import SRCluster, serialize_srcluster
from typing import List
from utils import minute_to_time_scale
from visualization import indicator_plot
from functools import wraps

def quote_asset2(dashboard_data, ax):
    fplt.plot(dashboard_data['quote_asset']['total'], width=3, ax=ax, legend='Total')
    fplt.plot(dashboard_data['quote_asset']['free'], width=2, ax=ax, legend='Free')
    fplt.plot(dashboard_data['quote_asset']['in_trade'], width=2, ax=ax, legend='In Trade')
    fplt.add_line((dashboard_data['quote_asset']['total'].index[0], dashboard_data['quote_asset']['total'].iloc[0]),
        (dashboard_data['quote_asset']['total'].index[-1], dashboard_data['quote_asset']['total'].iloc[0]), color='#000000', interactive=False)


def quote_asset(x, y, axes):
    axes['ax'].reset()
    axes['axo'].reset()
    axes['ax_bot'].reset()
    axes['axo_bot'].reset()
    
    disable_ax_bot(axes)
    fplt.plot(y['total'], width=3, ax=axes['ax'], legend='Total')
    fplt.plot(y['free'], width=2, ax=axes['ax'], legend='Free')
    fplt.plot(y['in_trade'], width=2, ax=axes['ax'], legend='In Trade')
    fplt.add_line((y['total'].index[0], y['total'].iloc[0]),
        (y['total'].index[-1], y['total'].iloc[0]), color='#000000', interactive=False)


def disable_ax_bot(axes):
    axes['ax'].set_visible(xaxis=True)
    axes['ax_bot'].hide()

def quote_asset_leak(dashboard_data, ax):
    fplt.plot(dashboard_data['quote_asset_leak']['binary'], width=3, ax=ax, legend='binary')


def enable_ax_bot(axes, **kwargs):
    fplt._ax_reset(axes['ax_bot'])

    axes['ax'].set_visible(xaxis=False)
    axes['ax_bot'].show()

    #if kwargs.get('reset', True): fplt._ax_reset(axes['ax_bot'])
    if y_range := kwargs.get('y_range', None): fplt.set_y_range(y_range[0], y_range[1], ax=axes['ax_bot'])
    if band := kwargs.get('band', None): fplt.add_band(band[0], band[1], color='#6335', ax=axes['ax_bot'])



def text(x, y, axes):
    fplt.plot(x, y=[1]*len(x), ax=axes['ax_bot'])
    for index, row in y.iterrows():
        fplt.add_text((index, 0.5), str(row[0]), color='#000000',anchor=(0,0), ax=axes['ax_bot'])
    pass

def adapt_cluster_indexes(x, y) -> List[SRCluster]:
    all_cluster = []
    for observation in y:
        raw_clusters = [serialize_srcluster(cluster_dict) for cluster_dict in observation['data']]
        candle_time_diff_sec = int((x[1]-x[0])/1000)
        observation_time = observation['ts']

        # NOTE: Assuming the same timeframe !
        end_candlestick_idx = (observation_time - x[1]/1000)/candle_time_diff_sec
        idx_offset = int(end_candlestick_idx - raw_clusters[0].chunk_end_index)

        for srcluster in raw_clusters:
            srcluster.chunk_end_index += idx_offset
            # NOTE:To visualize events, uncomment the sections below
            #srcluster.validation_index += idx_offset #= srcluster.chunk_end_index
            srcluster.validation_index = srcluster.chunk_end_index
            #srcluster.chunk_start_index += idx_offset #srcluster.chunk_end_index - 5
            srcluster.chunk_start_index += srcluster.chunk_end_index - 5
        
        all_cluster += raw_clusters
    return all_cluster


def adapt_clusters_decorator(type, color_map):
    def decorator(func):
        @wraps(func)
        def wrapper(x, y, axes):
            clusters = adapt_cluster_indexes(x, y)
            indicator_plot.disable_ax_bot(axes)
            indicator_plot.support_resistance_handler(x, clusters, axes, **{'type': type, 'cmap': color_map, 'details': True})
            return func(x, y, axes)
        return wrapper
    return decorator

@adapt_clusters_decorator('Support', indicator_plot.color_map_support_basic)
def support_meanshift(x, y, axes): return

@adapt_clusters_decorator('Resistance', indicator_plot.color_map_resistance_basic)
def resistance_meanshift(x, y, axes): return

@adapt_clusters_decorator('Support', indicator_plot.color_map_support_basic)
def support_dbscan(x, y, axes): return

@adapt_clusters_decorator('Resistance', indicator_plot.color_map_resistance_basic)
def resistance_dbscan(x, y, axes): return

@adapt_clusters_decorator('Support', indicator_plot.color_map_support_basic)
def support_birch(x, y, axes): return

@adapt_clusters_decorator('Resistance', indicator_plot.color_map_resistance_basic)
def resistance_birch(x, y, axes): return

@adapt_clusters_decorator('Support', indicator_plot.color_map_support_basic)
def support_optics(x, y, axes): return

@adapt_clusters_decorator('Resistance', indicator_plot.color_map_resistance_basic)
def resistance_optics(x, y, axes): return