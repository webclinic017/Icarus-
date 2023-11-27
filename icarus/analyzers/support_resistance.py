from dataclasses import asdict, dataclass, field
import math
from sklearn.cluster import KMeans, DBSCAN, MeanShift, OPTICS, Birch
import numpy as np
from utils import minute_to_time_scale
from statistics import mean
from enum import Enum
from typing import List, Dict
from itertools import groupby
from operator import itemgetter
import pandas as pd

class SREventType(str, Enum):
    BREAK = 'break'
    BOUNCE = 'bounce'
    IN_ZONE = 'in_zone'
    PASS_HORIZONTAL = 'pass_horizontal'
    PASS_VERTICAL = 'pass_vertical'


@dataclass
class SREvent():
    type: SREventType
    start_index: int
    end_index: int
    price_min: float
    price_mean: float
    price_max: float
    before: int
    after: int


def sr_eval_price_position(low: float, high: float, price_min: float, price_max: float) -> int:
    if low > price_max:
        return 1
    elif high < price_min:
        return -1
    else:
        return 0


def sr_evaluate_event_type(meaningful_move_th: int,  length: int, before_pos: int, after_pos: int, is_last_candle: bool) -> SREventType:
    if is_last_candle:
        return SREventType.IN_ZONE

    is_directions_same = after_pos == before_pos

    if length <= meaningful_move_th:
        if is_directions_same: return SREventType.PASS_VERTICAL
        else: return SREventType.BREAK
    else:
        return SREventType.PASS_HORIZONTAL


@dataclass
class SRConfig():
    kwargs: dict                        # Mandatory
    type: str
    source: str = ''
    min_members: int = None
    frame_length: int = None
    step_length: int = None

    # DBSCAN, OPTICS, BIRCH
    eps_coeff: float = 0.005            # TODO: Optimize this epsilon value based on volatility or sth else

    # MeanShift
    bandwidth_coeff: float = 0.01

    # OPTICS
    cluster_method: str = 'xi'

    # KMeans
    n_cluster: int = None

    def __post_init__(self):
        self.source = self.kwargs.get('source','')
        self.eps = self.kwargs.get('eps',0.005)
        self.min_members = self.kwargs.get('min_members', None)
        self.eps_coeff = self.kwargs.get('eps_coeff', 0.005)

    def parse_chunks_params(self, diff_in_minute, time_scales_config):
        if "step_length" in self.kwargs.keys() or "step_to_frame_ratio" in self.kwargs.keys():
            self.frame_length = time_scales_config[minute_to_time_scale(diff_in_minute)]

            if "step_length" in self.kwargs.keys():
                self.step_length = self.kwargs.get('step_length')
            elif "step_to_frame_ratio" in self.kwargs.keys():
                self.step_length = int(self.frame_length * self.kwargs.get('step_to_frame_ratio'))


@dataclass
class SRCluster():
    
    type: str
    centroids: list = field(default_factory=list)
    validation_index: int = 0
    # NOTE: relative_validation_index parameter can also be added by thinking that it may give a clue about a normal
    # validation time so that anomalies might be foreseen. However it also has other dependencies such as: 
    # - the chunk start index 
    # - chunk length
    # Thus it requires a bit more effort to figure out if it worths to effort to implement and investigate
    min_cluster_members: int = 0
    horizontal_distribution_score: float = 0.0          # Higher the better
    vertical_distribution_score: float = 0.0            # Lower the better
    chunk_start_index: int = None
    chunk_end_index: int = None
    distribution_score: float = None                    # Higher the better
    number_of_retest: int = None                        # Higher the better
    number_of_members: int = None                       # Higher the better
    distribution_efficiency: int = None                 # Higher the better
    price_mean: float = 0.0
    price_min: float = 0.0
    price_max: float = 0.0

    # SREvent counts
    count_bounce: int = 0
    count_break: int = 0
    count_pass_vertical: int = 0
    count_pass_horizontal: int = 0
    count_in_zone: int = 0

    events: List[SREvent] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.distribution_score = round(self.vertical_distribution_score and self.horizontal_distribution_score / self.vertical_distribution_score or 0, 2)
        self.number_of_members = len(self.centroids)
        self.number_of_retest = self.number_of_members-self.min_cluster_members
        self.distribution_efficiency = round(self.distribution_score * self.number_of_members,2)
        self.price_mean = mean(self.centroids)
        self.price_min = min(self.centroids)
        self.price_max = max(self.centroids)

    def __lt__(self, other):
        return self.price_mean < other.price_mean
    
    def relative_position(self, compare_price: float) -> int:
        if compare_price > self.price_max:
            # Cluster is below the price
            return -1
        elif compare_price < self.price_min:
            # Cluster is above the price
            return 1
        else:
            # Price is on the cluster
            return 0

def count_srevent(cluster: SRCluster, srevent_type: SREventType) -> int:
    counter = 0
    for event in cluster.events:
        if event.type == srevent_type:
            counter += 1
    return counter


def deserialize_srevents(raw_srevents: List) -> SREvent:
    return [SREvent(**raw_srevent) for raw_srevent in raw_srevents]


def deserialize_srcluster(raw_srcluster: Dict) -> SRCluster:
    srcluster = SRCluster(**raw_srcluster)
    if len(srcluster.events):
        srcluster.events = deserialize_srevents(srcluster.events)
    return srcluster


class SupportResistance():

    def eval_min_cluster_members(chunk_size):
        return max(round(chunk_size/100),3)

    async def eval_sup_res_cluster_horizontal_score(indices, num_of_candle):
        # NOTE: By dividing the indice diferences to len(dbscan_bear), we manage to represent the distance without the dependecy of number of candles:
        if len(indices) <= 1:
            return 0

        weights = list(range(1,len(indices)))
        return np.round(np.average(np.diff(indices) / num_of_candle, weights=weights),4)


    async def eval_sup_res_cluster_vertical_score(centroids, chart_price_range):
        if len(centroids) <= 1:
            return 0

        cluster_price_range = max(centroids) - min(centroids)
        cluster_price_range_perc = cluster_price_range / chart_price_range

        # The returned value should not be zero since it will be denominator in distribution_score calculation
        return max(np.round(cluster_price_range_perc/len(centroids), 6), 0.000001)


    async def eval_sup_res_clusters(algorithm, sr_config: SRConfig, candles, meta_chunks, candlesticks):
        sr_levels = []

        for meta_chunk in meta_chunks:
            chunk = candles[meta_chunk[0] : meta_chunk[1]]

            if sr_config.min_members == None:
                min_cluster_members = SupportResistance.eval_min_cluster_members(chunk.size)
            else:
                min_cluster_members = sr_config.min_members

            # If the attribute min_samples exist, we have to overwrite it
            if hasattr(algorithm, 'min_samples'):
                algorithm.set_params(min_samples=min_cluster_members) 

            candlesticks_chunk = candlesticks[meta_chunk[0] : meta_chunk[1]]
            chart_price_range = candlesticks_chunk['high'].max() - candlesticks_chunk['low'].min()

            if hasattr(algorithm, 'bandwidth'):
                bandwidth = float(chart_price_range * sr_config.bandwidth_coeff)
                algorithm.set_params(bandwidth=bandwidth)
            elif hasattr(algorithm, 'eps'):
                eps = float(chart_price_range * sr_config.eps_coeff)
                algorithm.set_params(eps=eps)
            elif hasattr(algorithm, 'threshold'):
                eps = float(chart_price_range * sr_config.eps_coeff)
                algorithm.set_params(threshold=eps)

            cluster_predictions = algorithm.fit_predict(chunk)
            cls_tokens = np.unique(cluster_predictions)

            for token in cls_tokens:
                # NOTE: Ignore outliers
                if token == -1:
                    continue

                indices = np.where(cluster_predictions == token)[0]
                centroids = chunk[indices].reshape(1,-1)[0].tolist()

                # NOTE: Ignore the cluster if all of the members are 0, or the not enough cluster members
                if not any(centroids) or len(centroids)<min_cluster_members:
                    continue

                srcluster = SRCluster(
                    sr_config.type,
                    centroids,
                    int(meta_chunk[0] + indices[min_cluster_members-1]),
                    min_cluster_members,
                    await SupportResistance.eval_sup_res_cluster_horizontal_score(indices, len(cluster_predictions)),
                    await SupportResistance.eval_sup_res_cluster_vertical_score(centroids, chart_price_range),
                    meta_chunk[0],
                    meta_chunk[1]-1     # NOTE: End index is not lenght but index thus 
                )
                sr_levels.append(srcluster)
        
        sr_levels.sort()
        return sr_levels

    async def create_meta_chunks(data_points, frame_length, step_length):
        meta_chunks = []
        if frame_length and data_points.size > frame_length:

            # Number of chunks that will be in full length: frame_length
            filled_chunk_num = math.ceil((data_points.size - frame_length)/step_length)

            # List of meta_chunks with the length of frame_length
            for i in range(filled_chunk_num):
                chunk_start_idx = i*step_length
                chunk_end_idx = chunk_start_idx+frame_length
                meta_chunks.append((chunk_start_idx,chunk_end_idx))
            
            # Check if we already hit the last candle. If not add the residual candles
            if meta_chunks[-1][1] != (data_points.size-1):
                meta_chunks.append((filled_chunk_num*step_length, data_points.size))

        else:
            meta_chunks.append((0,data_points.size))
        return meta_chunks


    async def _support_birch(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'support')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)

        bullish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        eps = float(chart_price_range * sr_config.eps_coeff)
        birch = Birch(branching_factor=15, n_clusters = None, threshold=eps)

        meta_chunks = await SupportResistance.create_meta_chunks(bullish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(birch, sr_config, bullish_frac, meta_chunks, candlesticks)
        return sr_clusters


    async def _resistance_birch(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'resistance')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)

        bearish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        eps = float(chart_price_range * sr_config.eps_coeff)
        birch = Birch(branching_factor=15, n_clusters = None, threshold=eps)
        # TODO: Add birch configs to sr_config

        meta_chunks = await SupportResistance.create_meta_chunks(bearish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(birch, sr_config, bearish_frac, meta_chunks, candlesticks)
        return sr_clusters


    async def _support_optics(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'support')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)  

        bullish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        eps = float(chart_price_range * sr_config.eps_coeff)
        optics = OPTICS(eps=eps, cluster_method=sr_config.cluster_method)

        meta_chunks = await SupportResistance.create_meta_chunks(bullish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(optics, sr_config, bullish_frac, meta_chunks, candlesticks)
        return sr_clusters


    async def _resistance_optics(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'resistance')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)

        bearish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        eps = float(chart_price_range * sr_config.eps_coeff) 
        optics = OPTICS(eps=eps, cluster_method=sr_config.cluster_method)

        meta_chunks = await SupportResistance.create_meta_chunks(bearish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(optics, sr_config, bearish_frac, meta_chunks, candlesticks)
        return sr_clusters

    async def _support_dbscan(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'support')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)

        bullish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        eps = float(chart_price_range * sr_config.eps_coeff)
        dbscan = DBSCAN(eps=eps)

        meta_chunks = await SupportResistance.create_meta_chunks(bullish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(dbscan, sr_config, bullish_frac, meta_chunks, candlesticks)
        return sr_clusters


    async def _resistance_dbscan(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'resistance')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)    

        bearish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        eps = float(chart_price_range * sr_config.eps_coeff)
        dbscan = DBSCAN(eps=eps) # NOTE: min_sample is set inside of the eval_sup_res_clusters method

        meta_chunks = await SupportResistance.create_meta_chunks(bearish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(dbscan, sr_config, bearish_frac, meta_chunks, candlesticks)
        return sr_clusters


    async def _support_meanshift(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'support')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)   

        bullish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        bandwidth = float(chart_price_range * sr_config.bandwidth_coeff)
        meanshift = MeanShift(bandwidth=bandwidth) 
        
        # TODO: Specifying bandwith halps a bit. I dont know why the estimation did not worked or how it is calculated
        #       Things to improve:
        #       - Min number of members can be added as post filter (seems like does not working well)

        meta_chunks = await SupportResistance.create_meta_chunks(bullish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(meanshift, sr_config, bullish_frac, meta_chunks, candlesticks)
        return sr_clusters


    async def _resistance_meanshift(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'resistance')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)   

        bearish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        bandwidth = float(chart_price_range * sr_config.bandwidth_coeff)
        meanshift = MeanShift(bandwidth=bandwidth) # TODO use bandwidth

        meta_chunks = await SupportResistance.create_meta_chunks(bearish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(meanshift, sr_config, bearish_frac, meta_chunks, candlesticks)
        return sr_clusters

    async def _support_kmeans(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'support')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)

        bullish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        kmeans = KMeans(
            n_clusters=sr_config.n_cluster, init='random',
            n_init=13, max_iter=300, 
            tol=1e-04, random_state=0
        )

        meta_chunks = await SupportResistance.create_meta_chunks(bullish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(kmeans, sr_config, bullish_frac, meta_chunks, candlesticks)
        return sr_clusters


    async def _resistance_kmeans(self, analysis, **kwargs):
        candlesticks = analysis['candlesticks']
        sr_config = SRConfig(kwargs, 'resistance')
        sr_config.parse_chunks_params(int((candlesticks.index[1]-candlesticks.index[0])/60000), self.time_scales_config)

        bearish_frac = np.nan_to_num(analysis[sr_config.source]).reshape(-1,1)
        chart_price_range = candlesticks['high'].max() - candlesticks['low'].min()
        kmeans = KMeans(
            n_clusters=sr_config.n_cluster, init='random',
            n_init=13, max_iter=300, 
            tol=1e-04, random_state=0
        )

        meta_chunks = await SupportResistance.create_meta_chunks(bearish_frac, sr_config.frame_length, sr_config.step_length)
        sr_clusters = await SupportResistance.eval_sup_res_clusters(kmeans, sr_config, bearish_frac, meta_chunks, candlesticks)
        return sr_clusters


    # _sr_<algorithm>
    async def _sr_dbscan(self, analysis, **kwargs):
        return {
            'support': await self._support_dbscan(analysis, **kwargs.get('support',{})),
            'resistance': await self._resistance_dbscan(analysis, **kwargs.get('resistance',{}))
        }


    async def _sr_kmeans(self, analysis, **kwargs):
        return {
            'support': await self._support_kmeans(analysis, **kwargs.get('support',{})),
            'resistance': await self._resistance_kmeans(analysis, **kwargs.get('resistance',{}))
        }


    async def _sr_birch(self, analysis, **kwargs):
        return {
            'support': await self._support_birch(analysis, **kwargs.get('support',{})),
            'resistance': await self._resistance_birch(analysis, **kwargs.get('resistance',{}))
        }


    async def _sr_optics(self, analysis, **kwargs):
        return {
            'support': await self._support_optics(analysis, **kwargs.get('support',{})),
            'resistance': await self._resistance_optics(analysis, **kwargs.get('resistance',{}))
        }


    async def _sr_meanshift(self, analysis, **kwargs):
        return {
            'support': await self._support_meanshift(analysis, **kwargs.get('support',{})),
            'resistance': await self._resistance_meanshift(analysis, **kwargs.get('resistance',{}))
        }


    async def _sr_events(self, analysis: Dict, **kwargs):
        sr_analyzers = kwargs.get('analyzers')
        sequence_th = kwargs.get('sequence_th')

        for sr in sr_analyzers:
            bounce_events = {} # Dict[ (): List[SREvent] ]

            # NOTE: SRCluster order is from lowest to highest price_mean for each cluster in analysis[sr]
            for cluster in analysis[sr]:
                price_min = cluster.price_min
                price_max = cluster.price_max

                chunk_candlesticks = analysis['candlesticks'].iloc[cluster.chunk_start_index : cluster.chunk_end_index+1] # NOTE: Give +1 offset since chunk_end_index is index not length
                sr_price_interactions = np.array([sr_eval_price_position(candle['low'], candle['high'], price_min, price_max) for _, candle in chunk_candlesticks.iterrows()])
                intersect = np.where(sr_price_interactions == 0)[0]
                chunk_length = len(chunk_candlesticks.index)

                for k, g in groupby(enumerate(intersect), lambda ix: ix[0] - ix[1]):
                    seq_idx = list(map(itemgetter(1), g))

                    before_position = int(sr_price_interactions[seq_idx[0]-1])

                    is_last_candle = False
                    if seq_idx[-1] + 1 >= chunk_length:
                        after_position = None
                        is_last_candle = True
                    else:
                        after_position = int(sr_price_interactions[seq_idx[-1]+1])
                    
                    sr_event_type = sr_evaluate_event_type(sequence_th, len(seq_idx), before_position, after_position, is_last_candle)
                    

                    sr_event = SREvent(
                        type=sr_event_type,
                        start_index=int(seq_idx[0]),
                        end_index=int(seq_idx[-1]),
                        price_min=cluster.price_min,
                        price_mean=cluster.price_mean,
                        price_max=cluster.price_max,
                        before=before_position,
                        after=after_position
                    )

                    # Save references for each SREventType.PASS_VERTICAL events to evaluate BOUNCE events from the big picture
                    if sr_event_type == SREventType.PASS_VERTICAL:
                        new_bounce_entry = (chunk_candlesticks.index[seq_idx[0]], len(seq_idx), before_position)
                        if new_bounce_entry not in bounce_events:
                            bounce_events[new_bounce_entry] = []
                        bounce_events[new_bounce_entry].append(sr_event)

                    cluster.events.append(sr_event)
            
            for key in bounce_events.keys():
                # NOTE: Check before_position to evaluate which tip point to label as SREventType.BOUNCE
                if key[2] == -1:
                    # Make the bottom most one (-1) SREventType.BOUNCE
                    bounce_events[key][-1].type = SREventType.BOUNCE
                elif key[2] == 1:
                    # Make the top most one (0) SREventType.BOUNCE
                    bounce_events[key][0].type = SREventType.BOUNCE

        for sr in sr_analyzers:
            for cluster in analysis[sr]:
                cluster.count_bounce = count_srevent(cluster, SREventType.BOUNCE)
                cluster.count_break = count_srevent(cluster, SREventType.BREAK)
                cluster.count_pass_horizontal = count_srevent(cluster, SREventType.PASS_HORIZONTAL)
                cluster.count_pass_vertical = count_srevent(cluster, SREventType.PASS_VERTICAL)
                cluster.count_in_zone = count_srevent(cluster, SREventType.IN_ZONE)
        return 


    async def _sr_event_filter(self, analysis: Dict, **kwargs):
        sr_analyzers = kwargs.get('analyzers')
        filters = kwargs.get('filters')

        for sr in sr_analyzers:
            if len(analysis[sr]) == 0:
                continue
            analysis[sr] = filter_by(analysis[sr], filters)

        return


def filter_by(clusters, filter_dict):
    if len(clusters) == 0:
        return []
    
    df = pd.DataFrame(clusters)
    for filter_field, filter_min_max in filter_dict.items():
        df = df.query(f"{filter_min_max[0]} <= {filter_field} <= {filter_min_max[1]}")

    return [deserialize_srcluster(cluster_dict) for cluster_dict in df.to_dict(orient='records')]


def multi_filter_by(filter_dict, clusters_bundle):
    if type(clusters_bundle[0]) == dict:
        for index, observation in enumerate(clusters_bundle):
            raw_clusters = [deserialize_srcluster(cluster_dict) for cluster_dict in observation['data']]
            clusters_bundle[index]['data'] = filter_by(raw_clusters, filter_dict)
    else:
        clusters_bundle = filter_by(clusters_bundle, filter_dict)

    return clusters_bundle
