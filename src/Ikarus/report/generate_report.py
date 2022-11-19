import json
import sys
import os
import asyncio
from binance import AsyncClient
from .. import broker
import datetime
from itertools import chain
import itertools
from ..analyzers import Analyzer
from .. import mongo_utils
from . import report_tools
from .report_writer import ReportWriter, GridSearchWriter
import ast
from copy import deepcopy


def write_to_config_file(config_dict, filename="generated_config.json"):
    config_file_path = os.path.dirname(str(sys.argv[1])) + '/' + filename
    f = open(config_file_path,'w')
    json.dump(config_dict, f,  indent=4)
    f.close()
    return config_file_path


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def get_reporter_name(indice):
    if type(indice[0]) == str:
        return indice[0]
    elif type(indice[0][0]) == str:
        return indice[0][0]
    else:
        return None


def generate_queries(reporter_config):
    grid_configs = list(itertools.product(*reporter_config['parameters'].values()))

    queries = []
    for grid_config in grid_configs:
        query = deepcopy(reporter_config['query_template'])
        for key_idx, key in enumerate(reporter_config['parameters'].keys()):
            query = ast.literal_eval(str(query).replace(key, grid_config[key_idx]))
        queries.append(query)
    return queries


def generate_indices(reporter_config):
    grid_configs = list(itertools.product(*reporter_config.get('parameters',{}).values()))

    queries = []
    for grid_config in grid_configs:
        query = deepcopy(reporter_config['indice_template'])
        for key_idx, key in enumerate(reporter_config['parameters'].keys()):
            query = ast.literal_eval(str(query).replace(key, grid_config[key_idx]))
        queries.append(query)
    return queries

def grid_search():
    reporter_instances = []
    all_parameters = list(config['grid_search_reporters']['grid'].keys())
    all_grid_values = list(config['grid_search_reporters']['grid'].values())

    # First loop is for generating the reporter items but not filling its query templates
    for reporter_config in config['grid_search_reporters']['reporters']:

        # Get only the related parameters
        grid_values, parameters = [], []
        for param, grid_value in zip(all_parameters, all_grid_values):
            if param in str(reporter_config):
                grid_values.append(grid_value)
                parameters.append(param)

        # Create grid configs
        grid_configs = list(itertools.product(*grid_values))

        # Create replace rules
        for grid_config in grid_configs:
            replace_rule = {}
            for param, rep_value in zip(parameters, grid_config):
                replace_rule[param] = rep_value

            replaced_text = replace_all(str(reporter_config), replace_rule)
            reporter_instance = ast.literal_eval(replaced_text)
            reporter_instances.append(reporter_instance)

    # Second loop is for generating the queries for reporters
    config_report = []
    for reporter_instance in reporter_instances:
        reporter = {}
        if 'query_template' in reporter_instance:
            reporter['source'] = reporter_instance['source']
            reporter['reporter'] = reporter_instance['reporter']
            reporter['collection'] = reporter_instance['collection']
            reporter['queries'] = generate_queries(reporter_instance)
            reporter['writers'] = reporter_instance['writers']
            config_report.append(reporter)
        
        elif 'indice_template' in reporter_instance:
            reporter['source'] = reporter_instance['source']
            reporter['reporter'] = reporter_instance['reporter']
            reporter['indices'] = generate_indices(reporter_instance)
            reporter['writers'] = reporter_instance['writers']
            config_report.append(reporter)

    return config_report


async def main():

    client = await AsyncClient.create(api_key=cred_info['Binance']['Test']['PUBLIC-KEY'],
                                    api_secret=cred_info['Binance']['Test']['SECRET-KEY'])
    bwrapper = broker.TestBinanceWrapper(client, config)
    mongo_client = mongo_utils.MongoClient(db='reports', **config['mongodb'])
    start_time = datetime.datetime.strptime(config['backtest']['start_time'], "%Y-%m-%d %H:%M:%S")
    start_timestamp = int(datetime.datetime.timestamp(start_time))*1000
    end_time = datetime.datetime.strptime(config['backtest']['end_time'], "%Y-%m-%d %H:%M:%S")
    end_timestamp = int(datetime.datetime.timestamp(end_time))*1000

    # Create pools for pair-scales
    time_scale_pool = []
    pair_pool = []
    for strategy in config['strategy'].values():
        time_scale_pool.append(strategy['time_scales'])
        pair_pool.append(strategy['pairs'])

    time_scale_pool = list(set(chain(*time_scale_pool)))
    pair_pool = list(chain(*pair_pool))

    meta_data_pool = list(itertools.product(time_scale_pool, pair_pool))

    data_dict = await bwrapper.download_all_data(meta_data_pool, start_timestamp, end_timestamp)
    analyzer = Analyzer(config)
    analysis_dict = await analyzer.analyze(data_dict)

    # Indice format: ()
    report_tool_coroutines = []
    indices = []

    if not config['report']:
        generated_report_config = grid_search()
        config['report'] = generated_report_config
        write_to_config_file(config)

    for report_config in config['report']:
        #timeframe, symbol, analyzer
        source = report_config.get('source', 'analyzer')

        if source == 'database': 
            # Use queries
            report_tool_coroutines.append(mongo_utils.do_aggregate_multi_query(mongo_client, report_config.get('collection', report_config), report_config['queries']))

        elif source == 'analyzer':
            # Use indices
            handler = getattr(report_tools, report_config['reporter'])
            analysis_data = [analysis_dict[reporter_indice[0]][reporter_indice[1]][reporter_indice[2]] for reporter_indice in report_config['indices']]
            report_tool_coroutines.append(handler(report_config['indices'], analysis_data)) # Use indices as the index

        elif source == 'candlesticks':
            # Use indices
            handler = getattr(report_tools, report_config['reporter'])
            candlestick_data = [data_dict[reporter_indice[0]][reporter_indice[1]] for reporter_indice in report_config['indices']]
            report_tool_coroutines.append(handler(report_config['indices'], candlestick_data)) # Use indices as the index

    # Get the statistics
    report_tool_results = list(await asyncio.gather(*report_tool_coroutines))

    # Write the statistics
    
    report_folder = os.path.dirname(str(sys.argv[1])) + '/' + config.get('report_folder_name', 'reports')
    report_writer = ReportWriter(report_folder, mongo_client)
    async_writers = []
    for report_config, report_dict in zip(config['report'], report_tool_results):
        #reporter, timeframe, symbol, analyzer = indice

        for writer_type in report_config.get('writers', []): #shitcode
            if hasattr(report_writer, writer_type):
                kwargs = {
                    'start_time': config['backtest']['start_time'],
                    'end_time': config['backtest']['end_time'],
                    'reporter': report_config['reporter']
                    #'pair': indice[2],
                    #'timeframe': indice[1]
                }

                if attr := getattr(report_writer, writer_type)(report_config.get('indices',[]),report_dict,**kwargs):
                    async_writers.append(attr)

    await asyncio.gather(*async_writers)
    report_writer.add_images()

    report_writer.md_file.create_md_file()
    pass

if __name__ == '__main__':
    print(sys.argv)
    f = open(str(sys.argv[1]),'r')
    config = json.load(f)
    
    with open(config['credential_file'], 'r') as cred_file:
        cred_info = json.load(cred_file)
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

