import numpy as np
import tensorflow as tf
import joblib
from analyzers.market_classification import enum_to_value, array_to_enum, Direction

class MachineLearning():

    async def _inference(self, candlesticks, **kwargs):
        return None
    
    async def _good_entry(self, candlesticks, **kwargs):
        ppc = await self._percentage_possible_change(candlesticks, **kwargs)

        pos_up_th = 0.01
        neg_bot_th = -0.005

        ppc['good_entry'] = (pos_up_th > ppc['pos_change']) & (neg_bot_th > ppc['neg_change'])

        return ppc['good_entry']
    

    async def _market_direction_lstm(self, analysis, **kwargs):

        df = analysis[kwargs['indicators'][0]]
        df = df.applymap(enum_to_value)
        df = df.dropna().astype(int)

        model = tf.keras.models.load_model(kwargs['model_path'])

        
        #input_data = df.iloc[:, :-1].values
        input_data = df.values

        num_of_step = kwargs['window_size']+1
        num_of_features = input_data.shape[1] // num_of_step

        input_data = input_data.reshape(input_data.shape[0], num_of_step, num_of_features)

        predictions = model.predict(input_data)
        predictions_binary = (predictions > 0.5).astype(int)

        classification = array_to_enum(predictions_binary, Direction)

        # Do padding
        target_length = len(analysis['candlesticks'])
        padding_length = target_length - len(classification)
        classification = np.pad(classification, (padding_length, 0), 'constant', constant_values=None)

        return classification


    async def _market_direction_logisticregression(self, analysis, **kwargs):

        df = analysis[kwargs['indicators'][0]]
        df = df.applymap(enum_to_value)
        df = df.dropna().astype(int)

        model = joblib.load(kwargs['model_path'])
        
        #input_data = df.iloc[:, :-1].values
        input_data = df.values
        predictions = model.predict(input_data)
        classification = array_to_enum(predictions, Direction)

        # Do padding
        target_length = len(analysis['candlesticks'])
        padding_length = target_length - len(classification)
        classification = np.pad(classification, (padding_length, 0), 'constant', constant_values=None)

        return classification
