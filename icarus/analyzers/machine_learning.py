import pandas as pd


class MachineLearning():

    async def _inference(self, candlesticks, **kwargs):
        return None
    
    async def _good_entry(self, candlesticks, **kwargs):
        ppc = await self._percentage_possible_change(candlesticks, **kwargs)

        pos_up_th = 0.01
        neg_bot_th = -0.005

        ppc['good_entry'] = (pos_up_th > ppc['pos_change']) & (neg_bot_th > ppc['neg_change'])

        return ppc['good_entry']
