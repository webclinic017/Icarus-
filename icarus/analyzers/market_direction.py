from enum import Enum

class Direction(Enum):
    UP = "up"
    DOWN = "down"
    SIDE = "side"


class MarketDirection():

    async def _direction_macd(self, candlesticks, **kwargs) -> Direction:

        macd_dict = await self._macd(candlesticks, **kwargs)

        if macd_dict['macdhist'][-1] > 0:
            direction = Direction.UP
        elif macd_dict['macdhist'][-1] < 0:
            direction = Direction.DOWN
        else:
            direction = Direction.SIDE
        
        return direction

    async def _direction_aroonosc(self, candlesticks, **kwargs) -> Direction:

        aroonosc = await self._aroonosc(candlesticks, **kwargs)

        if aroonosc[-1] > 0:
            direction = Direction.UP
        elif aroonosc[-1] < 0:
            direction = Direction.DOWN
        else:
            direction = Direction.SIDE
        
        return direction