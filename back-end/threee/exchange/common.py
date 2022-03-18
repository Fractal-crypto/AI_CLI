import asyncio
import logging
import time
from functools import wraps

from threee.exceptions import DDosProtection, RetryableOrderError, TemporaryError
from threee.mixins import LoggingMixin


__logging_mixin = None

def _get_logging_mixin():
    global __logging_mixin
    return __logging_mixin

#api 대시도 횟수
API_RETRY_COUNT = 4
API_FETCH_ORDER_RETRY_COUNT = 5

BAD_EXCHANGES = {
}

MAP_EXCHANGE_CHILDCLASS = {
    'binanceus': 'binance',
    'binanceje': 'binance'
}


EXCHANGE_HAS_REQUIRED = [
    'fetchOrder',
    'cancelOrder',
    'createOrder',
    'fetchBalance',
    'loadMarkets',
    'fetchOHLCV',
]
#download
EXCHANGE_HAS_OPTIONAL = [
    'fetchMyTrades',
    'fetchOrderBook', 'fetchL2OrderBook', 'fetchTicker',
    'fetchTickers',
    'fetchTrades',
]


def remove_credentials(config) -> None:
    """
    테스트를 위해서 딕셔너리 수정 키값
    """
    if config.get('dry_run', False):
        config['exchange']['key'] = ''
        config['exchange']['secret'] = ''
        config['exchange']['password'] = ''
        config['exchange']['uid'] = ''


def calculate_backoff(retrycount, max_retries):
    """
    데이터 재전송 전 대기시간 계산
    """
    return (max_retries - retrycount) ** 2 + 1


def retrier_async(f):
    async def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        kucoin = args[0].name == "Kucoin"
        try:
            return await f(*args, **kwargs)
        except TemporaryError as ex:
             None
    return wrapper


def retrier(_func=None, retries=API_RETRY_COUNT):
    #디코더
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            count = kwargs.pop('count', retries)
            try:
                return f(*args, **kwargs)
            except (TemporaryError, RetryableOrderError) as ex:
                    count -= 1
                    kwargs.update({'count': count})

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
