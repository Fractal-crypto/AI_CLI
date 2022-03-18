import logging

import threee.exchange as exchanges
from threee.exchange import MAP_EXCHANGE_CHILDCLASS, Exchange
from threee.resolvers import IResolver


class ExchangeResolver(IResolver):
    object_type = Exchange

    @staticmethod
    def load_exchange(exchange_name: str, config: dict, validate: bool = True) -> Exchange:
        """
        사용자 지정 교환 클래스를 로드
        """
        exchange_name = MAP_EXCHANGE_CHILDCLASS.get(exchange_name, exchange_name)
        exchange_name = exchange_name.title()
        exchange = None
        try:
            exchange = ExchangeResolver._load_exchange(exchange_name,
                                                       kwargs={'config': config,
                                                               'validate': validate})
        except ImportError:
            pass
        if not exchange:
            exchange = Exchange(config, validate=validate)
        return exchange

    @staticmethod
    def _load_exchange(exchange_name: str, kwargs: dict) -> Exchange:

        try:
            ex_class = getattr(exchanges, exchange_name)

            exchange = ex_class(**kwargs)
            if exchange:
                return exchange
        except AttributeError:
            pass

        
