import logging
from functools import partial
from typing import Dict, List

from cachetools import TTLCache, cached

from threee.constants import ListPairsWithTimeframes
from threee.exceptions import OperationalException
from threee.mixins import LoggingMixin
from threee.plugins.pairlist.IPairList import IPairList
from threee.plugins.pairlist.pairlist_helpers import expand_pairlist
from threee.resolvers import PairListResolver



class PairListManager(LoggingMixin):

    def __init__(self, exchange, config: dict) -> None:
        self._exchange = exchange
        self._config = config
        self._whitelist = self._config['exchange'].get('pair_whitelist')
        self._blacklist = self._config['exchange'].get('pair_blacklist', [])
        self._pairlist_handlers: List[IPairList] = []
        self._tickers_needed = False
        for pairlist_handler_config in self._config.get('pairlists', None):
            pairlist_handler = PairListResolver.load_pairlist(
                pairlist_handler_config['method'],
                exchange=exchange,
                pairlistmanager=self,
                config=config,
                pairlistconfig=pairlist_handler_config,
                pairlist_pos=len(self._pairlist_handlers)
            )
            self._tickers_needed |= pairlist_handler.needstickers
            self._pairlist_handlers.append(pairlist_handler)

        if not self._pairlist_handlers:
            raise OperationalException("No Pairlist Handlers defined")

        refresh_period = config.get('pairlist_refresh_period', 3600)
        LoggingMixin.__init__(self, None, refresh_period)

    @property
    def whitelist(self) -> List[str]:
        return self._whitelist

    @property
    def blacklist(self) -> List[str]:
        return self._blacklist

    @property
    def expanded_blacklist(self) -> List[str]:
        return expand_pairlist(self._blacklist, self._exchange.get_markets().keys())

    @property
    def name_list(self) -> List[str]:
        return [p.name for p in self._pairlist_handlers]

    def short_desc(self) -> List[Dict]:
        return [{p.name: p.short_desc()} for p in self._pairlist_handlers]

    @cached(TTLCache(maxsize=1, ttl=1800))
    def _get_cached_tickers(self):
        return self._exchange.get_tickers()

    def refresh_pairlist(self) -> None:
        tickers: Dict = {}
        if self._tickers_needed:
            tickers = self._get_cached_tickers()

        pairlist = self._pairlist_handlers[0].gen_pairlist(tickers)

        for pairlist_handler in self._pairlist_handlers[1:]:
            pairlist = pairlist_handler.filter_pairlist(pairlist, tickers)

        pairlist = self.verify_blacklist(pairlist, None)

        self._whitelist = pairlist

    def verify_blacklist(self, pairlist: List[str], logmethod) -> List[str]:
        try:
            blacklist = self.expanded_blacklist
        except ValueError as err:
            return []
        log_once = partial(self.log_once, logmethod=logmethod)
        for pair in pairlist.copy():
            if pair in blacklist:

                pairlist.remove(pair)
        return pairlist

    def verify_whitelist(self, pairlist: List[str], logmethod,
                         keep_invalid: bool = False) -> List[str]:
        try:

            whitelist = expand_pairlist(pairlist, self._exchange.get_markets().keys(), keep_invalid)
        except ValueError as err:

            return []
        return whitelist

    def create_pair_list(self, pairs: List[str], timeframe: str = None) -> ListPairsWithTimeframes:
        """
        Create list of pair tuples with (pair, timeframe)
        """
        return [(pair, timeframe or self._config['timeframe']) for pair in pairs]
