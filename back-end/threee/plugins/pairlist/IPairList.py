import logging
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from typing import Any, Dict, List

from threee.exceptions import OperationalException
from threee.exchange import Exchange, market_is_active
from threee.mixins import LoggingMixin


class IPairList(LoggingMixin, ABC):

    def __init__(self, exchange: Exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:

        self._enabled = True

        self._exchange: Exchange = exchange
        self._pairlistmanager = pairlistmanager
        self._config = config
        self._pairlistconfig = pairlistconfig
        self._pairlist_pos = pairlist_pos
        self.refresh_period = self._pairlistconfig.get('refresh_period', 1800)
        self._last_refresh = 0
        LoggingMixin.__init__(self, None, self.refresh_period)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractproperty
    def needstickers(self) -> bool:
        pass

    @abstractmethod
    def short_desc(self) -> str:
        pass

    def _validate_pair(self, pair: str, ticker: Dict[str, Any]) -> bool:
        pass

    def gen_pairlist(self, tickers: Dict) -> List[str]:
        pass

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:

        if self._enabled:

            for p in deepcopy(pairlist):

                if not self._validate_pair(p, tickers[p] if p in tickers else {}):
                    pairlist.remove(p)

        return pairlist

    def verify_blacklist(self, pairlist: List[str], logmethod) -> List[str]:

        return self._pairlistmanager.verify_blacklist(pairlist, logmethod)

    def verify_whitelist(self, pairlist: List[str], logmethod,
                         keep_invalid: bool = False) -> List[str]:

        return self._pairlistmanager.verify_whitelist(pairlist, logmethod, keep_invalid)

    def _whitelist_for_active_markets(self, pairlist: List[str]) -> List[str]:

        markets = self._exchange.markets
        if not markets:
            pass

        sanitized_whitelist: List[str] = []
        for pair in pairlist:

            if pair not in sanitized_whitelist:
                sanitized_whitelist.append(pair)

        return sanitized_whitelist
