"""
Static Pair List provider

Provides pair white list as it configured in config
"""
import logging
from copy import deepcopy
from typing import Any, Dict, List

from threee.plugins.pairlist.IPairList import IPairList


class StaticPairList(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._allow_inactive = self._pairlistconfig.get('allow_inactive', False)

    @property
    def needstickers(self) -> bool:
        return False

    def short_desc(self) -> str:
        return f"{self.name}"

    def gen_pairlist(self, tickers: Dict) -> List[str]:
        if self._allow_inactive:
            return self.verify_whitelist(
                self._config['exchange']['pair_whitelist'], None, keep_invalid=True
            )
        else:
            return self._whitelist_for_active_markets(
                self.verify_whitelist(self._config['exchange']['pair_whitelist'], None))

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        pairlist_ = deepcopy(pairlist)
        for pair in self._config['exchange']['pair_whitelist']:
            if pair not in pairlist_:
                pairlist_.append(pair)
        return pairlist_
