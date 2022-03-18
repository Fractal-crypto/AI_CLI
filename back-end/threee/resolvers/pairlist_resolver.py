from pathlib import Path

from threee.plugins.pairlist.IPairList import IPairList
from threee.resolvers import IResolver


class PairListResolver(IResolver):
    object_type = IPairList
    object_type_str = "Pairlist"
    user_subdir = None
    initial_search_path = Path(__file__).parent.parent.joinpath('plugins/pairlist').resolve()

    @staticmethod
    def load_pairlist(pairlist_name: str, exchange, pairlistmanager,
                      config: dict, pairlistconfig: dict, pairlist_pos: int) -> IPairList:
        return PairListResolver.load_object(pairlist_name, config,
                                            kwargs={'exchange': exchange,
                                                    'pairlistmanager': pairlistmanager,
                                                    'config': config,
                                                    'pairlistconfig': pairlistconfig,
                                                    'pairlist_pos': pairlist_pos},
                                            )
