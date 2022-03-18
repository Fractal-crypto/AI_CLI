from pathlib import Path
from typing import Dict

from threee.plugins.protections import IProtection
from threee.resolvers import IResolver


class ProtectionResolver(IResolver):
    object_type = IProtection
    object_type_str = "Protection"
    user_subdir = None
    initial_search_path = Path(__file__).parent.parent.joinpath('plugins/protections').resolve()

    @staticmethod
    def load_protection(protection_name: str, config: Dict, protection_config: Dict) -> IProtection:
        return ProtectionResolver.load_object(protection_name, config,
                                              kwargs={'config': config,
                                                      'protection_config': protection_config,
                                                      },
                                              )
