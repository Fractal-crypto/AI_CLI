import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from threee.persistence import PairLocks
from threee.persistence.models import PairLock
from threee.plugins.protections import IProtection
from threee.resolvers import ProtectionResolver


class ProtectionManager():

    def __init__(self, config: Dict, protections: List) -> None:
        self._config = config

        self._protection_handlers: List[IProtection] = []
        for protection_handler_config in protections:
            protection_handler = ProtectionResolver.load_protection(
                protection_handler_config['method'],
                config=config,
                protection_config=protection_handler_config,
            )
            self._protection_handlers.append(protection_handler)


    @property
    def name_list(self) -> List[str]:
        return [p.name for p in self._protection_handlers]

    def short_desc(self) -> List[Dict]:
        return [{p.name: p.short_desc()} for p in self._protection_handlers]

    def global_stop(self, now: Optional[datetime] = None) -> Optional[PairLock]:
        if not now:
            now = datetime.now(timezone.utc)
        result = None
        for protection_handler in self._protection_handlers:
            if protection_handler.has_global_stop:
                lock, until, reason = protection_handler.global_stop(now)

                if lock and until:
                    if not PairLocks.is_global_lock(until):
                        result = PairLocks.lock_pair('*', until, reason, now=now)
        return result

    def stop_per_pair(self, pair, now: Optional[datetime] = None) -> Optional[PairLock]:
        if not now:
            now = datetime.now(timezone.utc)
        result = None
        for protection_handler in self._protection_handlers:
            if protection_handler.has_local_stop:
                lock, until, reason = protection_handler.stop_per_pair(pair, now)
                if lock and until:
                    if not PairLocks.is_pair_locked(pair, until):
                        result = PairLocks.lock_pair(pair, until, reason, now=now)
        return result
