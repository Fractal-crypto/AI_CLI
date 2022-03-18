from typing import Callable

from cachetools import TTLCache, cached


class LoggingMixin():

    show_output = True

    def __init__(self, logger, refresh_period: int = 3600):
        self.logger = logger
        self.refresh_period = refresh_period
        self._log_cache: TTLCache = TTLCache(maxsize=1024, ttl=self.refresh_period)

    def log_once(self, message: str, logmethod: Callable) -> None:
        @cached(cache=self._log_cache)
        def _log_once(message: str):
            logmethod(message)

        # Log as debug first
        self.logger.debug(message)
        # Call hidden function.
        if self.show_output:
            _log_once(message)
