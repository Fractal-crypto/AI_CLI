
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from threee.exchange import timeframe_to_minutes
from threee.misc import plural
from threee.mixins import LoggingMixin
from threee.persistence import LocalTrade


ProtectionReturn = Tuple[bool, Optional[datetime], Optional[str]]


class IProtection(LoggingMixin, ABC):


    has_global_stop: bool = False
    has_local_stop: bool = False

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        self._config = config
        self._protection_config = protection_config
        self._stop_duration_candles: Optional[int] = None
        self._lookback_period_candles: Optional[int] = None

        tf_in_min = timeframe_to_minutes(config['timeframe'])
        if 'stop_duration_candles' in protection_config:
            self._stop_duration_candles = int(protection_config.get('stop_duration_candles', 1))
            self._stop_duration = (tf_in_min * self._stop_duration_candles)
        else:
            self._stop_duration_candles = None
            self._stop_duration = protection_config.get('stop_duration', 60)
        if 'lookback_period_candles' in protection_config:
            self._lookback_period_candles = int(protection_config.get('lookback_period_candles', 1))
            self._lookback_period = tf_in_min * self._lookback_period_candles
        else:
            self._lookback_period_candles = None
            self._lookback_period = int(protection_config.get('lookback_period', 60))

        LoggingMixin.__init__(self, None)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def stop_duration_str(self) -> str:
        if self._stop_duration_candles:
            return (f"{self._stop_duration_candles} "
                    f"{plural(self._stop_duration_candles, 'candle', 'candles')}")
        else:
            return (f"{self._stop_duration} "
                    f"{plural(self._stop_duration, 'minute', 'minutes')}")

    @property
    def lookback_period_str(self) -> str:
        if self._lookback_period_candles:
            return (f"{self._lookback_period_candles} "
                    f"{plural(self._lookback_period_candles, 'candle', 'candles')}")
        else:
            return (f"{self._lookback_period} "
                    f"{plural(self._lookback_period, 'minute', 'minutes')}")

    @abstractmethod
    def short_desc(self) -> str:
        pass
    @abstractmethod
    def global_stop(self, date_now: datetime) -> ProtectionReturn:
        pass

    @abstractmethod
    def stop_per_pair(self, pair: str, date_now: datetime) -> ProtectionReturn:
        pass

    @staticmethod
    def calculate_lock_end(trades: List[LocalTrade], stop_minutes: int) -> datetime:

        max_date: datetime = max([trade.close_date for trade in trades if trade.close_date])

        if max_date.tzinfo is None:
            max_date = max_date.replace(tzinfo=timezone.utc)

        until = max_date + timedelta(minutes=stop_minutes)

        return until
