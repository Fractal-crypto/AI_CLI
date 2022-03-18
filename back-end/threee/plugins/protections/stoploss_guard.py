
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from freqtrade.enums import SellType
from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


class StoplossGuard(IProtection):

    has_global_stop: bool = True
    has_local_stop: bool = True

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get('trade_limit', 10)
        self._disable_global_stop = protection_config.get('only_per_pair', False)

    def short_desc(self) -> str:
        return (f"{self.name} - Frequent Stoploss Guard, {self._trade_limit} stoplosses "
                f"within {self.lookback_period_str}.")

    def _reason(self) -> str:
        return (f'{self._trade_limit} stoplosses in {self._lookback_period} min, '
                f'locking for {self._stop_duration} min.')

    def _stoploss_guard(self, date_now: datetime, pair: str = None) -> ProtectionReturn:
        look_back_until = date_now - timedelta(minutes=self._lookback_period)

        trades1 = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)
        trades = [trade for trade in trades1 if (str(trade.sell_reason) in (
            SellType.TRAILING_STOP_LOSS.value, SellType.STOP_LOSS.value,
            SellType.STOPLOSS_ON_EXCHANGE.value)
            and trade.close_profit and trade.close_profit < 0)]

        if len(trades) < self._trade_limit:
            return False, None, None

        self.log_once(f"Trading stopped due to {self._trade_limit} "
                      f"stoplosses within {self._lookback_period} minutes.", logger.info)
        until = self.calculate_lock_end(trades, self._stop_duration)
        return True, until, self._reason()

    def global_stop(self, date_now: datetime) -> ProtectionReturn:

        if self._disable_global_stop:
            return False, None, None
        return self._stoploss_guard(date_now, None)

    def stop_per_pair(self, pair: str, date_now: datetime) -> ProtectionReturn:
    
        return self._stoploss_guard(date_now, pair)
