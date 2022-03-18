
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn


class LowProfitPairs(IProtection):

    has_global_stop: bool = False
    has_local_stop: bool = True

    def __init__(self, config: Dict[str, Any], protection_config: Dict[str, Any]) -> None:
        super().__init__(config, protection_config)

        self._trade_limit = protection_config.get('trade_limit', 1)
        self._required_profit = protection_config.get('required_profit', 0.0)

    def short_desc(self) -> str:
        return (f"{self.name} - Low Profit Protection, locks pairs with "
                f"profit < {self._required_profit} within {self.lookback_period_str}.")

    def _reason(self, profit: float) -> str:
        return (f'{profit} < {self._required_profit} in {self.lookback_period_str}, '
                f'locking for {self.stop_duration_str}.')

    def _low_profit(self, date_now: datetime, pair: str) -> ProtectionReturn:
        look_back_until = date_now - timedelta(minutes=self._lookback_period)

        trades = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)

        if len(trades) < self._trade_limit:
            return False, None, None

        profit = sum(trade.close_profit for trade in trades if trade.close_profit)
        if profit < self._required_profit:
            self.log_once(
                f"Trading for {pair} stopped due to {profit:.2f} < {self._required_profit} "
                f"within {self._lookback_period} minutes.", logger.info)
            until = self.calculate_lock_end(trades, self._stop_duration)

            return True, until, self._reason(profit)

        return False, None, None

    def global_stop(self, date_now: datetime) -> ProtectionReturn:

        return False, None, None

    def stop_per_pair(self, pair: str, date_now: datetime) -> ProtectionReturn:

        return self._low_profit(date_now, pair=pair)
