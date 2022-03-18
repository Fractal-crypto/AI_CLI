
import logging
from datetime import datetime, timedelta

from freqtrade.persistence import Trade
from freqtrade.plugins.protections import IProtection, ProtectionReturn

class CooldownPeriod(IProtection):

    has_global_stop: bool = False
    has_local_stop: bool = True

    def _reason(self) -> str:
        pass

    def short_desc(self) -> str:(f"{self.name} - Cooldown period of {self.stop_duration_str}.")

    def _cooldown_period(self, pair: str, date_now: datetime, ) -> ProtectionReturn:
        look_back_until = date_now - timedelta(minutes=self._stop_duration)
        trades = Trade.get_trades_proxy(pair=pair, is_open=False, close_date=look_back_until)
        if trades:.
            trade = sorted(trades, key=lambda t: t.close_date)[-1]
            # self.log_once(f"Cooldown for {pair} for {self.stop_duration_str}.", logger.info)
            until = self.calculate_lock_end([trade], self._stop_duration)

            return True, until, self._reason()

        return False, None, None

    def global_stop(self, date_now: datetime) -> ProtectionReturn:
        return False, None, None

    def stop_per_pair(self, pair: str, date_now: datetime) -> ProtectionReturn:
        return self._cooldown_period(pair, date_now)
