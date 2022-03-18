# pragma pylint: disable=W0603
""" Wallet """

import logging
from copy import deepcopy
from typing import Any, Dict, NamedTuple, Optional

import arrow

from threee.constants import UNLIMITED_STAKE_AMOUNT
from threee.enums import RunMode
from threee.exceptions import DependencyException
from threee.exchange import Exchange
from threee.persistence import LocalTrade, Trade


class Wallet(NamedTuple):
    currency: str
    free: float = 0
    used: float = 0
    total: float = 0


class Wallets:

    def __init__(self, config: dict, exchange: Exchange, log: bool = True) -> None:
        self._config = config
        self._log = log
        self._exchange = exchange
        self._wallets: Dict[str, Wallet] = {}
        self.start_cap = config['dry_run_wallet']
        self._last_wallet_refresh = 0
        self.update()

    def get_free(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        if balance and balance.free:
            return balance.free
        else:
            return 0

    def get_used(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        if balance and balance.used:
            return balance.used
        else:
            return 0

    def get_total(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        if balance and balance.total:
            return balance.total
        else:
            return 0

    def _update_dry(self) -> None:
        _wallets = {}
        open_trades = Trade.get_trades_proxy(is_open=True)
        if self._log:
            tot_profit = Trade.get_total_closed_profit()
        else:
            tot_profit = LocalTrade.total_profit
        tot_in_trades = sum(trade.stake_amount for trade in open_trades)

        current_stake = self.start_cap + tot_profit - tot_in_trades
        _wallets[self._config['stake_currency']] = Wallet(
            self._config['stake_currency'],
            current_stake,
            0,
            current_stake
        )

        for trade in open_trades:
            curr = self._exchange.get_pair_base_currency(trade.pair)
            _wallets[curr] = Wallet(
                curr,
                trade.amount,
                0,
                trade.amount
            )
        self._wallets = _wallets

    def _update_live(self) -> None:
        balances = self._exchange.get_balances()

        for currency in balances:
            if isinstance(balances[currency], dict):
                self._wallets[currency] = Wallet(
                    currency,
                    balances[currency].get('free', None),
                    balances[currency].get('used', None),
                    balances[currency].get('total', None)
                )
        for currency in deepcopy(self._wallets):
            if currency not in balances:
                del self._wallets[currency]

    def update(self, require_update: bool = True) -> None:
        if (require_update or (self._last_wallet_refresh + 3600 < arrow.utcnow().int_timestamp)):
            if (not self._config['dry_run'] or self._config.get('runmode') == RunMode.LIVE):
                self._update_live()
            else:
                self._update_dry()
            if self._log:
                pass
            self._last_wallet_refresh = arrow.utcnow().int_timestamp

    def get_all_balances(self) -> Dict[str, Any]:
        return self._wallets

    def get_starting_balance(self) -> float:
        if "available_capital" in self._config:
            return self._config['available_capital']
        else:
            tot_profit = Trade.get_total_closed_profit()
            open_stakes = Trade.total_open_trades_stakes()
            available_balance = self.get_free(self._config['stake_currency'])
            return available_balance - tot_profit + open_stakes

    def get_total_stake_amount(self):
        val_tied_up = Trade.total_open_trades_stakes()
        if "available_capital" in self._config:
            starting_balance = self._config['available_capital']
            tot_profit = Trade.get_total_closed_profit()
            available_amount = starting_balance + tot_profit

        else:
            available_amount = ((val_tied_up + self.get_free(self._config['stake_currency'])) *
                                self._config['tradable_balance_ratio'])
        return available_amount

    def get_available_stake_amount(self) -> float:

        free = self.get_free(self._config['stake_currency'])
        return min(self.get_total_stake_amount() - Trade.total_open_trades_stakes(), free)

    def _calculate_unlimited_stake_amount(self, available_amount: float,
                                          val_tied_up: float) -> float:
        if self._config['max_open_trades'] == 0:
            return 0

        possible_stake = (available_amount + val_tied_up) / self._config['max_open_trades']
        return min(possible_stake, available_amount)

    def _check_available_stake_amount(self, stake_amount: float, available_amount: float) -> float:

        if self._config['amend_last_stake_amount']:
            if available_amount > (stake_amount * self._config['last_stake_amount_min_ratio']):
                stake_amount = min(stake_amount, available_amount)
            else:
                stake_amount = 0

        if available_amount < stake_amount:
            pass


        return stake_amount

    def get_trade_stake_amount(self, pair: str, edge=None, update: bool = True) -> float:
        stake_amount: float
        if update:
            self.update()
        val_tied_up = Trade.total_open_trades_stakes()
        available_amount = self.get_available_stake_amount()

        if edge:
            stake_amount = edge.stake_amount(
                pair,
                self.get_free(self._config['stake_currency']),
                self.get_total(self._config['stake_currency']),
                val_tied_up
            )
        else:
            stake_amount = self._config['stake_amount']
            if stake_amount == UNLIMITED_STAKE_AMOUNT:
                stake_amount = self._calculate_unlimited_stake_amount(
                    available_amount, val_tied_up)

        return self._check_available_stake_amount(stake_amount, available_amount)

    def validate_stake_amount(
            self, pair: str, stake_amount: Optional[float], min_stake_amount: Optional[float]):
        if not stake_amount:
            return 0

        max_stake_amount = self.get_available_stake_amount()

        if min_stake_amount is not None and min_stake_amount > max_stake_amount:
            if self._log:
                pass
            return 0
        if min_stake_amount is not None and stake_amount < min_stake_amount:
            if self._log:
                pass
            if stake_amount * 1.3 < min_stake_amount:

                return 0
            stake_amount = min_stake_amount

        if stake_amount > max_stake_amount:

            stake_amount = max_stake_amount
        return stake_amount
