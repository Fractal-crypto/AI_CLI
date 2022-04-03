# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logica
"""
import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from pandas import DataFrame

from threee import constants
from threee.configuration import TimeRange, validate_config_consistency
from threee.constants import DATETIME_PRINT_FORMAT
from threee.data import history
from threee.data.btanalysis import find_existing_backtest_stats, trade_list_to_dataframe
from threee.data.converter import trim_dataframe, trim_dataframes
from threee.data.dataprovider import DataProvider
from threee.enums import BacktestState, SellType
from threee.exceptions import DependencyException, OperationalException
from threee.exchange import timeframe_to_minutes, timeframe_to_seconds
from threee.misc import get_strategy_run_id
from threee.mixins import LoggingMixin
from threee.optimize.bt_progress import BTProgress
from threee.optimize.optimize_reports import (generate_backtest_stats, show_backtest_results,
                                                 store_backtest_stats)
from threee.persistence import LocalTrade, Order, PairLocks, Trade
from threee.plugins.pairlistmanager import PairListManager
from threee.plugins.protectionmanager import ProtectionManager
from threee.resolvers import ExchangeResolver, StrategyResolver
from threee.strategy.interface import IStrategy, SellCheckTuple
from threee.strategy.strategy_wrapper import strategy_safe_wrapper
from threee.wallets import Wallets


logger = logging.getLogger(__name__)

# Indexes for backtest tuples
DATE_IDX = 0
BUY_IDX = 1
OPEN_IDX = 2
CLOSE_IDX = 3
SELL_IDX = 4
LOW_IDX = 5
HIGH_IDX = 6
BUY_TAG_IDX = 7
EXIT_TAG_IDX = 8


class Backtesting:

    def __init__(self, config: Dict[str, Any]) -> None:

        LoggingMixin.show_output = False
        self.config = config
        self.results: Dict[str, Any] = {}
        self.trade_id_counter: int = 0
        self.order_id_counter: int = 0

        config['dry_run'] = True
        self.run_ids: Dict[str, str] = {}
        self.strategylist: List[IStrategy] = []
        self.all_results: Dict[str, Dict] = {}

        self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)
        self.dataprovider = DataProvider(self.config, self.exchange)

        if self.config.get('strategy_list', None):
            for strat in list(self.config['strategy_list']):
                stratconf = deepcopy(self.config)
                stratconf['strategy'] = strat
                self.strategylist.append(StrategyResolver.load_strategy(stratconf))
                validate_config_consistency(stratconf)

        else:
            self.strategylist.append(StrategyResolver.load_strategy(self.config))
            validate_config_consistency(self.config)


        self.timeframe = str(self.config.get('timeframe'))
        self.timeframe_min = timeframe_to_minutes(self.timeframe)
        self.init_backtest_detail()
        self.pairlists = PairListManager(self.exchange, self.config)


        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.pairlists.refresh_pairlist()

        if config.get('fee', None) is not None:
            self.fee = config['fee']
        else:
            self.fee = self.exchange.get_fee(symbol=self.pairlists.whitelist[0])

        self.timerange = TimeRange.parse_timerange(
            None if self.config.get('timerange') is None else str(self.config.get('timerange')))
        self.required_startup = max([strat.startup_candle_count for strat in self.strategylist])
        self.config['startup_candle_count'] = self.required_startup
        self.exchange.validate_required_startup_candles(self.required_startup, self.timeframe)
        self.init_backtest()

    def __del__(self):
        self.cleanup()

    @staticmethod
    def cleanup():
        LoggingMixin.show_output = True
        PairLocks.use_db = True
        Trade.use_db = True

    def init_backtest_detail(self):
        self.timeframe_detail = str(self.config.get('timeframe_detail', ''))
        if self.timeframe_detail:
            self.timeframe_detail_min = timeframe_to_minutes(self.timeframe_detail)
        else:
            self.timeframe_detail_min = 0
        self.detail_data: Dict[str, DataFrame] = {}

    def init_backtest(self):

        self.prepare_backtest(False)

        self.wallets = Wallets(self.config, self.exchange, log=False)

        self.progress = BTProgress()
        self.abort = False

    def _set_strategy(self, strategy: IStrategy):
        """
        백테스팅에 전략 로드
        """
        self.strategy: IStrategy = strategy
        strategy.dp = self.dataprovider
        strategy.wallets = self.wallets
        self.strategy.order_types['stoploss_on_exchange'] = False

    def _load_protections(self, strategy: IStrategy):
        if self.config.get('enable_protections', False):
            conf = self.config
            if hasattr(strategy, 'protections'):
                conf = deepcopy(conf)
                conf['protections'] = strategy.protections
            self.protections = ProtectionManager(self.config, strategy.protections)

    def load_bt_data(self) -> Tuple[Dict[str, DataFrame], TimeRange]:
        self.progress.init_step(BacktestState.DATALOAD, 1)

        data = history.load_data(
            datadir=self.config['datadir'],
            pairs=self.pairlists.whitelist,
            timeframe=self.timeframe,
            timerange=self.timerange,
            startup_candles=self.required_startup,
            fail_without_data=True,
            data_format=self.config.get('dataformat_ohlcv', 'json'),
        )

        min_date, max_date = history.get_timerange(data)
        self.timerange.adjust_start_if_necessary(timeframe_to_seconds(self.timeframe),
                                                 self.required_startup, min_date)

        self.progress.set_new_value(1)
        return data, self.timerange

    def load_bt_data_detail(self) -> None:
        if self.timeframe_detail:
            self.detail_data = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=self.timeframe_detail,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config.get('dataformat_ohlcv', 'json'),
            )
        else:
            self.detail_data = {}

    def prepare_backtest(self, enable_protections):
        PairLocks.use_db = False
        PairLocks.timeframe = self.config['timeframe']
        Trade.use_db = False
        PairLocks.reset_locks()
        Trade.reset_trades()
        self.rejected_trades = 0
        self.timedout_entry_orders = 0
        self.timedout_exit_orders = 0
        self.dataprovider.clear_cache()
        if enable_protections:
            self._load_protections(self.strategy)

    def check_abort(self):

        if self.abort:
            self.abort = False

    def _get_ohlcv_as_lists(self, processed: Dict[str, DataFrame]) -> Dict[str, Tuple]:

        headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high', 'buy_tag', 'exit_tag']
        data: Dict = {}
        self.progress.init_step(BacktestState.CONVERT, len(processed))

        for pair in processed.keys():
            pair_data = processed[pair]
            self.check_abort()
            self.progress.increment()
            if not pair_data.empty:
                pair_data.loc[:, 'buy'] = 0
                pair_data.loc[:, 'sell'] = 0
                pair_data.loc[:, 'buy_tag'] = None
                pair_data.loc[:, 'exit_tag'] = None

            df_analyzed = self.strategy.advise_sell(
                self.strategy.advise_buy(pair_data, {'pair': pair}), {'pair': pair}).copy()
            df_analyzed = processed[pair] = pair_data = trim_dataframe(
                df_analyzed, self.timerange, startup_candles=self.required_startup)
            self.dataprovider._set_cached_df(pair, self.timeframe, df_analyzed)
            df_analyzed = df_analyzed.copy()
            df_analyzed.loc[:, 'buy'] = df_analyzed.loc[:, 'buy'].shift(1)
            df_analyzed.loc[:, 'sell'] = df_analyzed.loc[:, 'sell'].shift(1)
            df_analyzed.loc[:, 'buy_tag'] = df_analyzed.loc[:, 'buy_tag'].shift(1)
            df_analyzed.loc[:, 'exit_tag'] = df_analyzed.loc[:, 'exit_tag'].shift(1)
            df_analyzed = df_analyzed.drop(df_analyzed.head(1).index)
            data[pair] = df_analyzed[headers].values.tolist()
        return data

    def _get_close_rate(self, sell_row: Tuple, trade: LocalTrade, sell: SellCheckTuple,
                        trade_dur: int) -> float:
        """
        백테스팅 결과에 대한 가까운 결과 가져오기
        """
        if sell.sell_type in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
            if trade.stop_loss > sell_row[HIGH_IDX]:
                return sell_row[OPEN_IDX]
            if sell.sell_type == SellType.TRAILING_STOP_LOSS and trade_dur == 0:
                if (
                    not self.strategy.use_custom_stoploss and self.strategy.trailing_stop
                    and self.strategy.trailing_only_offset_is_reached
                    and self.strategy.trailing_stop_positive_offset is not None
                    and self.strategy.trailing_stop_positive
                ):
                    stop_rate = (sell_row[OPEN_IDX] *
                                 (1 + abs(self.strategy.trailing_stop_positive_offset) -
                                  abs(self.strategy.trailing_stop_positive)))
                else:

                    stop_rate = sell_row[OPEN_IDX] * (1 - abs(trade.stop_loss_pct))
                    assert stop_rate < sell_row[HIGH_IDX]
                return max(sell_row[LOW_IDX], stop_rate)

            return trade.stop_loss
        elif sell.sell_type == (SellType.ROI):
            roi_entry, roi = self.strategy.min_roi_reached_entry(trade_dur)
            if roi is not None and roi_entry is not None:
                if roi == -1 and roi_entry % self.timeframe_min == 0:
                    return sell_row[OPEN_IDX]
                close_rate = - (trade.open_rate * roi + trade.open_rate *
                                (1 + trade.fee_open)) / (trade.fee_close - 1)

                if (trade_dur > 0 and trade_dur == roi_entry
                        and roi_entry % self.timeframe_min == 0
                        and sell_row[OPEN_IDX] > close_rate):
                    return sell_row[OPEN_IDX]

                if (
                    trade_dur == 0

                    and sell_row[OPEN_IDX] > sell_row[CLOSE_IDX]
                    and trade.open_rate < sell_row[OPEN_IDX]
                    and close_rate > sell_row[CLOSE_IDX]
                ):
                    pass
                return min(max(close_rate, sell_row[LOW_IDX]), sell_row[HIGH_IDX])

            else:
                return sell_row[OPEN_IDX]
        else:
            return sell_row[OPEN_IDX]

    def _get_adjust_trade_entry_for_candle(self, trade: LocalTrade, row: Tuple
                                           ) -> LocalTrade:

        current_profit = trade.calc_profit_ratio(row[OPEN_IDX])
        min_stake = self.exchange.get_min_pair_stake_amount(trade.pair, row[OPEN_IDX], -0.1)
        max_stake = self.wallets.get_available_stake_amount()
        stake_amount = strategy_safe_wrapper(self.strategy.adjust_trade_position,
                                             default_retval=None)(
            trade=trade, current_time=row[DATE_IDX].to_pydatetime(), current_rate=row[OPEN_IDX],
            current_profit=current_profit, min_stake=min_stake, max_stake=max_stake)

        if stake_amount is not None and stake_amount > 0.0:
            pos_trade = self._enter_trade(trade.pair, row, stake_amount, trade)
            if pos_trade is not None:
                self.wallets.update()
                return pos_trade

        return trade

    def _get_order_filled(self, rate: float, row: Tuple) -> bool:
        return row[LOW_IDX] <= rate <= row[HIGH_IDX]

    def _get_sell_trade_entry_for_candle(self, trade: LocalTrade,
                                         sell_row: Tuple) -> Optional[LocalTrade]:

        if self.strategy.position_adjustment_enable:
            check_adjust_buy = True
            if self.strategy.max_entry_position_adjustment > -1:
                count_of_buys = trade.nr_of_successful_buys
                check_adjust_buy = (count_of_buys <= self.strategy.max_entry_position_adjustment)
            if check_adjust_buy:
                trade = self._get_adjust_trade_entry_for_candle(trade, sell_row)

        sell_candle_time = sell_row[DATE_IDX].to_pydatetime()
        sell = self.strategy.should_sell(trade, sell_row[OPEN_IDX],
                                         sell_candle_time, sell_row[BUY_IDX],
                                         sell_row[SELL_IDX],
                                         low=sell_row[LOW_IDX], high=sell_row[HIGH_IDX])

        if sell.sell_flag:
            trade.close_date = sell_candle_time

            trade_dur = int((trade.close_date_utc - trade.open_date_utc).total_seconds() // 60)
            try:
                closerate = self._get_close_rate(sell_row, trade, sell, trade_dur)
            except ValueError:
                return None
            current_profit = trade.calc_profit_ratio(closerate)
            order_type = self.strategy.order_types['sell']
            if sell.sell_type in (SellType.SELL_SIGNAL, SellType.CUSTOM_SELL):
                if order_type == 'limit':
                    closerate = strategy_safe_wrapper(self.strategy.custom_exit_price,
                                                      default_retval=closerate)(
                        pair=trade.pair, trade=trade,
                        current_time=sell_candle_time,
                        proposed_rate=closerate, current_profit=current_profit)
                    closerate = max(closerate, sell_row[LOW_IDX])
            time_in_force = self.strategy.order_time_in_force['sell']

            if not strategy_safe_wrapper(self.strategy.confirm_trade_exit, default_retval=True)(
                    pair=trade.pair, trade=trade, order_type='limit', amount=trade.amount,
                    rate=closerate,
                    time_in_force=time_in_force,
                    sell_reason=sell.sell_reason,
                    current_time=sell_candle_time):
                return None

            trade.sell_reason = sell.sell_reason

            if(
                len(sell_row) > EXIT_TAG_IDX
                and sell_row[EXIT_TAG_IDX] is not None
                and len(sell_row[EXIT_TAG_IDX]) > 0
            ):
                trade.sell_reason = sell_row[EXIT_TAG_IDX]

            self.order_id_counter += 1
            order = Order(
                id=self.order_id_counter,
                ft_trade_id=trade.id,
                order_date=sell_candle_time,
                order_update_date=sell_candle_time,
                ft_is_open=True,
                ft_pair=trade.pair,
                order_id=str(self.order_id_counter),
                symbol=trade.pair,
                ft_order_side="sell",
                side="sell",
                order_type=order_type,
                status="open",
                price=closerate,
                average=closerate,
                amount=trade.amount,
                filled=0,
                remaining=trade.amount,
                cost=trade.amount * closerate,
            )
            trade.orders.append(order)
            return trade

        return None

    def _get_sell_trade_entry(self, trade: LocalTrade, sell_row: Tuple) -> Optional[LocalTrade]:
        if self.timeframe_detail and trade.pair in self.detail_data:
            sell_candle_time = sell_row[DATE_IDX].to_pydatetime()
            sell_candle_end = sell_candle_time + timedelta(minutes=self.timeframe_min)

            detail_data = self.detail_data[trade.pair]
            detail_data = detail_data.loc[
                (detail_data['date'] >= sell_candle_time) &
                (detail_data['date'] < sell_candle_end)
            ].copy()
            if len(detail_data) == 0:
                return self._get_sell_trade_entry_for_candle(trade, sell_row)
            detail_data.loc[:, 'buy'] = sell_row[BUY_IDX]
            detail_data.loc[:, 'sell'] = sell_row[SELL_IDX]
            detail_data.loc[:, 'buy_tag'] = sell_row[BUY_TAG_IDX]
            detail_data.loc[:, 'exit_tag'] = sell_row[EXIT_TAG_IDX]
            headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high', 'buy_tag', 'exit_tag']
            for det_row in detail_data[headers].values.tolist():
                res = self._get_sell_trade_entry_for_candle(trade, det_row)
                if res:
                    return res

            return None

        else:
            return self._get_sell_trade_entry_for_candle(trade, sell_row)

    def _enter_trade(self, pair: str, row: Tuple, stake_amount: Optional[float] = None,
                     trade: Optional[LocalTrade] = None) -> Optional[LocalTrade]:

        current_time = row[DATE_IDX].to_pydatetime()
        entry_tag = row[BUY_TAG_IDX] if len(row) >= BUY_TAG_IDX + 1 else None
        order_type = self.strategy.order_types['buy']
        propose_rate = row[OPEN_IDX]
        if order_type == 'limit':
            propose_rate = strategy_safe_wrapper(self.strategy.custom_entry_price,
                                                 default_retval=row[OPEN_IDX])(
                pair=pair, current_time=current_time,
                proposed_rate=propose_rate, entry_tag=entry_tag)
            propose_rate = min(propose_rate, row[HIGH_IDX])

        min_stake_amount = self.exchange.get_min_pair_stake_amount(pair, propose_rate, -0.05) or 0
        max_stake_amount = self.wallets.get_available_stake_amount()

        pos_adjust = trade is not None
        if not pos_adjust:
            try:
                stake_amount = self.wallets.get_trade_stake_amount(pair, None, update=False)
            except DependencyException:
                return None

            stake_amount = strategy_safe_wrapper(self.strategy.custom_stake_amount,
                                                 default_retval=stake_amount)(
                pair=pair, current_time=current_time, current_rate=propose_rate,
                proposed_stake=stake_amount, min_stake=min_stake_amount, max_stake=max_stake_amount,
                entry_tag=entry_tag)

        stake_amount = self.wallets.validate_stake_amount(pair, stake_amount, min_stake_amount)

        if not stake_amount:
            return trade

        time_in_force = self.strategy.order_time_in_force['buy']
        if not pos_adjust:
            if not strategy_safe_wrapper(self.strategy.confirm_trade_entry, default_retval=True)(
                    pair=pair, order_type=order_type, amount=stake_amount, rate=propose_rate,
                    time_in_force=time_in_force, current_time=current_time,
                    entry_tag=entry_tag):
                return None

        if stake_amount and (not min_stake_amount or stake_amount > min_stake_amount):
            self.order_id_counter += 1
            amount = round(stake_amount / propose_rate, 8)
            if trade is None:
                # Enter trade
                self.trade_id_counter += 1
                trade = LocalTrade(
                    id=self.trade_id_counter,
                    open_order_id=self.order_id_counter,
                    pair=pair,
                    open_rate=propose_rate,
                    open_rate_requested=propose_rate,
                    open_date=current_time,
                    stake_amount=stake_amount,
                    amount=amount,
                    amount_requested=amount,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                    buy_tag=entry_tag,
                    exchange='backtesting',
                    orders=[]
                )

            trade.adjust_stop_loss(trade.open_rate, self.strategy.stoploss, initial=True)

            order = Order(
                id=self.order_id_counter,
                ft_trade_id=trade.id,
                ft_is_open=True,
                ft_pair=trade.pair,
                order_id=str(self.order_id_counter),
                symbol=trade.pair,
                ft_order_side="buy",
                side="buy",
                order_type=order_type,
                status="open",
                order_date=current_time,
                order_filled_date=current_time,
                order_update_date=current_time,
                price=propose_rate,
                average=propose_rate,
                amount=amount,
                filled=0,
                remaining=amount,
                cost=stake_amount + trade.fee_open,
            )
            if pos_adjust and self._get_order_filled(order.price, row):
                order.close_bt_order(current_time)
            else:
                trade.open_order_id = str(self.order_id_counter)
            trade.orders.append(order)
            trade.recalc_trade_from_orders()

        return trade

    def handle_left_open(self, open_trades: Dict[str, List[LocalTrade]],
                         data: Dict[str, List[Tuple]]) -> List[LocalTrade]:
        """
        백테스팅 종료 시 미결 거래 처리
        """
        trades = []
        for pair in open_trades.keys():
            if len(open_trades[pair]) > 0:
                for trade in open_trades[pair]:
                    if trade.open_order_id and trade.nr_of_successful_buys == 0:

                        continue
                    sell_row = data[pair][-1]

                    trade.close_date = sell_row[DATE_IDX].to_pydatetime()
                    trade.sell_reason = SellType.FORCE_SELL.value
                    trade.close(sell_row[OPEN_IDX], show_msg=False)
                    LocalTrade.close_bt_trade(trade)
                    trade1 = deepcopy(trade)
                    trade1.is_open = True
                    trades.append(trade1)
        return trades

    def trade_slot_available(self, max_open_trades: int, open_trade_count: int) -> bool:
        if max_open_trades <= 0 or open_trade_count < max_open_trades:
            return True
        self.rejected_trades += 1
        return False

    def run_protections(self, enable_protections, pair: str, current_time: datetime):
        if enable_protections:
            self.protections.stop_per_pair(pair, current_time)
            self.protections.global_stop(current_time)

    def check_order_cancel(self, trade: LocalTrade, current_time) -> bool:
        for order in [o for o in trade.orders if o.ft_is_open]:

            timedout = self.strategy.ft_check_timed_out(order.side, trade, order, current_time)
            if timedout:
                if order.side == 'buy':
                    self.timedout_entry_orders += 1
                    if trade.nr_of_successful_buys == 0:

                        return True
                    else:
                        del trade.orders[trade.orders.index(order)]
                if order.side == 'sell':
                    self.timedout_exit_orders += 1
                    del trade.orders[trade.orders.index(order)]

        return False

    def validate_row(
            self, data: Dict, pair: str, row_index: int, current_time: datetime) -> Optional[Tuple]:
        try:
            row = data[pair][row_index]
        except IndexError:
            return None

        if row[DATE_IDX] > current_time:
            return None
        return row

    def backtest(self, processed: Dict,
                 start_date: datetime, end_date: datetime,
                 max_open_trades: int = 0, position_stacking: bool = False,
                 enable_protections: bool = False) -> Dict[str, Any]:
        """
        백테스팅 기능 구현
        """
        trades: List[LocalTrade] = []
        self.prepare_backtest(enable_protections)
        self.wallets.update()
        data: Dict = self._get_ohlcv_as_lists(processed)
        indexes: Dict = defaultdict(int)
        current_time = start_date + timedelta(minutes=self.timeframe_min)

        open_trades: Dict[str, List[LocalTrade]] = defaultdict(list)
        open_trade_count = 0

        self.progress.init_step(BacktestState.BACKTEST, int(
            (end_date - start_date) / timedelta(minutes=self.timeframe_min)))
        while current_time <= end_date:
            open_trade_count_start = open_trade_count
            self.check_abort()
            for i, pair in enumerate(data):
                row_index = indexes[pair]
                row = self.validate_row(data, pair, row_index, current_time)
                if not row:
                    continue

                row_index += 1
                indexes[pair] = row_index
                self.dataprovider._set_dataframe_max_index(row_index)

                # 구매 기능
                if (
                    (position_stacking or len(open_trades[pair]) == 0)
                    and self.trade_slot_available(max_open_trades, open_trade_count_start)
                    and current_time != end_date
                    and row[BUY_IDX] == 1
                    and row[SELL_IDX] != 1
                    and not PairLocks.is_pair_locked(pair, row[DATE_IDX])
                ):
                    trade = self._enter_trade(pair, row)
                    if trade:
                        open_trade_count_start += 1
                        open_trade_count += 1
                        open_trades[pair].append(trade)

                for trade in list(open_trades[pair]):
                    order = trade.select_order('buy', is_open=True)
                    if order and self._get_order_filled(order.price, row):
                        order.close_bt_order(current_time)
                        trade.open_order_id = None
                        LocalTrade.add_bt_trade(trade)
                        self.wallets.update()


                    if not trade.open_order_id:
                        self._get_sell_trade_entry(trade, row)

                    order = trade.select_order('sell', is_open=True)
                    if order and self._get_order_filled(order.price, row):
                        trade.open_order_id = None
                        trade.close_date = current_time
                        trade.close(order.price, show_msg=False)


                        open_trade_count -= 1
                        open_trades[pair].remove(trade)
                        LocalTrade.close_bt_trade(trade)
                        trades.append(trade)
                        self.wallets.update()
                        self.run_protections(enable_protections, pair, current_time)


                    if self.check_order_cancel(trade, current_time):

                        open_trade_count -= 1
                        open_trades[pair].remove(trade)
                        self.wallets.update()


            self.progress.increment()
            current_time += timedelta(minutes=self.timeframe_min)

        trades += self.handle_left_open(open_trades, data=data)
        self.wallets.update()

        results = trade_list_to_dataframe(trades)
        return {
            'results': results,
            'config': self.strategy.config,
            'locks': PairLocks.get_all_locks(),
            'rejected_signals': self.rejected_trades,
            'timedout_entry_orders': self.timedout_entry_orders,
            'timedout_exit_orders': self.timedout_exit_orders,
            'final_balance': self.wallets.get_total(self.strategy.config['stake_currency']),
        }

    def backtest_one_strategy(self, strat: IStrategy, data: Dict[str, DataFrame],
                              timerange: TimeRange):
        self.progress.init_step(BacktestState.ANALYZE, 0)

        backtest_start_time = datetime.now(timezone.utc)
        self._set_strategy(strat)

        strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)()

        if self.config.get('use_max_market_positions', True):

            max_open_trades = self.strategy.config['max_open_trades']
        else:
            max_open_trades = 0


        preprocessed = self.strategy.advise_all_indicators(data)


        preprocessed_tmp = trim_dataframes(preprocessed, timerange, self.required_startup)


        min_date, max_date = history.get_timerange(preprocessed_tmp)
        results = self.backtest(
            processed=preprocessed,
            start_date=min_date,
            end_date=max_date,
            max_open_trades=max_open_trades,
            position_stacking=self.config.get('position_stacking', False),
            enable_protections=self.config.get('enable_protections', False),
        )
        backtest_end_time = datetime.now(timezone.utc)
        results.update({
            'run_id': self.run_ids.get(strat.get_strategy_name(), ''),
            'backtest_start_time': int(backtest_start_time.timestamp()),
            'backtest_end_time': int(backtest_end_time.timestamp()),
        })
        self.all_results[self.strategy.get_strategy_name()] = results

        return min_date, max_date

    def _get_min_cached_backtest_date(self):
        min_backtest_date = None
        backtest_cache_age = self.config.get('backtest_cache', constants.BACKTEST_CACHE_DEFAULT)
        if self.timerange.stopts == 0 or datetime.fromtimestamp(
           self.timerange.stopts, tz=timezone.utc) > datetime.now(tz=timezone.utc):
            pass
        elif backtest_cache_age == 'day':
            min_backtest_date = datetime.now(tz=timezone.utc) - timedelta(days=1)
        elif backtest_cache_age == 'week':
            min_backtest_date = datetime.now(tz=timezone.utc) - timedelta(weeks=1)
        elif backtest_cache_age == 'month':
            min_backtest_date = datetime.now(tz=timezone.utc) - timedelta(weeks=4)
        return min_backtest_date

    def load_prior_backtest(self):
        self.run_ids = {
            strategy.get_strategy_name(): get_strategy_run_id(strategy)
            for strategy in self.strategylist
        }

        min_backtest_date = self._get_min_cached_backtest_date()
        if min_backtest_date is not None:
            self.results = find_existing_backtest_stats(
                self.config['user_data_dir'] / 'backtest_results', self.run_ids, min_backtest_date)

    def start(self) -> None:

        data: Dict[str, Any] = {}

        data, timerange = self.load_bt_data()
        self.load_bt_data_detail()

        self.load_prior_backtest()

        for strat in self.strategylist:
            if self.results and strat.get_strategy_name() in self.results['strategy']:
                continue
            min_date, max_date = self.backtest_one_strategy(strat, data, timerange)

        if len(self.all_results) > 0:
            results = generate_backtest_stats(
                data, self.all_results, min_date=min_date, max_date=max_date)
            if self.results:
                self.results['metadata'].update(results['metadata'])
                self.results['strategy'].update(results['strategy'])
                self.results['strategy_comparison'].extend(results['strategy_comparison'])
            else:
                self.results = results

            if self.config.get('export', 'none') == 'trades':
                store_backtest_stats(self.config['exportfilename'], self.results)

        if 'strategy_list' in self.config and len(self.results) > 0:
            self.results['strategy_comparison'] = sorted(
                self.results['strategy_comparison'],
                key=lambda c: self.config['strategy_list'].index(c['key']))
            self.results['strategy'] = dict(
                sorted(self.results['strategy'].items(),
                       key=lambda kv: self.config['strategy_list'].index(kv[0])))

        if len(self.strategylist) > 0:
            show_backtest_results(self.config, self.results)
