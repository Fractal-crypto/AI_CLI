"""
IStrategy interface
This module defines the interface to apply for strategies
"""
import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

import arrow
from pandas import DataFrame

from threee.constants import ListPairsWithTimeframes
from threee.data.dataprovider import DataProvider
from threee.enums import SellType, SignalTagType, SignalType
from threee.exceptions import OperationalException, StrategyError
from threee.exchange import timeframe_to_minutes, timeframe_to_seconds
from threee.exchange.exchange import timeframe_to_next_date
from threee.persistence import PairLocks, Trade
from threee.persistence.models import LocalTrade, Order
from threee.strategy.hyper import HyperStrategyMixin
from threee.strategy.informative_decorator import (InformativeData, PopulateIndicators,
                                                      _create_and_merge_informative_pair,
                                                      _format_pair_name)
from threee.strategy.strategy_wrapper import strategy_safe_wrapper
from threee.wallets import Wallets


CUSTOM_SELL_MAX_LENGTH = 64


class SellCheckTuple:
    sell_type: SellType
    sell_reason: str = ''

    def __init__(self, sell_type: SellType, sell_reason: str = ''):
        self.sell_type = sell_type
        self.sell_reason = sell_reason or sell_type.value

    @property
    def sell_flag(self):
        return self.sell_type != SellType.NONE


class IStrategy(ABC, HyperStrategyMixin):
    INTERFACE_VERSION: int = 2

    _populate_fun_len: int = 0
    _buy_fun_len: int = 0
    _sell_fun_len: int = 0
    _ft_params_from_file: Dict
    minimal_roi: Dict = {}

    stoploss: float

    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached = False
    use_custom_stoploss: bool = False

    ticker_interval: str
    timeframe: str

    order_types: Dict = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
    }

    order_time_in_force: Dict = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    process_only_new_candles: bool = False

    use_sell_signal: bool
    sell_profit_only: bool
    sell_profit_offset: float
    ignore_roi_if_buy_signal: bool

    position_adjustment_enable: bool = False
    max_entry_position_adjustment: int = -1

    ignore_buying_expired_candle_after: int = 0

    disable_dataframe_checks: bool = False

    startup_candle_count: int = 0

    protections: List = []

    dp: Optional[DataProvider]
    wallets: Optional[Wallets] = None
    stake_currency: str
    __source__: str = ''

    plot_config: Dict = {}

    def __init__(self, config: dict) -> None:
        self.config = config
        self._last_candle_seen_per_pair: Dict[str, datetime] = {}
        super().__init__(config)

        self._ft_informative: List[Tuple[InformativeData, PopulateIndicators]] = []
        for attr_name in dir(self.__class__):
            cls_method = getattr(self.__class__, attr_name)
            if not callable(cls_method):
                continue
            informative_data_list = getattr(cls_method, '_ft_informative', None)
            if not isinstance(informative_data_list, list):
                continue
            strategy_timeframe_minutes = timeframe_to_minutes(self.timeframe)
            for informative_data in informative_data_list:
                if timeframe_to_minutes(informative_data.timeframe) < strategy_timeframe_minutes:
                    raise OperationalException('Informative timeframe must be equal or higher than '
                                               'strategy timeframe!')
                self._ft_informative.append((informative_data, cls_method))

    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    @abstractmethod
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    @abstractmethod
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def bot_loop_start(self, **kwargs) -> None:

        pass

    def check_buy_timeout(self, pair: str, trade: Trade, order: dict,
                          current_time: datetime, **kwargs) -> bool:

        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict,
                           current_time: datetime, **kwargs) -> bool:

        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            **kwargs) -> bool:

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        return True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        return self.stoploss

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], **kwargs) -> float:

        return proposed_rate

    def custom_exit_price(self, pair: str, trade: Trade,
                          current_time: datetime, proposed_rate: float,
                          current_profit: float, **kwargs) -> float:

        return proposed_rate

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: Optional[str], **kwargs) -> float:

        return proposed_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs) -> Optional[float]:

        return None

    def informative_pairs(self) -> ListPairsWithTimeframes:

        return []

    def version(self) -> Optional[str]:

        return None

    def gather_informative_pairs(self) -> ListPairsWithTimeframes:

        informative_pairs = self.informative_pairs()
        for inf_data, _ in self._ft_informative:
            if inf_data.asset:
                pair_tf = (_format_pair_name(self.config, inf_data.asset), inf_data.timeframe)
                informative_pairs.append(pair_tf)
            else:
                if not self.dp:
                    pass
                for pair in self.dp.current_whitelist():
                    informative_pairs.append((pair, inf_data.timeframe))
        return list(set(informative_pairs))

    def get_strategy_name(self) -> str:

        return self.__class__.__name__

    def lock_pair(self, pair: str, until: datetime, reason: str = None) -> None:

        PairLocks.lock_pair(pair, until, reason)

    def unlock_pair(self, pair: str) -> None:

        PairLocks.unlock_pair(pair, datetime.now(timezone.utc))

    def unlock_reason(self, reason: str) -> None:

        PairLocks.unlock_reason(reason, datetime.now(timezone.utc))

    def is_pair_locked(self, pair: str, candle_date: datetime = None) -> bool:


        if not candle_date:
            # Simple call ...
            return PairLocks.is_pair_locked(pair)
        else:
            lock_time = timeframe_to_next_date(self.timeframe, candle_date)
            return PairLocks.is_pair_locked(pair, lock_time)

    def analyze_ticker(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.advise_indicators(dataframe, metadata)
        dataframe = self.advise_buy(dataframe, metadata)
        dataframe = self.advise_sell(dataframe, metadata)
        return dataframe

    def _analyze_ticker_internal(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        pair = str(metadata.get('pair'))
        if (not self.process_only_new_candles or
                self._last_candle_seen_per_pair.get(pair, None) != dataframe.iloc[-1]['date']):

            dataframe = self.analyze_ticker(dataframe, metadata)
            self._last_candle_seen_per_pair[pair] = dataframe.iloc[-1]['date']
            if self.dp:
                self.dp._set_cached_df(pair, self.timeframe, dataframe)
        else:
            dataframe['buy'] = 0
            dataframe['sell'] = 0
            dataframe['buy_tag'] = None
            dataframe['exit_tag'] = None


        return dataframe

    def analyze_pair(self, pair: str) -> None:

        if not self.dp:
            raise OperationalException("DataProvider not found.")
        dataframe = self.dp.ohlcv(pair, self.timeframe)
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            return

        try:
            df_len, df_close, df_date = self.preserve_df(dataframe)

            dataframe = strategy_safe_wrapper(
                self._analyze_ticker_internal, message=""
            )(dataframe, {'pair': pair})

            self.assert_df(dataframe, df_len, df_close, df_date)
        except StrategyError as error:
            return

        if dataframe.empty:
            return

    def analyze(self, pairs: List[str]) -> None:
        for pair in pairs:
            self.analyze_pair(pair)

    @staticmethod
    def preserve_df(dataframe: DataFrame) -> Tuple[int, float, datetime]:

        return len(dataframe), dataframe["close"].iloc[-1], dataframe["date"].iloc[-1]

    def assert_df(self, dataframe: DataFrame, df_len: int, df_close: float, df_date: datetime):

        message_template = "Dataframe returned from strategy has mismatching {}."
        message = ""
        if dataframe is None:
            message = "No dataframe returned (return statement missing?)."
        elif 'buy' not in dataframe:
            message = "Buy column not set."
        elif df_len != len(dataframe):
            message = message_template.format("length")
        elif df_close != dataframe["close"].iloc[-1]:
            message = message_template.format("last close price")
        elif df_date != dataframe["date"].iloc[-1]:
            message = message_template.format("last date")
        if message:
            if self.disable_dataframe_checks:
                pass
            else:
                raise StrategyError(message)

    def get_signal(
        self,
        pair: str,
        timeframe: str,
        dataframe: DataFrame
    ) -> Tuple[bool, bool, Optional[str], Optional[str]]:

        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            return False, False, None, None

        latest_date = dataframe['date'].max()
        latest = dataframe.loc[dataframe['date'] == latest_date].iloc[-1]
        latest_date = arrow.get(latest_date)

        timeframe_minutes = timeframe_to_minutes(timeframe)
        offset = self.config.get('exchange', {}).get('outdated_offset', 5)
        if latest_date < (arrow.utcnow().shift(minutes=-(timeframe_minutes * 2 + offset))):
            return False, False, None, None

        buy = latest[SignalType.BUY.value] == 1

        sell = False
        if SignalType.SELL.value in latest:
            sell = latest[SignalType.SELL.value] == 1

        buy_tag = latest.get(SignalTagType.BUY_TAG.value, None)
        exit_tag = latest.get(SignalTagType.EXIT_TAG.value, None)
        buy_tag = buy_tag if isinstance(buy_tag, str) else None
        exit_tag = exit_tag if isinstance(exit_tag, str) else None

        timeframe_seconds = timeframe_to_seconds(timeframe)
        if self.ignore_expired_candle(latest_date=latest_date,
                                      current_time=datetime.now(timezone.utc),
                                      timeframe_seconds=timeframe_seconds,
                                      buy=buy):
            return False, sell, buy_tag, exit_tag
        return buy, sell, buy_tag, exit_tag

    def ignore_expired_candle(self, latest_date: datetime, current_time: datetime,
                              timeframe_seconds: int, buy: bool):
        if self.ignore_buying_expired_candle_after and buy:
            time_delta = current_time - (latest_date + timedelta(seconds=timeframe_seconds))
            return time_delta.total_seconds() > self.ignore_buying_expired_candle_after
        else:
            return False

    def should_sell(self, trade: Trade, rate: float, current_time: datetime, buy: bool,
                    sell: bool, low: float = None, high: float = None,
                    force_stoploss: float = 0) -> SellCheckTuple:

        current_rate = rate
        current_profit = trade.calc_profit_ratio(current_rate)

        trade.adjust_min_max_rates(high or current_rate, low or current_rate)

        stoplossflag = self.stop_loss_reached(current_rate=current_rate, trade=trade,
                                              current_time=current_time,
                                              current_profit=current_profit,
                                              force_stoploss=force_stoploss, low=low, high=high)

        current_rate = high or rate
        current_profit = trade.calc_profit_ratio(current_rate)

        roi_reached = (not (buy and self.ignore_roi_if_buy_signal)
                       and self.min_roi_reached(trade=trade, current_profit=current_profit,
                                                current_time=current_time))

        sell_signal = SellType.NONE
        custom_reason = ''
        current_rate = rate
        current_profit = trade.calc_profit_ratio(current_rate)

        if (self.sell_profit_only and current_profit <= self.sell_profit_offset):
            pass
        elif self.use_sell_signal and not buy:
            if sell:
                sell_signal = SellType.SELL_SIGNAL
            else:
                custom_reason = strategy_safe_wrapper(self.custom_sell, default_retval=False)(
                    pair=trade.pair, trade=trade, current_time=current_time,
                    current_rate=current_rate, current_profit=current_profit)
                if custom_reason:
                    sell_signal = SellType.CUSTOM_SELL
                    if isinstance(custom_reason, str):
                        if len(custom_reason) > CUSTOM_SELL_MAX_LENGTH:

                            custom_reason = custom_reason[:CUSTOM_SELL_MAX_LENGTH]
                    else:
                        custom_reason = None
            if sell_signal in (SellType.CUSTOM_SELL, SellType.SELL_SIGNAL):
                return SellCheckTuple(sell_type=sell_signal, sell_reason=custom_reason)


        if roi_reached and stoplossflag.sell_type != SellType.STOP_LOSS:
            return SellCheckTuple(sell_type=SellType.ROI)

        if stoplossflag.sell_flag:

            return stoplossflag

        return SellCheckTuple(sell_type=SellType.NONE)

    def stop_loss_reached(self, current_rate: float, trade: Trade,
                          current_time: datetime, current_profit: float,
                          force_stoploss: float, low: float = None,
                          high: float = None) -> SellCheckTuple:
        stop_loss_value = force_stoploss if force_stoploss else self.stoploss

        trade.adjust_stop_loss(trade.open_rate, stop_loss_value, initial=True)

        if self.use_custom_stoploss and trade.stop_loss < (low or current_rate):
            stop_loss_value = strategy_safe_wrapper(self.custom_stoploss, default_retval=None
                                                    )(pair=trade.pair, trade=trade,
                                                      current_time=current_time,
                                                      current_rate=current_rate,
                                                      current_profit=current_profit)
            if stop_loss_value:

                trade.adjust_stop_loss(current_rate, stop_loss_value)
            else:
                pass

        if self.trailing_stop and trade.stop_loss < (low or current_rate):
            sl_offset = self.trailing_stop_positive_offset

            high_profit = current_profit if not high else trade.calc_profit_ratio(high)

            if not (self.trailing_only_offset_is_reached and high_profit < sl_offset):
                if self.trailing_stop_positive is not None and high_profit > sl_offset:
                    stop_loss_value = self.trailing_stop_positive

                trade.adjust_stop_loss(high or current_rate, stop_loss_value)

        if ((trade.stop_loss >= (low or current_rate)) and
                (not self.order_types.get('stoploss_on_exchange') or self.config['dry_run'])):

            sell_type = SellType.STOP_LOSS

            # If initial stoploss is not the same as current one then it is trailing.
            if trade.initial_stop_loss != trade.stop_loss:
                sell_type = SellType.TRAILING_STOP_LOSS

            return SellCheckTuple(sell_type=sell_type)

        return SellCheckTuple(sell_type=SellType.NONE)

    def min_roi_reached_entry(self, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:
        roi_list = list(filter(lambda x: x <= trade_dur, self.minimal_roi.keys()))
        if not roi_list:
            return None, None
        roi_entry = max(roi_list)
        return roi_entry, self.minimal_roi[roi_entry]

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi

    def ft_check_timed_out(self, side: str, trade: LocalTrade, order: Order,
                           current_time: datetime) -> bool:
        timeout = self.config.get('unfilledtimeout', {}).get(side)
        if timeout is not None:
            timeout_unit = self.config.get('unfilledtimeout', {}).get('unit', 'minutes')
            timeout_kwargs = {timeout_unit: -timeout}
            timeout_threshold = current_time + timedelta(**timeout_kwargs)
            timedout = (order.status == 'open' and order.side == side
                        and order.order_date_utc < timeout_threshold)
            if timedout:
                return True
        time_method = self.check_sell_timeout if order.side == 'sell' else self.check_buy_timeout

        return strategy_safe_wrapper(time_method,
                                     default_retval=False)(
                                        pair=trade.pair, trade=trade, order=order,
                                        current_time=current_time)

    def advise_all_indicators(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        return {pair: self.advise_indicators(pair_data.copy(), {'pair': pair}).copy()
                for pair, pair_data in data.items()}

    def advise_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for inf_data, populate_fn in self._ft_informative:
            dataframe = _create_and_merge_informative_pair(
                self, dataframe, metadata, inf_data, populate_fn)

        if self._populate_fun_len == 2:
            warnings.warn("deprecated - check out the Sample strategy to see "
                          "the current function headers!", DeprecationWarning)
            return self.populate_indicators(dataframe)
        else:
            return self.populate_indicators(dataframe, metadata)

    def advise_buy(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self._buy_fun_len == 2:
            warnings.warn("deprecated - check out the Sample strategy to see "
                          "the current function headers!", DeprecationWarning)
            return self.populate_buy_trend(dataframe)  # type: ignore
        else:
            return self.populate_buy_trend(dataframe, metadata)

    def advise_sell(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self._sell_fun_len == 2:
            warnings.warn("deprecated - check out the Sample strategy to see "
                          "the current function headers!", DeprecationWarning)
            return self.populate_sell_trend(dataframe)  # type: ignore
        else:
            return self.populate_sell_trend(dataframe, metadata)
