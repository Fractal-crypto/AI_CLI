"""
three is the main module of this bot. It contains the class three()
"""
import copy
import logging
import traceback
from datetime import datetime, timezone
from math import isclose
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from threee import __version__, constants
from threee.configuration import validate_config_consistency
from threee.data.converter import order_book_to_dataframe
from threee.data.dataprovider import DataProvider
# from threee.edge import Edge
from threee.enums import RPCMessageType, RunMode, SellType, State
from threee.exceptions import (DependencyException, ExchangeError, InsufficientFundsError,
                                  InvalidOrderException, PricingError)
from threee.exchange import timeframe_to_minutes, timeframe_to_seconds
from threee.misc import safe_value_fallback, safe_value_fallback2
from threee.mixins import LoggingMixin
from threee.persistence import Order, PairLocks, Trade, cleanup_db, init_db
from threee.plugins.pairlistmanager import PairListManager
from threee.plugins.protectionmanager import ProtectionManager
from threee.resolvers import ExchangeResolver, StrategyResolver
from threee.rpc import RPCManager
from threee.strategy.interface import IStrategy, SellCheckTuple
from threee.strategy.strategy_wrapper import strategy_safe_wrapper
from threee.wallets import Wallets

logger = logging.getLogger(__name__)
class threeeBot(LoggingMixin):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.active_pair_whitelist: List[str] = []


        self.state = State.STOPPED

        self.config = config

        self.strategy: IStrategy = StrategyResolver.load_strategy(self.config)

        validate_config_consistency(config)

        self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)

        init_db(self.config.get('db_url', None), clean_open_orders=self.config['dry_run'])

        self.wallets = Wallets(self.config, self.exchange)

        PairLocks.timeframe = self.config['timeframe']

        self.protections = ProtectionManager(self.config, self.strategy.protections)

        self.rpc: RPCManager = RPCManager(self)

        self.pairlists = PairListManager(self.exchange, self.config)

        self.dataprovider = DataProvider(self.config, self.exchange, self.pairlists)


        self.strategy.dp = self.dataprovider

        self.strategy.wallets = self.wallets

        self.edge = Edge(self.config, self.exchange, self.strategy) if \
            self.config.get('edge', {}).get('enabled', False) else None

        self.active_pair_whitelist = self._refresh_active_whitelist()

        initial_state = self.config.get('initial_state')
        self.state = State[initial_state.upper()] if initial_state else State.STOPPED

        self._exit_lock = Lock()
        LoggingMixin.__init__(self, None, timeframe_to_seconds(self.strategy.timeframe))

        self.last_process = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def notify_status(self, msg: str) -> None:
        self.rpc.send_msg({
            'type': RPCMessageType.STATUS,
            'status': msg
        })

    def cleanup(self) -> None:

        if self.config['cancel_open_orders_on_exit']:
            self.cancel_all_open_orders()

        self.check_for_open_trades()

        self.rpc.cleanup()
        cleanup_db()
        self.exchange.close()

    def startup(self) -> None:
        self.rpc.startup_messages(self.config, self.pairlists, self.protections)
        if not self.edge:
            Trade.stoploss_reinitialization(self.strategy.stoploss)

        self.startup_update_open_orders()

    def process(self) -> None:
        self.exchange.reload_markets()

        self.update_closed_trades_without_assigned_fees()

        trades = Trade.get_open_trades()

        self.active_pair_whitelist = self._refresh_active_whitelist(trades)

        self.dataprovider.refresh(self.pairlists.create_pair_list(self.active_pair_whitelist),
                                  self.strategy.gather_informative_pairs())

        strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)()

        self.strategy.analyze(self.active_pair_whitelist)

        with self._exit_lock:
            self.check_handle_timedout()

        with self._exit_lock:
            trades = Trade.get_open_trades()
            self.exit_positions(trades)

        if self.strategy.position_adjustment_enable:
            with self._exit_lock:
                self.process_open_trade_positions()

        if self.get_free_open_trades():
            self.enter_positions()

        Trade.commit()
        self.last_process = datetime.now(timezone.utc)

    def process_stopped(self) -> None:
        if self.config['cancel_open_orders_on_exit']:
            self.cancel_all_open_orders()

    def check_for_open_trades(self):
        open_trades = Trade.get_trades([Trade.is_open.is_(True)]).all()

        if len(open_trades) != 0 and self.state != State.RELOAD_CONFIG:
            msg = {}
            self.rpc.send_msg(msg)

    def _refresh_active_whitelist(self, trades: List[Trade] = []) -> List[str]:
        self.pairlists.refresh_pairlist()
        _whitelist = self.pairlists.whitelist

        if self.edge:
            self.edge.calculate(_whitelist)
            _whitelist = self.edge.adjust(_whitelist)

        if trades:
            _whitelist.extend([trade.pair for trade in trades if trade.pair not in _whitelist])
        return _whitelist

    def get_free_open_trades(self) -> int:
        open_trades = len(Trade.get_open_trades())
        return max(0, self.config['max_open_trades'] - open_trades)

    def startup_update_open_orders(self):
        if self.config['dry_run'] or self.config['exchange'].get('skip_open_order_update', False):
            return

        orders = Order.get_open_orders()
        for order in orders:
            try:
                fo = self.exchange.fetch_order_or_stoploss_order(order.order_id, order.ft_pair,
                                                                 order.ft_order_side == 'stoploss')

                self.update_trade_state(order.trade, order.order_id, fo)

            except:
                pass

    def update_closed_trades_without_assigned_fees(self):
        if self.config['dry_run']:
            return

        trades: List[Trade] = Trade.get_sold_trades_without_assigned_fees()
        for trade in trades:

            if not trade.is_open and not trade.fee_updated('sell'):
                order = trade.select_order('sell', False)
                if order:
                    self.update_trade_state(trade, order.order_id,
                                            stoploss_order=order.ft_order_side == 'stoploss',
                                            send_msg=False)

        trades: List[Trade] = Trade.get_open_trades_without_assigned_fees()
        for trade in trades:
            if trade.is_open and not trade.fee_updated('buy'):
                order = trade.select_order('buy', False)
                open_order = trade.select_order('buy', True)
                if order and open_order is None:
                    self.update_trade_state(trade, order.order_id, send_msg=False)

    def handle_insufficient_funds(self, trade: Trade):
        for order in trade.orders:
            fo = None
            if not order.ft_is_open:
                continue
            try:
                fo = self.exchange.fetch_order_or_stoploss_order(order.order_id, order.ft_pair,
                                                                 order.ft_order_side == 'stoploss')
                if order.ft_order_side == 'stoploss':
                    if fo and fo['status'] == 'open':
                        trade.stoploss_order_id = order.order_id
                elif order.ft_order_side == 'sell':
                    if fo and fo['status'] == 'open':
                        trade.open_order_id = order.order_id
                elif order.ft_order_side == 'buy':
                    if fo and fo['status'] == 'open':
                        trade.open_order_id = order.order_id
                if fo:
                    self.update_trade_state(trade, order.order_id, fo,
                                            stoploss_order=order.ft_order_side == 'stoploss')

            except :
                pass

#
    def enter_positions(self) -> int:
        trades_created = 0

        whitelist = copy.deepcopy(self.active_pair_whitelist)
        if not whitelist:
            return trades_created
        for trade in Trade.get_open_trades():
            if trade.pair in whitelist:
                whitelist.remove(trade.pair)

        if not whitelist:
            return trades_created
        if PairLocks.is_global_lock():
            lock = PairLocks.get_pair_longest_lock('*')
            return trades_created
        for pair in whitelist:
            try:
                trades_created += self.create_trade(pair)
            except :
                pass

        return trades_created

    def create_trade(self, pair: str) -> bool:

        analyzed_df, _ = self.dataprovider.get_analyzed_dataframe(pair, self.strategy.timeframe)
        nowtime = analyzed_df.iloc[-1]['date'] if len(analyzed_df) > 0 else None
        if self.strategy.is_pair_locked(pair, nowtime):
            lock = PairLocks.get_pair_longest_lock(pair, nowtime)
            if lock:
                None
            else:
                pass
            return False

        if not self.get_free_open_trades():
            return False

        (buy, sell, buy_tag, _) = self.strategy.get_signal(
            pair,
            self.strategy.timeframe,
            analyzed_df
        )

        if buy and not sell:
            stake_amount = self.wallets.get_trade_stake_amount(pair, self.edge)

            bid_check_dom = self.config.get('bid_strategy', {}).get('check_depth_of_market', {})
            if ((bid_check_dom.get('enabled', False)) and
                    (bid_check_dom.get('bids_to_ask_delta', 0) > 0)):
                if self._check_depth_of_market_buy(pair, bid_check_dom):
                    return self.execute_entry(pair, stake_amount, buy_tag=buy_tag)
                else:
                    return False

            return self.execute_entry(pair, stake_amount, buy_tag=buy_tag)
        else:
            return False

    def process_open_trade_positions(self):
        for trade in Trade.get_open_trades():
            if trade.open_order_id is None:
                try:
                    self.check_and_call_adjust_trade_position(trade)
                except :
                    pass

    def check_and_call_adjust_trade_position(self, trade: Trade):
        if self.strategy.max_entry_position_adjustment > -1:
            count_of_buys = trade.nr_of_successful_buys
            if count_of_buys > self.strategy.max_entry_position_adjustment:
                return
        else:
            pass
        current_rate = self.exchange.get_rate(trade.pair, refresh=True, side="buy")
        current_profit = trade.calc_profit_ratio(current_rate)

        min_stake_amount = self.exchange.get_min_pair_stake_amount(trade.pair,
                                                                   current_rate,
                                                                   self.strategy.stoploss)
        max_stake_amount = self.wallets.get_available_stake_amount()
        stake_amount = strategy_safe_wrapper(self.strategy.adjust_trade_position,
                                             default_retval=None)(
            trade=trade, current_time=datetime.now(timezone.utc), current_rate=current_rate,
            current_profit=current_profit, min_stake=min_stake_amount, max_stake=max_stake_amount)

        if stake_amount is not None and stake_amount > 0.0:
            self.execute_entry(trade.pair, stake_amount, trade=trade)

        if stake_amount is not None and stake_amount < 0.0:
            pass

    def _check_depth_of_market_buy(self, pair: str, conf: Dict) -> bool:
        conf_bids_to_ask_delta = conf.get('bids_to_ask_delta', 0)
        order_book = self.exchange.fetch_l2_order_book(pair, 1000)
        order_book_data_frame = order_book_to_dataframe(order_book['bids'], order_book['asks'])
        order_book_bids = order_book_data_frame['b_size'].sum()
        order_book_asks = order_book_data_frame['a_size'].sum()
        bids_ask_delta = order_book_bids / order_book_asks

        if bids_ask_delta >= conf_bids_to_ask_delta:
            return True
        else:
            return False

    def execute_entry(self, pair: str, stake_amount: float, price: Optional[float] = None, *,
                      ordertype: Optional[str] = None, buy_tag: Optional[str] = None,
                      trade: Optional[Trade] = None) -> bool:

        pos_adjust = trade is not None

        enter_limit_requested, stake_amount = self.get_valid_enter_price_and_stake(
            pair, price, stake_amount, buy_tag, trade)

        if not stake_amount:
            return False



        amount = stake_amount / enter_limit_requested
        order_type = ordertype or self.strategy.order_types['buy']
        time_in_force = self.strategy.order_time_in_force['buy']

        if not pos_adjust and not strategy_safe_wrapper(
                self.strategy.confirm_trade_entry, default_retval=True)(
                pair=pair, order_type=order_type, amount=amount, rate=enter_limit_requested,
                time_in_force=time_in_force, current_time=datetime.now(timezone.utc),
                entry_tag=buy_tag):
            return False
        order = self.exchange.create_order(pair=pair, ordertype=order_type, side="buy",
                                           amount=amount, rate=enter_limit_requested,
                                           time_in_force=time_in_force)
        order_obj = Order.parse_from_ccxt_object(order, pair, 'buy')
        order_id = order['id']
        order_status = order.get('status', None)
        enter_limit_filled_price = enter_limit_requested
        amount_requested = amount

        if order_status == 'expired' or order_status == 'rejected':
            order_tif = self.strategy.order_time_in_force['buy']

            # return false if the order is not filled
            if float(order['filled']) == 0:
                               order_tif, order_type, pair, order_status, self.exchange.name


            else:
                stake_amount = order['cost']
                amount = safe_value_fallback(order, 'filled', 'amount')
                enter_limit_filled_price = safe_value_fallback(order, 'average', 'price')

        elif order_status == 'closed':
            stake_amount = order['cost']
            amount = safe_value_fallback(order, 'filled', 'amount')
            enter_limit_filled_price = safe_value_fallback(order, 'average', 'price')

        fee = self.exchange.get_fee(symbol=pair, taker_or_maker='maker')
        if trade is None:
            trade = Trade(
                pair=pair,
                stake_amount=stake_amount,
                amount=amount,
                is_open=True,
                amount_requested=amount_requested,
                fee_open=fee,
                fee_close=fee,
                open_rate=enter_limit_filled_price,
                open_rate_requested=enter_limit_requested,
                open_date=datetime.utcnow(),
                exchange=self.exchange.id,
                open_order_id=order_id,
                fee_open_currency=None,
                strategy=self.strategy.get_strategy_name(),
                buy_tag=buy_tag,
                timeframe=timeframe_to_minutes(self.config['timeframe'])
            )
        else:
            trade.is_open = True
            trade.fee_open_currency = None
            trade.open_rate_requested = enter_limit_requested
            trade.open_order_id = order_id

        trade.orders.append(order_obj)
        trade.recalc_trade_from_orders()
        Trade.query.session.add(trade)
        Trade.commit()

        self.wallets.update()

        self._notify_enter(trade, order, order_type)

        if pos_adjust:
            if order_status == 'closed':
                trade = self.cancel_stoploss_on_exchange(trade)
            else:
                pass

        if order_status == 'closed':
            self.update_trade_state(trade, order_id, order)

        return True

    def cancel_stoploss_on_exchange(self, trade: Trade) -> Trade:
        if self.strategy.order_types.get('stoploss_on_exchange') and trade.stoploss_order_id:
            try:
                co = self.exchange.cancel_stoploss_order_with_result(
                    trade.stoploss_order_id, trade.pair, trade.amount)
                trade.update_order(co)
            except InvalidOrderException:
                pass
        return trade

    def get_valid_enter_price_and_stake(
            self, pair: str, price: Optional[float], stake_amount: float,
            entry_tag: Optional[str],
            trade: Optional[Trade]) -> Tuple[float, float]:
        if price:
            enter_limit_requested = price
        else:
            proposed_enter_rate = self.exchange.get_rate(pair, refresh=True, side="buy")
            custom_entry_price = strategy_safe_wrapper(self.strategy.custom_entry_price,
                                                       default_retval=proposed_enter_rate)(
                pair=pair, current_time=datetime.now(timezone.utc),
                proposed_rate=proposed_enter_rate, entry_tag=entry_tag)

            enter_limit_requested = self.get_valid_price(custom_entry_price, proposed_enter_rate)
        if not enter_limit_requested:
            raise PricingError('Could not determine buy price.')
        min_stake_amount = self.exchange.get_min_pair_stake_amount(pair, enter_limit_requested,
                                                                   self.strategy.stoploss)
        if not self.edge and trade is None:
            max_stake_amount = self.wallets.get_available_stake_amount()
            stake_amount = strategy_safe_wrapper(self.strategy.custom_stake_amount,
                                                 default_retval=stake_amount)(
                pair=pair, current_time=datetime.now(timezone.utc),
                current_rate=enter_limit_requested, proposed_stake=stake_amount,
                min_stake=min_stake_amount, max_stake=max_stake_amount, entry_tag=entry_tag)
        stake_amount = self.wallets.validate_stake_amount(pair, stake_amount, min_stake_amount)
        return enter_limit_requested, stake_amount

    def _notify_enter(self, trade: Trade, order: Dict, order_type: Optional[str] = None,
                      fill: bool = False) -> None:
        open_rate = safe_value_fallback(order, 'average', 'price')
        if open_rate is None:
            open_rate = trade.open_rate

        current_rate = trade.open_rate_requested
        if self.dataprovider.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            current_rate = self.exchange.get_rate(trade.pair, refresh=False, side="buy")

        msg = {
            'trade_id': trade.id,
            'type': RPCMessageType.BUY_FILL if fill else RPCMessageType.BUY,
            'buy_tag': trade.buy_tag,
            'exchange': self.exchange.name.capitalize(),
            'pair': trade.pair,
            'limit': open_rate,  # Deprecated (?)
            'open_rate': open_rate,
            'order_type': order_type,
            'stake_amount': trade.stake_amount,
            'stake_currency': self.config['stake_currency'],
            'fiat_currency': self.config.get('fiat_display_currency', None),
            'amount': safe_value_fallback(order, 'filled', 'amount') or trade.amount,
            'open_date': trade.open_date or datetime.utcnow(),
            'current_rate': current_rate,
        }

        self.rpc.send_msg(msg)

    def _notify_enter_cancel(self, trade: Trade, order_type: str, reason: str) -> None:
        current_rate = self.exchange.get_rate(trade.pair, refresh=False, side="buy")

        msg = {
            'trade_id': trade.id,
            'type': RPCMessageType.BUY_CANCEL,
            'buy_tag': trade.buy_tag,
            'exchange': self.exchange.name.capitalize(),
            'pair': trade.pair,
            'limit': trade.open_rate,
            'order_type': order_type,
            'stake_amount': trade.stake_amount,
            'stake_currency': self.config['stake_currency'],
            'fiat_currency': self.config.get('fiat_display_currency', None),
            'amount': trade.amount,
            'open_date': trade.open_date,
            'current_rate': current_rate,
            'reason': reason,
        }

        self.rpc.send_msg(msg)


    def exit_positions(self, trades: List[Any]) -> int:
        trades_closed = 0
        for trade in trades:
            try:

                if (self.strategy.order_types.get('stoploss_on_exchange') and
                        self.handle_stoploss_on_exchange(trade)):
                    trades_closed += 1
                    Trade.commit()
                    continue
                if trade.open_order_id is None and trade.is_open and self.handle_trade(trade):
                    trades_closed += 1

            except :
                pass

        if trades_closed:
            self.wallets.update()

        return trades_closed

    def handle_trade(self, trade: Trade) -> bool:
        if not trade.is_open:
            raise DependencyException(f'Attempt to handle closed trade: {trade}')


        (buy, sell) = (False, False)
        exit_tag = None

        if (self.config.get('use_sell_signal', True) or
                self.config.get('ignore_roi_if_buy_signal', False)):
            analyzed_df, _ = self.dataprovider.get_analyzed_dataframe(trade.pair,
                                                                      self.strategy.timeframe)

            (buy, sell, _, exit_tag) = self.strategy.get_signal(
                trade.pair,
                self.strategy.timeframe,
                analyzed_df
            )

        sell_rate = self.exchange.get_rate(trade.pair, refresh=True, side="sell")
        if self._check_and_execute_exit(trade, sell_rate, buy, sell, exit_tag):
            return True

        return False

    def create_stoploss_order(self, trade: Trade, stop_price: float) -> bool:
        try:
            stoploss_order = self.exchange.stoploss(pair=trade.pair, amount=trade.amount,
                                                    stop_price=stop_price,
                                                    order_types=self.strategy.order_types)

            order_obj = Order.parse_from_ccxt_object(stoploss_order, trade.pair, 'stoploss')
            trade.orders.append(order_obj)
            trade.stoploss_order_id = str(stoploss_order['id'])
            return True

            self.handle_insufficient_funds(trade)

        except InvalidOrderException as e:
            trade.stoploss_order_id = None
            self.execute_trade_exit(trade, trade.stop_loss, sell_reason=SellCheckTuple(
                sell_type=SellType.EMERGENCY_SELL))

        except ExchangeError:
            trade.stoploss_order_id = None
        return False

    def handle_stoploss_on_exchange(self, trade: Trade) -> bool:

        stoploss_order = None

        try:
            stoploss_order = self.exchange.fetch_stoploss_order(
                trade.stoploss_order_id, trade.pair) if trade.stoploss_order_id else None
        except :
            pass

        if stoploss_order:
            trade.update_order(stoploss_order)

        if stoploss_order and stoploss_order['status'] in ('closed', 'triggered'):
            trade.sell_reason = SellType.STOPLOSS_ON_EXCHANGE.value
            self.update_trade_state(trade, trade.stoploss_order_id, stoploss_order,
                                    stoploss_order=True)
            self.strategy.lock_pair(trade.pair, datetime.now(timezone.utc),
                                    reason='Auto lock')
            self._notify_exit(trade, "stoploss")
            return True

        if trade.open_order_id or not trade.is_open:
            return False

        if not stoploss_order:
            stoploss = self.edge.stoploss(pair=trade.pair) if self.edge else self.strategy.stoploss
            stop_price = trade.open_rate * (1 + stoploss)

            if self.create_stoploss_order(trade=trade, stop_price=stop_price):
                trade.stoploss_last_update = datetime.utcnow()
                return False

        if (trade.is_open
                and stoploss_order
                and stoploss_order['status'] in ('canceled', 'cancelled')):
            if self.create_stoploss_order(trade=trade, stop_price=trade.stop_loss):
                return False
            else:
                trade.stoploss_order_id = None
        if (
            trade.is_open and stoploss_order
            and stoploss_order.get('status_stop') != 'triggered'
            and (self.config.get('trailing_stop', False)
                 or self.config.get('use_custom_stoploss', False))
        ):
            self.handle_trailing_stoploss_on_exchange(trade, stoploss_order)

        return False

    def handle_trailing_stoploss_on_exchange(self, trade: Trade, order: Dict) -> None:
        stoploss_norm = self.exchange.price_to_precision(trade.pair, trade.stop_loss)

        if self.exchange.stoploss_adjust(stoploss_norm, order):
            update_beat = self.strategy.order_types.get('stoploss_on_exchange_interval', 60)
            if (datetime.utcnow() - trade.stoploss_last_update).total_seconds() >= update_beat:
                try:
                    co = self.exchange.cancel_stoploss_order_with_result(order['id'], trade.pair,
                                                                         trade.amount)
                    trade.update_order(co)
                except:
                    pass



    def _check_and_execute_exit(self, trade: Trade, exit_rate: float,
                                buy: bool, sell: bool, exit_tag: Optional[str]) -> bool:
        should_sell = self.strategy.should_sell(
            trade, exit_rate, datetime.now(timezone.utc), buy, sell,
            force_stoploss=self.edge.stoploss(trade.pair) if self.edge else 0
        )

        if should_sell.sell_flag:
            self.execute_trade_exit(trade, exit_rate, should_sell, exit_tag=exit_tag)
            return True
        return False

    def check_handle_timedout(self) -> None:


        for trade in Trade.get_open_order_trades():
            try:
                if not trade.open_order_id:
                    continue
                order = self.exchange.fetch_order(trade.open_order_id, trade.pair)
            except (ExchangeError):
                continue

            fully_cancelled = self.update_trade_state(trade, trade.open_order_id, order)

            order_obj = trade.select_order_by_order_id(trade.open_order_id)

            if (order['side'] == 'buy' and (order['status'] == 'open' or fully_cancelled) and (
                    fully_cancelled
                    or (order_obj and self.strategy.ft_check_timed_out(
                        'buy', trade, order_obj, datetime.now(timezone.utc))
                        ))):
                self.handle_cancel_enter(trade, order, constants.CANCEL_REASON['TIMEOUT'])

            elif (order['side'] == 'sell' and (order['status'] == 'open' or fully_cancelled) and (
                  fully_cancelled
                  or (order_obj and self.strategy.ft_check_timed_out(
                      'sell', trade, order_obj, datetime.now(timezone.utc))
                      ))):
                canceled = self.handle_cancel_exit(trade, order, constants.CANCEL_REASON['TIMEOUT'])
                canceled_count = trade.get_exit_order_count()
                max_timeouts = self.config.get('unfilledtimeout', {}).get('exit_timeout_count', 0)
                if canceled and max_timeouts > 0 and canceled_count >= max_timeouts:
                    try:
                        self.execute_trade_exit(
                            trade, order.get('price'),
                            sell_reason=SellCheckTuple(sell_type=SellType.EMERGENCY_SELL))
                    except DependencyException as exception:
                        pass

    def cancel_all_open_orders(self) -> None:

        for trade in Trade.get_open_order_trades():
            try:
                order = self.exchange.fetch_order(trade.open_order_id, trade.pair)
            except (ExchangeError):
                continue

            if order['side'] == 'buy':
                self.handle_cancel_enter(trade, order, constants.CANCEL_REASON['ALL_CANCELLED'])

            elif order['side'] == 'sell':
                self.handle_cancel_exit(trade, order, constants.CANCEL_REASON['ALL_CANCELLED'])
        Trade.commit()

    def handle_cancel_enter(self, trade: Trade, order: Dict, reason: str) -> bool:
        was_trade_fully_canceled = False

        if order['status'] not in constants.NON_OPEN_EXCHANGE_STATES:
            filled_val: float = order.get('filled', 0.0) or 0.0
            filled_stake = filled_val * trade.open_rate
            minstake = self.exchange.get_min_pair_stake_amount(
                trade.pair, trade.open_rate, self.strategy.stoploss)

            if filled_val > 0 and minstake and filled_stake < minstake:
                return False
            corder = self.exchange.cancel_order_with_result(trade.open_order_id, trade.pair,
                                                            trade.amount)
            if corder.get('status') not in constants.NON_OPEN_EXCHANGE_STATES:
                return False
        else:
            corder = order
            reason = constants.CANCEL_REASON['CANCELLED_ON_EXCHANGE']
        filled_amount = safe_value_fallback2(corder, order, 'filled', 'filled')
        if isclose(filled_amount, 0.0, abs_tol=constants.MATH_CLOSE_PREC):
            if len(trade.orders) <= 1:
                trade.delete()
                was_trade_fully_canceled = True
                reason += f", {constants.CANCEL_REASON['FULLY_CANCELLED']}"
            else:
                self.update_trade_state(trade, trade.open_order_id, corder)
                trade.open_order_id = None
        else:
            trade.amount = filled_amount
            trade.stake_amount = trade.amount * trade.open_rate
            self.update_trade_state(trade, trade.open_order_id, corder)

            trade.open_order_id = None
            reason += f", {constants.CANCEL_REASON['PARTIALLY_FILLED']}"

        self.wallets.update()
        self._notify_enter_cancel(trade, order_type=self.strategy.order_types['buy'],
                                  reason=reason)
        return was_trade_fully_canceled

    def handle_cancel_exit(self, trade: Trade, order: Dict, reason: str) -> bool:
        cancelled = False
        if order['remaining'] == order['amount'] or order.get('filled') == 0.0:
            if not self.exchange.check_order_canceled_empty(order):
                try:
                    co = self.exchange.cancel_order_with_result(trade.open_order_id, trade.pair,
                                                                trade.amount)
                    trade.update_order(co)
                except InvalidOrderException:
                    return False
            else:
                reason = constants.CANCEL_REASON['CANCELLED_ON_EXCHANGE']
                trade.update_order(order)

            trade.close_rate = None
            trade.close_rate_requested = None
            trade.close_profit = None
            trade.close_profit_abs = None
            trade.close_date = None
            trade.is_open = True
            trade.open_order_id = None
            trade.sell_reason = None
            cancelled = True
        else:
            reason = constants.CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
            cancelled = False

        self.wallets.update()
        self._notify_exit_cancel(
            trade,
            order_type=self.strategy.order_types['sell'],
            reason=reason
        )
        return cancelled

    def _safe_exit_amount(self, pair: str, amount: float) -> float:
        self.wallets.update()
        trade_base_currency = self.exchange.get_pair_base_currency(pair)
        wallet_amount = self.wallets.get_free(trade_base_currency)
        if wallet_amount >= amount:
            return amount
        elif wallet_amount > amount * 0.98:
            return wallet_amount
        else:
            pass

    def execute_trade_exit(
            self,
            trade: Trade,
            limit: float,
            sell_reason: SellCheckTuple,
            *,
            exit_tag: Optional[str] = None,
            ordertype: Optional[str] = None,
            ) -> bool:
        sell_type = 'sell'
        if sell_reason.sell_type in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
            sell_type = 'stoploss'
        if (self.config['dry_run'] and sell_type == 'stoploss'
                and self.strategy.order_types['stoploss_on_exchange']):
            limit = trade.stop_loss

        proposed_limit_rate = limit
        current_profit = trade.calc_profit_ratio(limit)
        custom_exit_price = strategy_safe_wrapper(self.strategy.custom_exit_price,
                                                  default_retval=proposed_limit_rate)(
            pair=trade.pair, trade=trade,
            current_time=datetime.now(timezone.utc),
            proposed_rate=proposed_limit_rate, current_profit=current_profit)

        limit = self.get_valid_price(custom_exit_price, proposed_limit_rate)

        trade = self.cancel_stoploss_on_exchange(trade)

        order_type = ordertype or self.strategy.order_types[sell_type]
        if sell_reason.sell_type == SellType.EMERGENCY_SELL:
            order_type = self.strategy.order_types.get("emergencysell", "market")

        amount = self._safe_exit_amount(trade.pair, trade.amount)
        time_in_force = self.strategy.order_time_in_force['sell']

        if not strategy_safe_wrapper(self.strategy.confirm_trade_exit, default_retval=True)(
                pair=trade.pair, trade=trade, order_type=order_type, amount=amount, rate=limit,
                time_in_force=time_in_force, sell_reason=sell_reason.sell_reason,
                current_time=datetime.now(timezone.utc)):
            return False

        try:
            order = self.exchange.create_order(pair=trade.pair,
                                               ordertype=order_type, side="sell",
                                               amount=amount, rate=limit,
                                               time_in_force=time_in_force
                                               )
        except InsufficientFundsError as e:
            self.handle_insufficient_funds(trade)
            return False

        order_obj = Order.parse_from_ccxt_object(order, trade.pair, 'sell')
        trade.orders.append(order_obj)

        trade.open_order_id = order['id']
        trade.sell_order_status = ''
        trade.close_rate_requested = limit
        trade.sell_reason = exit_tag or sell_reason.sell_reason

        self.strategy.lock_pair(trade.pair, datetime.now(timezone.utc),
                                reason='Auto lock')

        self._notify_exit(trade, order_type)
        if order.get('status', 'unknown') in ('closed', 'expired'):
            self.update_trade_state(trade, trade.open_order_id, order)
        Trade.commit()

        return True

    def _notify_exit(self, trade: Trade, order_type: str, fill: bool = False) -> None:
        profit_rate = trade.close_rate if trade.close_rate else trade.close_rate_requested
        profit_trade = trade.calc_profit(rate=profit_rate)
        current_rate = self.exchange.get_rate(
            trade.pair, refresh=False, side="sell") if not fill else None
        profit_ratio = trade.calc_profit_ratio(profit_rate)
        gain = "profit" if profit_ratio > 0 else "loss"

        msg = {
            'type': (RPCMessageType.SELL_FILL if fill
                     else RPCMessageType.SELL),
            'trade_id': trade.id,
            'exchange': trade.exchange.capitalize(),
            'pair': trade.pair,
            'gain': gain,
            'limit': profit_rate,
            'order_type': order_type,
            'amount': trade.amount,
            'open_rate': trade.open_rate,
            'close_rate': trade.close_rate,
            'current_rate': current_rate,
            'profit_amount': profit_trade,
            'profit_ratio': profit_ratio,
            'buy_tag': trade.buy_tag,
            'sell_reason': trade.sell_reason,
            'open_date': trade.open_date,
            'close_date': trade.close_date or datetime.utcnow(),
            'stake_currency': self.config['stake_currency'],
            'fiat_currency': self.config.get('fiat_display_currency', None),
        }

        if 'fiat_display_currency' in self.config:
            msg.update({
                'fiat_currency': self.config['fiat_display_currency'],
            })

        self.rpc.send_msg(msg)

    def _notify_exit_cancel(self, trade: Trade, order_type: str, reason: str) -> None:
        if trade.sell_order_status == reason:
            return
        else:
            trade.sell_order_status = reason

        profit_rate = trade.close_rate if trade.close_rate else trade.close_rate_requested
        profit_trade = trade.calc_profit(rate=profit_rate)
        current_rate = self.exchange.get_rate(trade.pair, refresh=False, side="sell")
        profit_ratio = trade.calc_profit_ratio(profit_rate)
        gain = "profit" if profit_ratio > 0 else "loss"

        msg = {
            'type': RPCMessageType.SELL_CANCEL,
            'trade_id': trade.id,
            'exchange': trade.exchange.capitalize(),
            'pair': trade.pair,
            'gain': gain,
            'limit': profit_rate or 0,
            'order_type': order_type,
            'amount': trade.amount,
            'open_rate': trade.open_rate,
            'current_rate': current_rate,
            'profit_amount': profit_trade,
            'profit_ratio': profit_ratio,
            'buy_tag': trade.buy_tag,
            'sell_reason': trade.sell_reason,
            'open_date': trade.open_date,
            'close_date': trade.close_date or datetime.now(timezone.utc),
            'stake_currency': self.config['stake_currency'],
            'fiat_currency': self.config.get('fiat_display_currency', None),
            'reason': reason,
        }

        if 'fiat_display_currency' in self.config:
            msg.update({
                'fiat_currency': self.config['fiat_display_currency'],
            })

        self.rpc.send_msg(msg)

    def update_trade_state(self, trade: Trade, order_id: str, action_order: Dict[str, Any] = None,
                           stoploss_order: bool = False, send_msg: bool = True) -> bool:
        if not order_id:
            return False

        try:
            order = action_order or self.exchange.fetch_order_or_stoploss_order(order_id,
                                                                                trade.pair,
                                                                                stoploss_order)
        except InvalidOrderException as exception:
            return False

        trade.update_order(order)

        if self.exchange.check_order_canceled_empty(order):
            return True

        order_obj = trade.select_order_by_order_id(order_id)
        if not order_obj:
            raise DependencyException(
                f"Order_obj not found for {order_id}. This should not have happened.")
        self.handle_order_fee(trade, order_obj, order)

        trade.update_trade(order_obj)
        trade.recalc_trade_from_orders()
        Trade.commit()

        if order['status'] in constants.NON_OPEN_EXCHANGE_STATES:
            if order.get('side', None) == 'buy':
                trade = self.cancel_stoploss_on_exchange(trade)
            self.wallets.update()

        if not trade.is_open:
            if send_msg and not stoploss_order and not trade.open_order_id:
                self._notify_exit(trade, '', True)
            self.handle_protections(trade.pair)
        elif send_msg and not trade.open_order_id:
            self._notify_enter(trade, order, fill=True)

        return False

    def handle_protections(self, pair: str) -> None:
        prot_trig = self.protections.stop_per_pair(pair)
        if prot_trig:
            msg = {'type': RPCMessageType.PROTECTION_TRIGGER, }
            msg.update(prot_trig.to_json())
            self.rpc.send_msg(msg)

        prot_trig_glb = self.protections.global_stop()
        if prot_trig_glb:
            msg = {'type': RPCMessageType.PROTECTION_TRIGGER_GLOBAL, }
            msg.update(prot_trig_glb.to_json())
            self.rpc.send_msg(msg)

    def apply_fee_conditional(self, trade: Trade, trade_base_currency: str,
                              amount: float, fee_abs: float) -> float:
        self.wallets.update()
        if fee_abs != 0 and self.wallets.get_free(trade_base_currency) >= amount:
            pass
        elif fee_abs != 0:

            real_amount = self.exchange.amount_to_precision(trade.pair, amount - fee_abs)
            return real_amount
        return amount

    def handle_order_fee(self, trade: Trade, order_obj: Order, order: Dict[str, Any]) -> None:
        try:
            new_amount = self.get_real_amount(trade, order)
            if not isclose(safe_value_fallback(order, 'filled', 'amount'), new_amount,
                           abs_tol=constants.MATH_CLOSE_PREC):
                order_obj.ft_fee_base = trade.amount - new_amount
        except :
            pass

    def get_real_amount(self, trade: Trade, order: Dict) -> float:
        order_amount = safe_value_fallback(order, 'filled', 'amount')
        if trade.fee_updated(order.get('side', '')) or order['status'] == 'open':
            return order_amount

        trade_base_currency = self.exchange.get_pair_base_currency(trade.pair)
        if self.exchange.order_has_fee(order):
            fee_cost, fee_currency, fee_rate = self.exchange.extract_cost_curr_rate(order)
            if fee_rate is None or fee_rate < 0.02:
                trade.update_fee(fee_cost, fee_currency, fee_rate, order.get('side', ''))
                if trade_base_currency == fee_currency:
                    return self.apply_fee_conditional(trade, trade_base_currency,
                                                      amount=order_amount, fee_abs=fee_cost)
                return order_amount
        return self.fee_detection_from_trades(trade, order, order_amount, order.get('trades', []))

    def fee_detection_from_trades(self, trade: Trade, order: Dict, order_amount: float,
                                  trades: List) -> float:
        if not trades:
            trades = self.exchange.get_trades_for_order(
                self.exchange.get_order_id_conditional(order), trade.pair, trade.open_date)

        if len(trades) == 0:
            return order_amount
        fee_currency = None
        amount = 0
        fee_abs = 0.0
        fee_cost = 0.0
        trade_base_currency = self.exchange.get_pair_base_currency(trade.pair)
        fee_rate_array: List[float] = []
        for exectrade in trades:
            amount += exectrade['amount']
            if self.exchange.order_has_fee(exectrade):
                fee_cost_, fee_currency, fee_rate_ = self.exchange.extract_cost_curr_rate(exectrade)
                fee_cost += fee_cost_
                if fee_rate_ is not None:
                    fee_rate_array.append(fee_rate_)
                if trade_base_currency == fee_currency:
                    fee_abs += fee_cost_
        if fee_currency:
            fee_rate = sum(fee_rate_array) / float(len(fee_rate_array)) if fee_rate_array else None
            if fee_rate is not None and fee_rate < 0.02:
                trade.update_fee(fee_cost, fee_currency, fee_rate, order.get('side', ''))

        if not isclose(amount, order_amount, abs_tol=constants.MATH_CLOSE_PREC):
            raise DependencyException("Half bought? Amounts don't match")

        if fee_abs != 0:
            return self.apply_fee_conditional(trade, trade_base_currency,
                                              amount=amount, fee_abs=fee_abs)
        else:
            return amount

    def get_valid_price(self, custom_price: float, proposed_price: float) -> float:
        if custom_price:
            try:
                valid_custom_price = float(custom_price)
            except ValueError:
                valid_custom_price = proposed_price
        else:
            valid_custom_price = proposed_price

        cust_p_max_dist_r = self.config.get('custom_price_max_distance_ratio', 0.02)
        min_custom_price_allowed = proposed_price - (proposed_price * cust_p_max_dist_r)
        max_custom_price_allowed = proposed_price + (proposed_price * cust_p_max_dist_r)
        return max(
            min(valid_custom_price, max_custom_price_allowed),
            min_custom_price_allowed)
