"""
SQLite로 거래를 지속 관리
"""
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, String,
                        create_engine, desc, func, inspect)
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.orm import Query, declarative_base, relationship, scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql.schema import UniqueConstraint

from threee.constants import DATETIME_PRINT_FORMAT, NON_OPEN_EXCHANGE_STATES
from threee.enums import SellType
from threee.exceptions import DependencyException, OperationalException
from threee.persistence.migrations import check_migrate


logger = logging.getLogger(__name__)


_DECL_BASE: Any = declarative_base()
_SQL_DOCS_URL = 'http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls'


def init_db(db_url: str, clean_open_orders: bool = False) -> None:
    """
    주어진 conf로 이 모듈을 초기화
    """
    kwargs = {}

    if db_url == 'sqlite:///':
        raise OperationalException(
            f'Bad db-url {db_url}. For in-memory database, please use `sqlite://`.')
    if db_url == 'sqlite://':
        kwargs.update({
            'poolclass': StaticPool,
        })
    if db_url.startswith('sqlite://'):
        kwargs.update({
            'connect_args': {'check_same_thread': False},
        })

    try:
        engine = create_engine(db_url, future=True, **kwargs)
    except NoSuchModuleError:
        pass
    Trade._session = scoped_session(sessionmaker(bind=engine, autoflush=True))
    Trade.query = Trade._session.query_property()
    Order.query = Trade._session.query_property()
    PairLock.query = Trade._session.query_property()

    previous_tables = inspect(engine).get_table_names()
    _DECL_BASE.metadata.create_all(engine)
    check_migrate(engine, decl_base=_DECL_BASE, previous_tables=previous_tables)

    if clean_open_orders and db_url != 'sqlite://':
        clean_dry_run_db()


def cleanup_db() -> None:
    Trade.commit()


def clean_dry_run_db() -> None:
    for trade in Trade.query.filter(Trade.open_order_id.isnot(None)).all():
        if 'dry_run' in trade.open_order_id:
            trade.open_order_id = None
    Trade.commit()


class Order(_DECL_BASE):
    """
    주문 데이터베이스 모델
    """
    __tablename__ = 'orders'
    __table_args__ = (UniqueConstraint('ft_pair', 'order_id', name="_order_pair_order_id"),)

    id = Column(Integer, primary_key=True)
    ft_trade_id = Column(Integer, ForeignKey('trades.id'), index=True)

    trade = relationship("Trade", back_populates="orders")

    ft_order_side: str = Column(String(25), nullable=False)
    ft_pair: str = Column(String(25), nullable=False)
    ft_is_open = Column(Boolean, nullable=False, default=True, index=True)

    order_id: str = Column(String(255), nullable=False, index=True)
    status = Column(String(255), nullable=True)
    symbol = Column(String(25), nullable=True)
    order_type: str = Column(String(50), nullable=True)
    side = Column(String(25), nullable=True)
    price = Column(Float, nullable=True)
    average = Column(Float, nullable=True)
    amount = Column(Float, nullable=True)
    filled = Column(Float, nullable=True)
    remaining = Column(Float, nullable=True)
    cost = Column(Float, nullable=True)
    order_date = Column(DateTime, nullable=True, default=datetime.utcnow)
    order_filled_date = Column(DateTime, nullable=True)
    order_update_date = Column(DateTime, nullable=True)

    ft_fee_base = Column(Float, nullable=True)

    @property
    def order_date_utc(self) -> datetime:
        return self.order_date.replace(tzinfo=timezone.utc)

    @property
    def safe_price(self) -> float:
        return self.average or self.price

    @property
    def safe_filled(self) -> float:
        return self.filled or self.amount or 0.0

    @property
    def safe_fee_base(self) -> float:
        return self.ft_fee_base or 0.0

    @property
    def safe_amount_after_fee(self) -> float:
        return self.safe_filled - self.safe_fee_base

    def __repr__(self):

        return (f'Order(id={self.id}, order_id={self.order_id}, trade_id={self.ft_trade_id}, '
                f'side={self.side}, order_type={self.order_type}, status={self.status})')

    def update_from_ccxt_object(self, order):
        if self.order_id != str(order['id']):
            raise DependencyException("Order-id's don't match")

        self.status = order.get('status', self.status)
        self.symbol = order.get('symbol', self.symbol)
        self.order_type = order.get('type', self.order_type)
        self.side = order.get('side', self.side)
        self.price = order.get('price', self.price)
        self.amount = order.get('amount', self.amount)
        self.filled = order.get('filled', self.filled)
        self.average = order.get('average', self.average)
        self.remaining = order.get('remaining', self.remaining)
        self.cost = order.get('cost', self.cost)
        if 'timestamp' in order and order['timestamp'] is not None:
            self.order_date = datetime.fromtimestamp(order['timestamp'] / 1000, tz=timezone.utc)

        self.ft_is_open = True
        if self.status in NON_OPEN_EXCHANGE_STATES:
            self.ft_is_open = False
            if (order.get('filled', 0.0) or 0.0) > 0:
                self.order_filled_date = datetime.now(timezone.utc)
        self.order_update_date = datetime.now(timezone.utc)

    def to_json(self) -> Dict[str, Any]:
        return {
            'pair': self.ft_pair,
            'order_id': self.order_id,
            'status': self.status,
            'amount': self.amount,
            'average': round(self.average, 8) if self.average else 0,
            'safe_price': self.safe_price,
            'cost': self.cost if self.cost else 0,
            'filled': self.filled,
            'ft_order_side': self.ft_order_side,
            'is_open': self.ft_is_open,
            'order_date': self.order_date.strftime(DATETIME_PRINT_FORMAT)
            if self.order_date else None,
            'order_timestamp': int(self.order_date.replace(
                tzinfo=timezone.utc).timestamp() * 1000) if self.order_date else None,
            'order_filled_date': self.order_filled_date.strftime(DATETIME_PRINT_FORMAT)
            if self.order_filled_date else None,
            'order_filled_timestamp': int(self.order_filled_date.replace(
                tzinfo=timezone.utc).timestamp() * 1000) if self.order_filled_date else None,
            'order_type': self.order_type,
            'price': self.price,
            'remaining': self.remaining,
        }

    def close_bt_order(self, close_date: datetime):
        self.order_filled_date = close_date
        self.filled = self.amount
        self.status = 'closed'
        self.ft_is_open = False

    @staticmethod
    def update_orders(orders: List['Order'], order: Dict[str, Any]):
        if not isinstance(order, dict):
            return

        filtered_orders = [o for o in orders if o.order_id == order.get('id')]
        if filtered_orders:
            oobj = filtered_orders[0]
            oobj.update_from_ccxt_object(order)
            Order.query.session.commit()
        else:
            pass

    @staticmethod
    def parse_from_ccxt_object(order: Dict[str, Any], pair: str, side: str) -> 'Order':
        o = Order(order_id=str(order['id']), ft_order_side=side, ft_pair=pair)

        o.update_from_ccxt_object(order)
        return o

    @staticmethod
    def get_open_orders() -> List['Order']:
        return Order.query.filter(Order.ft_is_open.is_(True)).all()


class LocalTrade():
    use_db: bool = False
    trades: List['LocalTrade'] = []
    trades_open: List['LocalTrade'] = []
    total_profit: float = 0

    id: int = 0

    orders: List[Order] = []

    exchange: str = ''
    pair: str = ''
    is_open: bool = True
    fee_open: float = 0.0
    fee_open_cost: Optional[float] = None
    fee_open_currency: str = ''
    fee_close: float = 0.0
    fee_close_cost: Optional[float] = None
    fee_close_currency: str = ''
    open_rate: float = 0.0
    open_rate_requested: Optional[float] = None
    open_trade_value: float = 0.0
    close_rate: Optional[float] = None
    close_rate_requested: Optional[float] = None
    close_profit: Optional[float] = None
    close_profit_abs: Optional[float] = None
    stake_amount: float = 0.0
    amount: float = 0.0
    amount_requested: Optional[float] = None
    open_date: datetime
    close_date: Optional[datetime] = None
    open_order_id: Optional[str] = None
    stop_loss: float = 0.0
    stop_loss_pct: float = 0.0
    initial_stop_loss: float = 0.0
    initial_stop_loss_pct: Optional[float] = None
    stoploss_order_id: Optional[str] = None
    stoploss_last_update: Optional[datetime] = None
    max_rate: float = 0.0
    min_rate: float = 0.0
    sell_reason: str = ''
    sell_order_status: str = ''
    strategy: str = ''
    buy_tag: Optional[str] = None
    timeframe: Optional[int] = None

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.recalc_open_trade_value()

    def __repr__(self):
        open_since = self.open_date.strftime(DATETIME_PRINT_FORMAT) if self.is_open else 'closed'

        return (f'Trade(id={self.id}, pair={self.pair}, amount={self.amount:.8f}, '
                f'open_rate={self.open_rate:.8f}, open_since={open_since})')

    @property
    def open_date_utc(self):
        return self.open_date.replace(tzinfo=timezone.utc)

    @property
    def close_date_utc(self):
        return self.close_date.replace(tzinfo=timezone.utc)

    def to_json(self) -> Dict[str, Any]:
        filled_orders = self.select_filled_orders()
        orders = [order.to_json() for order in filled_orders]

        return {
            'trade_id': self.id,
            'pair': self.pair,
            'is_open': self.is_open,
            'exchange': self.exchange,
            'amount': round(self.amount, 8),
            'amount_requested': round(self.amount_requested, 8) if self.amount_requested else None,
            'stake_amount': round(self.stake_amount, 8),
            'strategy': self.strategy,
            'buy_tag': self.buy_tag,
            'timeframe': self.timeframe,

            'fee_open': self.fee_open,
            'fee_open_cost': self.fee_open_cost,
            'fee_open_currency': self.fee_open_currency,
            'fee_close': self.fee_close,
            'fee_close_cost': self.fee_close_cost,
            'fee_close_currency': self.fee_close_currency,

            'open_date': self.open_date.strftime(DATETIME_PRINT_FORMAT),
            'open_timestamp': int(self.open_date.replace(tzinfo=timezone.utc).timestamp() * 1000),
            'open_rate': self.open_rate,
            'open_rate_requested': self.open_rate_requested,
            'open_trade_value': round(self.open_trade_value, 8),

            'close_date': (self.close_date.strftime(DATETIME_PRINT_FORMAT)
                           if self.close_date else None),
            'close_timestamp': int(self.close_date.replace(
                tzinfo=timezone.utc).timestamp() * 1000) if self.close_date else None,
            'close_rate': self.close_rate,
            'close_rate_requested': self.close_rate_requested,
            'close_profit': self.close_profit,  # Deprecated
            'close_profit_pct': round(self.close_profit * 100, 2) if self.close_profit else None,
            'close_profit_abs': self.close_profit_abs,  # Deprecated

            'trade_duration_s': (int((self.close_date_utc - self.open_date_utc).total_seconds())
                                 if self.close_date else None),
            'trade_duration': (int((self.close_date_utc - self.open_date_utc).total_seconds() // 60)
                               if self.close_date else None),

            'profit_ratio': self.close_profit,
            'profit_pct': round(self.close_profit * 100, 2) if self.close_profit else None,
            'profit_abs': self.close_profit_abs,

            'sell_reason': self.sell_reason,
            'sell_order_status': self.sell_order_status,
            'stop_loss_abs': self.stop_loss,
            'stop_loss_ratio': self.stop_loss_pct if self.stop_loss_pct else None,
            'stop_loss_pct': (self.stop_loss_pct * 100) if self.stop_loss_pct else None,
            'stoploss_order_id': self.stoploss_order_id,
            'stoploss_last_update': (self.stoploss_last_update.strftime(DATETIME_PRINT_FORMAT)
                                     if self.stoploss_last_update else None),
            'stoploss_last_update_timestamp': int(self.stoploss_last_update.replace(
                tzinfo=timezone.utc).timestamp() * 1000) if self.stoploss_last_update else None,
            'initial_stop_loss_abs': self.initial_stop_loss,
            'initial_stop_loss_ratio': (self.initial_stop_loss_pct
                                        if self.initial_stop_loss_pct else None),
            'initial_stop_loss_pct': (self.initial_stop_loss_pct * 100
                                      if self.initial_stop_loss_pct else None),
            'min_rate': self.min_rate,
            'max_rate': self.max_rate,

            'open_order_id': self.open_order_id,
            'orders': orders,
        }

    @staticmethod
    def reset_trades() -> None:
        LocalTrade.trades = []
        LocalTrade.trades_open = []
        LocalTrade.total_profit = 0

    def adjust_min_max_rates(self, current_price: float, current_price_low: float) -> None:
        self.max_rate = max(current_price, self.max_rate or self.open_rate)
        self.min_rate = min(current_price_low, self.min_rate or self.open_rate)

    def _set_new_stoploss(self, new_loss: float, stoploss: float):
        self.stop_loss = new_loss
        self.stop_loss_pct = -1 * abs(stoploss)
        self.stoploss_last_update = datetime.utcnow()

    def adjust_stop_loss(self, current_price: float, stoploss: float,
                         initial: bool = False) -> None:
        if initial and not (self.stop_loss is None or self.stop_loss == 0):
            return

        new_loss = float(current_price * (1 - abs(stoploss)))

        if self.initial_stop_loss_pct is None:
            self._set_new_stoploss(new_loss, stoploss)
            self.initial_stop_loss = new_loss
            self.initial_stop_loss_pct = -1 * abs(stoploss)

        else:
            if new_loss > self.stop_loss:  # stop losses only walk up, never down!
                self._set_new_stoploss(new_loss, stoploss)
            else:
                pass

    def update_trade(self, order: Order) -> None:
        if order.status == 'open' or order.safe_price is None:
            return


        if order.ft_order_side == 'buy':
            self.open_rate = order.safe_price
            self.amount = order.safe_amount_after_fee
            if self.is_open:
                pass
            self.open_order_id = None
            self.recalc_trade_from_orders()
        elif order.ft_order_side == 'sell':
            if self.is_open:
                pass
            self.close(order.safe_price)
        elif order.ft_order_side == 'stoploss':
            self.stoploss_order_id = None
            self.close_rate_requested = self.stop_loss
            self.sell_reason = SellType.STOPLOSS_ON_EXCHANGE.value
            if self.is_open:
                pass
            self.close(order.safe_price)
        else:
            raise ValueError(f'Unknown order type: {order.order_type}')
        Trade.commit()

    def close(self, rate: float, *, show_msg: bool = True) -> None:
        self.close_rate = rate
        self.close_profit = self.calc_profit_ratio()
        self.close_profit_abs = self.calc_profit()
        self.close_date = self.close_date or datetime.utcnow()
        self.is_open = False
        self.sell_order_status = 'closed'
        self.open_order_id = None


    def update_fee(self, fee_cost: float, fee_currency: Optional[str], fee_rate: Optional[float],
                   side: str) -> None:
        if side == 'buy' and self.fee_open_currency is None:
            self.fee_open_cost = fee_cost
            self.fee_open_currency = fee_currency
            if fee_rate is not None:
                self.fee_open = fee_rate
                self.fee_close = fee_rate
        elif side == 'sell' and self.fee_close_currency is None:
            self.fee_close_cost = fee_cost
            self.fee_close_currency = fee_currency
            if fee_rate is not None:
                self.fee_close = fee_rate

    def fee_updated(self, side: str) -> bool:
        if side == 'buy':
            return self.fee_open_currency is not None
        elif side == 'sell':
            return self.fee_close_currency is not None
        else:
            return False

    def update_order(self, order: Dict) -> None:
        Order.update_orders(self.orders, order)

    def get_exit_order_count(self) -> int:
        return len([o for o in self.orders if o.ft_order_side == 'sell'])

    def _calc_open_trade_value(self) -> float:
        buy_trade = Decimal(self.amount) * Decimal(self.open_rate)
        fees = buy_trade * Decimal(self.fee_open)
        return float(buy_trade + fees)

    def recalc_open_trade_value(self) -> None:
        self.open_trade_value = self._calc_open_trade_value()

    def calc_close_trade_value(self, rate: Optional[float] = None,
                               fee: Optional[float] = None) -> float:

        if rate is None and not self.close_rate:
            return 0.0

        sell_trade = Decimal(self.amount) * Decimal(rate or self.close_rate)  # type: ignore
        fees = sell_trade * Decimal(fee or self.fee_close)
        return float(sell_trade - fees)

    def calc_profit(self, rate: Optional[float] = None,
                    fee: Optional[float] = None) -> float:
        close_trade_value = self.calc_close_trade_value(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )
        profit = close_trade_value - self.open_trade_value
        return float(f"{profit:.8f}")

    def calc_profit_ratio(self, rate: Optional[float] = None,
                          fee: Optional[float] = None) -> float:
        close_trade_value = self.calc_close_trade_value(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )
        if self.open_trade_value == 0.0:
            return 0.0
        profit_ratio = (close_trade_value / self.open_trade_value) - 1
        return float(f"{profit_ratio:.8f}")

    def recalc_trade_from_orders(self):

        if len(self.select_filled_orders('buy')) < 2:

            self.recalc_open_trade_value()
            return

        total_amount = 0.0
        total_stake = 0.0
        for o in self.orders:
            if (o.ft_is_open or
                    (o.ft_order_side != 'buy') or
                    (o.status not in NON_OPEN_EXCHANGE_STATES)):
                continue

            tmp_amount = o.safe_amount_after_fee
            tmp_price = o.average or o.price
            if o.filled is not None:
                tmp_amount = o.filled
            if tmp_amount > 0.0 and tmp_price is not None:
                total_amount += tmp_amount
                total_stake += tmp_price * tmp_amount

        if total_amount > 0:
            self.open_rate = total_stake / total_amount
            self.stake_amount = total_stake
            self.amount = total_amount
            self.fee_open_cost = self.fee_open * self.stake_amount
            self.recalc_open_trade_value()
            if self.stop_loss_pct is not None and self.open_rate is not None:
                self.adjust_stop_loss(self.open_rate, self.stop_loss_pct)

    def select_order_by_order_id(self, order_id: str) -> Optional[Order]:
        for o in self.orders:
            if o.order_id == order_id:
                return o
        return None

    def select_order(
            self, order_side: str = None, is_open: Optional[bool] = None) -> Optional[Order]:
        orders = self.orders
        if order_side:
            orders = [o for o in self.orders if o.ft_order_side == order_side]
        if is_open is not None:
            orders = [o for o in orders if o.ft_is_open == is_open]
        if len(orders) > 0:
            return orders[-1]
        else:
            return None

    def select_filled_orders(self, order_side: Optional[str] = None) -> List['Order']:
        return [o for o in self.orders if ((o.ft_order_side == order_side) or (order_side is None))
                and o.ft_is_open is False and
                (o.filled or 0) > 0 and
                o.status in NON_OPEN_EXCHANGE_STATES]

    @property
    def nr_of_successful_buys(self) -> int:
        """
        채워진 구매 주문의 수를 계산
        """

        return len(self.select_filled_orders('buy'))

    @property
    def nr_of_successful_sells(self) -> int:
        return len(self.select_filled_orders('sell'))

    @staticmethod
    def get_trades_proxy(*, pair: str = None, is_open: bool = None,
                         open_date: datetime = None, close_date: datetime = None,
                         ) -> List['LocalTrade']:
        if is_open is not None:
            if is_open:
                sel_trades = LocalTrade.trades_open
            else:
                sel_trades = LocalTrade.trades

        else:
            sel_trades = list(LocalTrade.trades + LocalTrade.trades_open)

        if pair:
            sel_trades = [trade for trade in sel_trades if trade.pair == pair]
        if open_date:
            sel_trades = [trade for trade in sel_trades if trade.open_date > open_date]
        if close_date:
            sel_trades = [trade for trade in sel_trades if trade.close_date
                          and trade.close_date > close_date]

        return sel_trades

    @staticmethod
    def close_bt_trade(trade):
        LocalTrade.trades_open.remove(trade)
        LocalTrade.trades.append(trade)
        LocalTrade.total_profit += trade.close_profit_abs

    @staticmethod
    def add_bt_trade(trade):
        if trade.is_open:
            LocalTrade.trades_open.append(trade)
        else:
            LocalTrade.trades.append(trade)

    @staticmethod
    def get_open_trades() -> List[Any]:
        return Trade.get_trades_proxy(is_open=True)

    @staticmethod
    def stoploss_reinitialization(desired_stoploss):
        for trade in Trade.get_open_trades():
            if (trade.stop_loss == trade.initial_stop_loss
                    and trade.initial_stop_loss_pct != desired_stoploss):

                trade.stop_loss = None
                trade.initial_stop_loss_pct = None
                trade.adjust_stop_loss(trade.open_rate, desired_stoploss)


class Trade(_DECL_BASE, LocalTrade):
    __tablename__ = 'trades'

    use_db: bool = True

    id = Column(Integer, primary_key=True)

    orders = relationship("Order", order_by="Order.id", cascade="all, delete-orphan", lazy="joined")

    exchange = Column(String(25), nullable=False)
    pair = Column(String(25), nullable=False, index=True)
    is_open = Column(Boolean, nullable=False, default=True, index=True)
    fee_open = Column(Float, nullable=False, default=0.0)
    fee_open_cost = Column(Float, nullable=True)
    fee_open_currency = Column(String(25), nullable=True)
    fee_close = Column(Float, nullable=False, default=0.0)
    fee_close_cost = Column(Float, nullable=True)
    fee_close_currency = Column(String(25), nullable=True)
    open_rate: float = Column(Float)
    open_rate_requested = Column(Float)
    open_trade_value = Column(Float)
    close_rate: Optional[float] = Column(Float)
    close_rate_requested = Column(Float)
    close_profit = Column(Float)
    close_profit_abs = Column(Float)
    stake_amount = Column(Float, nullable=False)
    amount = Column(Float)
    amount_requested = Column(Float)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    close_date = Column(DateTime)
    open_order_id = Column(String(255))
    stop_loss = Column(Float, nullable=True, default=0.0)
    stop_loss_pct = Column(Float, nullable=True)
    initial_stop_loss = Column(Float, nullable=True, default=0.0)
    initial_stop_loss_pct = Column(Float, nullable=True)
    stoploss_order_id = Column(String(255), nullable=True, index=True)
    stoploss_last_update = Column(DateTime, nullable=True)
    max_rate = Column(Float, nullable=True, default=0.0)
    min_rate = Column(Float, nullable=True)
    sell_reason = Column(String(100), nullable=True)
    sell_order_status = Column(String(100), nullable=True)
    strategy = Column(String(100), nullable=True)
    buy_tag = Column(String(100), nullable=True)
    timeframe = Column(Integer, nullable=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recalc_open_trade_value()

    def delete(self) -> None:

        for order in self.orders:
            Order.query.session.delete(order)

        Trade.query.session.delete(self)
        Trade.commit()

    @staticmethod
    def commit():
        Trade.query.session.commit()

    @staticmethod
    def get_trades_proxy(*, pair: str = None, is_open: bool = None,
                         open_date: datetime = None, close_date: datetime = None,
                         ) -> List['LocalTrade']:
        if Trade.use_db:
            trade_filter = []
            if pair:
                trade_filter.append(Trade.pair == pair)
            if open_date:
                trade_filter.append(Trade.open_date > open_date)
            if close_date:
                trade_filter.append(Trade.close_date > close_date)
            if is_open is not None:
                trade_filter.append(Trade.is_open.is_(is_open))
            return Trade.get_trades(trade_filter).all()
        else:
            return LocalTrade.get_trades_proxy(
                pair=pair, is_open=is_open,
                open_date=open_date,
                close_date=close_date
            )

    @staticmethod
    def get_trades(trade_filter=None) -> Query:
        if not Trade.use_db:
            pass
        if trade_filter is not None:
            if not isinstance(trade_filter, list):
                trade_filter = [trade_filter]
            return Trade.query.filter(*trade_filter)
        else:
            return Trade.query

    @staticmethod
    def get_open_order_trades() -> List['Trade']:
        """
        열려 있는 모든 거래를 반환
        """
        return Trade.get_trades(Trade.open_order_id.isnot(None)).all()

    @staticmethod
    def get_open_trades_without_assigned_fees():
        return Trade.get_trades([Trade.fee_open_currency.is_(None),
                                 Trade.orders.any(),
                                 Trade.is_open.is_(True),
                                 ]).all()

    @staticmethod
    def get_sold_trades_without_assigned_fees():
        return Trade.get_trades([Trade.fee_close_currency.is_(None),
                                 Trade.orders.any(),
                                 Trade.is_open.is_(False),
                                 ]).all()

    @staticmethod
    def get_total_closed_profit() -> float:
        if Trade.use_db:
            total_profit = Trade.query.with_entities(
                func.sum(Trade.close_profit_abs)).filter(Trade.is_open.is_(False)).scalar()
        else:
            total_profit = sum(
                t.close_profit_abs for t in LocalTrade.get_trades_proxy(is_open=False))
        return total_profit or 0

    @staticmethod
    def total_open_trades_stakes() -> float:
        if Trade.use_db:
            total_open_stake_amount = Trade.query.with_entities(
                func.sum(Trade.stake_amount)).filter(Trade.is_open.is_(True)).scalar()
        else:
            total_open_stake_amount = sum(
                t.stake_amount for t in LocalTrade.get_trades_proxy(is_open=True))
        return total_open_stake_amount or 0

    @staticmethod
    def get_overall_performance(minutes=None) -> List[Dict[str, Any]]:
        filters = [Trade.is_open.is_(False)]
        if minutes:
            start_date = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            filters.append(Trade.close_date >= start_date)
        pair_rates = Trade.query.with_entities(
            Trade.pair,
            func.sum(Trade.close_profit).label('profit_sum'),
            func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
            func.count(Trade.pair).label('count')
        ).filter(*filters)\
            .group_by(Trade.pair) \
            .order_by(desc('profit_sum_abs')) \
            .all()
        return [
            {
                'pair': pair,
                'profit_ratio': profit,
                'profit': round(profit * 100, 2),  # Compatibility mode
                'profit_pct': round(profit * 100, 2),
                'profit_abs': profit_abs,
                'count': count
            }
            for pair, profit, profit_abs, count in pair_rates
        ]

    @staticmethod
    def get_buy_tag_performance(pair: Optional[str]) -> List[Dict[str, Any]]:

        filters = [Trade.is_open.is_(False)]
        if(pair is not None):
            filters.append(Trade.pair == pair)

        buy_tag_perf = Trade.query.with_entities(
            Trade.buy_tag,
            func.sum(Trade.close_profit).label('profit_sum'),
            func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
            func.count(Trade.pair).label('count')
        ).filter(*filters)\
            .group_by(Trade.buy_tag) \
            .order_by(desc('profit_sum_abs')) \
            .all()

        return [
            {
                'buy_tag': buy_tag if buy_tag is not None else "Other",
                'profit_ratio': profit,
                'profit_pct': round(profit * 100, 2),
                'profit_abs': profit_abs,
                'count': count
            }
            for buy_tag, profit, profit_abs, count in buy_tag_perf
        ]

    @staticmethod
    def get_sell_reason_performance(pair: Optional[str]) -> List[Dict[str, Any]]:

        filters = [Trade.is_open.is_(False)]
        if(pair is not None):
            filters.append(Trade.pair == pair)

        sell_tag_perf = Trade.query.with_entities(
            Trade.sell_reason,
            func.sum(Trade.close_profit).label('profit_sum'),
            func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
            func.count(Trade.pair).label('count')
        ).filter(*filters)\
            .group_by(Trade.sell_reason) \
            .order_by(desc('profit_sum_abs')) \
            .all()

        return [
            {
                'sell_reason': sell_reason if sell_reason is not None else "Other",
                'profit_ratio': profit,
                'profit_pct': round(profit * 100, 2),
                'profit_abs': profit_abs,
                'count': count
            }
            for sell_reason, profit, profit_abs, count in sell_tag_perf
        ]

    @staticmethod
    def get_mix_tag_performance(pair: Optional[str]) -> List[Dict[str, Any]]:

        filters = [Trade.is_open.is_(False)]
        if(pair is not None):
            filters.append(Trade.pair == pair)

        mix_tag_perf = Trade.query.with_entities(
            Trade.id,
            Trade.buy_tag,
            Trade.sell_reason,
            func.sum(Trade.close_profit).label('profit_sum'),
            func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
            func.count(Trade.pair).label('count')
        ).filter(*filters)\
            .group_by(Trade.id) \
            .order_by(desc('profit_sum_abs')) \
            .all()

        return_list: List[Dict] = []
        for id, buy_tag, sell_reason, profit, profit_abs, count in mix_tag_perf:
            buy_tag = buy_tag if buy_tag is not None else "Other"
            sell_reason = sell_reason if sell_reason is not None else "Other"

            if(sell_reason is not None and buy_tag is not None):
                mix_tag = buy_tag + " " + sell_reason
                i = 0
                if not any(item["mix_tag"] == mix_tag for item in return_list):
                    return_list.append({'mix_tag': mix_tag,
                                        'profit': profit,
                                        'profit_pct': round(profit * 100, 2),
                                        'profit_abs': profit_abs,
                                        'count': count})
                else:
                    while i < len(return_list):
                        if return_list[i]["mix_tag"] == mix_tag:
                            return_list[i] = {
                                'mix_tag': mix_tag,
                                'profit': profit + return_list[i]["profit"],
                                'profit_pct': round(profit + return_list[i]["profit"] * 100, 2),
                                'profit_abs': profit_abs + return_list[i]["profit_abs"],
                                'count': 1 + return_list[i]["count"]}
                        i += 1

        return return_list

    @staticmethod
    def get_best_pair(start_date: datetime = datetime.fromtimestamp(0)):
        best_pair = Trade.query.with_entities(
            Trade.pair, func.sum(Trade.close_profit).label('profit_sum')
        ).filter(Trade.is_open.is_(False) & (Trade.close_date >= start_date)) \
            .group_by(Trade.pair) \
            .order_by(desc('profit_sum')).first()
        return best_pair


class PairLock(_DECL_BASE):
    __tablename__ = 'pairlocks'

    id = Column(Integer, primary_key=True)

    pair = Column(String(25), nullable=False, index=True)
    reason = Column(String(255), nullable=True)

    lock_time = Column(DateTime, nullable=False)

    lock_end_time = Column(DateTime, nullable=False, index=True)

    active = Column(Boolean, nullable=False, default=True, index=True)

    def __repr__(self):
        lock_time = self.lock_time.strftime(DATETIME_PRINT_FORMAT)
        lock_end_time = self.lock_end_time.strftime(DATETIME_PRINT_FORMAT)
        return (f'PairLock(id={self.id}, pair={self.pair}, lock_time={lock_time}, '
                f'lock_end_time={lock_end_time}, reason={self.reason}, active={self.active})')

    @staticmethod
    def query_pair_locks(pair: Optional[str], now: datetime) -> Query:
        filters = [PairLock.lock_end_time > now,
                   # Only active locks
                   PairLock.active.is_(True), ]
        if pair:
            filters.append(PairLock.pair == pair)
        return PairLock.query.filter(
            *filters
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'pair': self.pair,
            'lock_time': self.lock_time.strftime(DATETIME_PRINT_FORMAT),
            'lock_timestamp': int(self.lock_time.replace(tzinfo=timezone.utc).timestamp() * 1000),
            'lock_end_time': self.lock_end_time.strftime(DATETIME_PRINT_FORMAT),
            'lock_end_timestamp': int(self.lock_end_time.replace(tzinfo=timezone.utc
                                                                 ).timestamp() * 1000),
            'reason': self.reason,
            'active': self.active,
        }
