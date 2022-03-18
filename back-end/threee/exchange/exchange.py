import asyncio
import http
import inspect
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import arrow
import ccxt
import ccxt.async_support as ccxt_async
from cachetools import TTLCache
from ccxt.base.decimal_to_precision import (ROUND_DOWN, ROUND_UP, TICK_SIZE, TRUNCATE,
                                            decimal_to_precision)
from pandas import DataFrame

from threee.constants import (DEFAULT_AMOUNT_RESERVE_PERCENT, NON_OPEN_EXCHANGE_STATES,
                                 ListPairsWithTimeframes)
from threee.data.converter import ohlcv_to_dataframe, trades_dict_to_list
from threee.exceptions import (DDosProtection, ExchangeError, InsufficientFundsError,
                                  InvalidOrderException, OperationalException, PricingError,
                                  RetryableOrderError, TemporaryError)
from threee.exchange.common import (API_FETCH_ORDER_RETRY_COUNT, BAD_EXCHANGES,
                                       EXCHANGE_HAS_OPTIONAL, EXCHANGE_HAS_REQUIRED,
                                       remove_credentials, retrier, retrier_async)
from threee.misc import chunks, deep_merge_dicts, safe_value_fallback2
from threee.plugins.pairlist.pairlist_helpers import expand_pairlist


CcxtModuleType = Any

http.cookies.Morsel._reserved["samesite"] = "SameSite"
logger = logging.getLogger(__name__)

class Exchange:

    _config: Dict = {}
    # ccxt sync/async 초기화에 직접 추가할 매개변수
    _ccxt_config: Dict = {}
    # 매수/매도 호출에 직접 추가할 매개변수
    _params: Dict = {}
    # ccxt 개체에 추가
    _headers: Dict = {}
    #기본 default값 지정
    _ft_has_default: Dict = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["gtc"],
        "time_in_force_parameter": "timeInForce",
        "ohlcv_params": {},
        "ohlcv_candle_limit": 500,
        "ohlcv_partial_candle": True,
        "ohlcv_volume_currency": "base",
        "trades_pagination": "time",
        "trades_pagination_arg": "since",
        "l2_limit_range": None,
        "l2_limit_range_required": True,
    }
    _ft_has: Dict = {}

    def __init__(self, config: Dict[str, Any], validate: bool = True) -> None:
        """
        config.json파일이 있으면 모든 모듈 초기화
        """
        self._api: ccxt.Exchange = None
        self._api_async: ccxt_async.Exchange = None
        self._markets: Dict = {}
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._config.update(config)
        # 새로 고침 시간 유지
        self._pairs_last_refresh_time: Dict[Tuple[str, str], int] = {}
        self._last_markets_refresh: int = 0
        self._fetch_tickers_cache: TTLCache = TTLCache(maxsize=1, ttl=60 * 10)
        self._sell_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)
        self._buy_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)
        self._klines: Dict[Tuple[str, str], DataFrame] = {}
        # 가상거래
        self._dry_run_open_orders: Dict[str, Any] = {}
        remove_credentials(config)
        exchange_config = config['exchange']
        self.log_responses = exchange_config.get('log_responses', False)

        # Deep merge ft_has with default ft_has options
        self._ft_has = deep_merge_dicts(self._ft_has, deepcopy(self._ft_has_default))
        if exchange_config.get('_ft_has_params'):
            self._ft_has = deep_merge_dicts(exchange_config.get('_ft_has_params'),
                                            self._ft_has)
            logger.info("Overriding exchange._ft_has with config params, result: %s", self._ft_has)

        # Assign this directly for easy access
        self._ohlcv_partial_candle = self._ft_has['ohlcv_partial_candle']

        self._trades_pagination = self._ft_has['trades_pagination']
        self._trades_pagination_arg = self._ft_has['trades_pagination_arg']

        # Initialize ccxt objects
        ccxt_config = self._ccxt_config.copy()
        ccxt_config = deep_merge_dicts(exchange_config.get('ccxt_config', {}), ccxt_config)
        ccxt_config = deep_merge_dicts(exchange_config.get('ccxt_sync_config', {}), ccxt_config)

        self._api = self._init_ccxt(exchange_config, ccxt_kwargs=ccxt_config)

        ccxt_async_config = self._ccxt_config.copy()
        ccxt_async_config = deep_merge_dicts(exchange_config.get('ccxt_config', {}),
                                             ccxt_async_config)
        ccxt_async_config = deep_merge_dicts(exchange_config.get('ccxt_async_config', {}),
                                             ccxt_async_config)
        self._api_async = self._init_ccxt(
            exchange_config, ccxt_async, ccxt_kwargs=ccxt_async_config)
        if validate:
            # 시간 프레임 확인
            self.validate_timeframes(config.get('timeframe'))
            self._load_markets()
            self.validate_stakecurrency(config['stake_currency'])
            if not exchange_config.get('skip_pair_validation'):
                self.validate_pairs(config['exchange']['pair_whitelist'])
            self.validate_ordertypes(config.get('order_types', {}))
            self.validate_order_time_in_force(config.get('order_time_in_force', {}))
            self.required_candle_call_count = self.validate_required_startup_candles(
                config.get('startup_candle_count', 0), config.get('timeframe', ''))
        self.markets_refresh_interval: int = exchange_config.get(
            "markets_refresh_interval", 60) * 60

    def __del__(self):
        """
        Destructor - clean up async stuff
        """
        self.close()

    def close(self):
        if (self._api_async and inspect.iscoroutinefunction(self._api_async.close)
                and self._api_async.session):
            self.loop.run_until_complete(self._api_async.close())

    def _init_ccxt(self, exchange_config: Dict[str, Any], ccxt_module: CcxtModuleType = ccxt,
                   ccxt_kwargs: Dict = {}) -> ccxt.Exchange:
        """
        ccxt를 초기화하고 유효한 반환
        """
        name = exchange_config['name']
        ex_config = {
            'apiKey': exchange_config.get('key'),
            'secret': exchange_config.get('secret'),
            'password': exchange_config.get('password'),
            'uid': exchange_config.get('uid', ''),
        }

        if self._headers:
            # 사용자를 혼동하지 않도록 위의 출력
            ccxt_kwargs = deep_merge_dicts({'headers': self._headers}, ccxt_kwargs)
        if ccxt_kwargs:
            ex_config.update(ccxt_kwargs)
        try:
            api = getattr(ccxt_module, name.lower())(ex_config)
        except (KeyError, AttributeError) as e:
            raise OperationalException() from e
        except ccxt.BaseError as e:
            raise OperationalException() from e
        self.set_sandbox(api, exchange_config, name)

        return api

    @property
    def name(self) -> str:
        return self._api.name

    @property
    def id(self) -> str:
        return self._api.id

    @property
    def timeframes(self) -> List[str]:
        return list((self._api.timeframes or {}).keys())

    @property
    def markets(self) -> Dict:
        if not self._markets:
            self._load_markets()
        return self._markets

    @property
    def precisionMode(self) -> str:
        return self._api.precisionMode

    def ohlcv_candle_limit(self, timeframe: str) -> int:
        """
        교환 ohlcv 대이터 한도 다를경우 오류 해결
        """
        return int(self._ft_has.get('ohlcv_candle_limit_per_timeframe', {}).get(
            timeframe, self._ft_has.get('ohlcv_candle_limit')))

    def get_markets(self, base_currencies: List[str] = None, quote_currencies: List[str] = None,
                    pairs_only: bool = False, active_only: bool = False) -> Dict[str, Any]:
        """
        필터링된 코인으로교환

        """
        markets = self.markets
        if not markets:
            raise OperationalException("Markets were not loaded.")

        if base_currencies:
            markets = {k: v for k, v in markets.items() if v['base'] in base_currencies}
        if quote_currencies:
            markets = {k: v for k, v in markets.items() if v['quote'] in quote_currencies}
        if pairs_only:
            markets = {k: v for k, v in markets.items() if self.market_is_tradable(v)}
        if active_only:
            markets = {k: v for k, v in markets.items() if market_is_active(v)}
        return markets

    def get_quote_currencies(self) -> List[str]:
        """
        지원되는 견적 통화 목록 반환
        """
        markets = self.markets
        return sorted(set([x['quote'] for _, x in markets.items()]))

    def get_pair_quote_currency(self, pair: str) -> str:
        return self.markets.get(pair, {}).get('quote', '')

    def get_pair_base_currency(self, pair: str) -> str:
        return self.markets.get(pair, {}).get('base', '')

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        #거래 가능한지 확인
        symbol_parts = market['symbol'].split('/')
        return (len(symbol_parts) == 2 and
                len(symbol_parts[0]) > 0 and
                len(symbol_parts[1]) > 0 and
                symbol_parts[0] == market.get('base') and
                symbol_parts[1] == market.get('quote')
                )

    def klines(self, pair_interval: Tuple[str, str], copy: bool = True) -> DataFrame:
        if pair_interval in self._klines:
            return self._klines[pair_interval].copy() if copy else self._klines[pair_interval]
        else:
            return DataFrame()

    def set_sandbox(self, api: ccxt.Exchange, exchange_config: dict, name: str) -> None:
        if exchange_config.get('sandbox'):
            if api.urls.get('test'):
                api.urls['api'] = api.urls['test']

    def _load_async_markets(self, reload: bool = False) -> None:
        try:
            if self._api_async:
                self.loop.run_until_complete(
                    self._api_async.load_markets(reload=reload))
        except (asyncio.TimeoutError, ccxt.BaseError) as e:
            return

    def _load_markets(self) -> None:
        """  비동기 동기 둘다 마켓로드 """
        try:
            self._markets = self._api.load_markets()
            self._load_async_markets()
            self._last_markets_refresh = arrow.utcnow().int_timestamp
        except ccxt.BaseError:
            None

    def reload_markets(self) -> None:
        # 시장을 새로고침해야 하는지 확인
        if (self._last_markets_refresh > 0) and (
                self._last_markets_refresh + self.markets_refresh_interval
                > arrow.utcnow().int_timestamp):
            return None
        try:
            self._markets = self._api.load_markets(reload=True)
            # Also reload async markets to avoid issues with newly listed pairs
            self._load_async_markets(reload=True)
            self._last_markets_refresh = arrow.utcnow().int_timestamp
        except ccxt.BaseError:
            None

    def validate_stakecurrency(self, stake_currency: str) -> None:
        """
        거래소에서 사용 가능한 통화와 지분 통화를 확인
        """
        quote_currencies = self.get_quote_currencies()

    def validate_pairs(self, pairs: List[str]) -> None:
        """
        거래소에서 거래 가능한지 확인
        """
        extended_pairs = expand_pairlist(pairs, list(self.markets), keep_invalid=True)
        invalid_pairs = []
        for pair in extended_pairs:

            if (self._config['stake_currency'] and
                    self.get_pair_quote_currency(pair) != self._config['stake_currency']):
                invalid_pairs.append(pair)


    def get_valid_pair_combination(self, curr_1: str, curr_2: str) -> str:
        """
        유효한 쌍 조합 종목 이름 찾기
        """
        for pair in [f"{curr_1}/{curr_2}", f"{curr_2}/{curr_1}"]:
            if pair in self.markets and self.markets[pair].get('active'):
                return pair


    def validate_timeframes(self, timeframe: Optional[str]) -> None:
        None

    def validate_ordertypes(self, order_types: Dict) -> None:
        None

    def validate_order_time_in_force(self, order_time_in_force: Dict) -> None:
        None

    def validate_required_startup_candles(self, startup_candles: int, timeframe: str) -> int:
        """
        필요한 startup_candles가 ohlcv_candle_limit()보다 큰지 확인
        """
        candle_limit = self.ohlcv_candle_limit(timeframe)
        # 하나의  데이터 하나 필요
        candle_count = startup_candles + 1
        required_candle_call_count = int(
            (candle_count / candle_limit) + (0 if candle_count % candle_limit == 0 else 1))

        return required_candle_call_count

    def exchange_has(self, endpoint: str) -> bool:
        #API 엔드포인트를 구현하는지 확인
        return endpoint in self._api.has and self._api.has[endpoint]

    def amount_to_precision(self, pair: str, amount: float) -> float:
        #구매 또는 판매 금액을 반환
        if self.markets[pair]['precision']['amount']:
            amount = float(decimal_to_precision(amount, rounding_mode=TRUNCATE,
                                                precision=self.markets[pair]['precision']['amount'],
                                                counting_mode=self.precisionMode,
                                                ))

        return amount

    def price_to_precision(self, pair: str, price: float) -> float:
        """
        Exchange에서 허용하는 정밀도로 반올림된 가격을 반환
        """
        if self.markets[pair]['precision']['price']:
            if self.precisionMode == TICK_SIZE:
                precision = self.markets[pair]['precision']['price']
                missing = price % precision
                if missing != 0:
                    price = round(price - missing + precision, 10)
            else:
                symbol_prec = self.markets[pair]['precision']['price']
                big_price = price * pow(10, symbol_prec)
                price = ceil(big_price) / pow(10, symbol_prec)
        return price

    def price_get_one_pip(self, pair: str, price: float) -> float:
        precision = self.markets[pair]['precision']['price']
        if self.precisionMode == TICK_SIZE:
            return precision
        else:
            return 1 / pow(10, precision)

    def get_min_pair_stake_amount(self, pair: str, price: float,
                                  stoploss: float) -> Optional[float]:
        try:
            market = self.markets[pair]
        except KeyError:
            None

        if 'limits' not in market:
            return None

        min_stake_amounts = []
        limits = market['limits']
        if ('cost' in limits and 'min' in limits['cost']
                and limits['cost']['min'] is not None):
            min_stake_amounts.append(limits['cost']['min'])

        if ('amount' in limits and 'min' in limits['amount']
                and limits['amount']['min'] is not None):
            min_stake_amounts.append(limits['amount']['min'] * price)

        if not min_stake_amounts:
            return None


        amount_reserve_percent = 1.0 + self._config.get('amount_reserve_percent',
                                                        DEFAULT_AMOUNT_RESERVE_PERCENT)
        amount_reserve_percent = (
            amount_reserve_percent / (1 - abs(stoploss)) if abs(stoploss) != 1 else 1.5
        )

        amount_reserve_percent = max(min(amount_reserve_percent, 1.5), 1)

        return max(min_stake_amounts) * amount_reserve_percent

    def create_dry_run_order(self, pair: str, ordertype: str, side: str, amount: float,
                             rate: float, params: Dict = {},
                             stop_loss: bool = False) -> Dict[str, Any]:
        order_id = f'dry_run_{side}_{datetime.now().timestamp()}'
        _amount = self.amount_to_precision(pair, amount)
        dry_order: Dict[str, Any] = {
            'id': order_id,
            'symbol': pair,
            'price': rate,
            'average': rate,
            'amount': _amount,
            'cost': _amount * rate,
            'type': ordertype,
            'side': side,
            'filled': 0,
            'remaining': _amount,
            'datetime': arrow.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'timestamp': arrow.utcnow().int_timestamp * 1000,
            'status': "closed" if ordertype == "market" and not stop_loss else "open",
            'fee': None,
            'info': {}
        }
        if stop_loss:
            dry_order["info"] = {"stopPrice": dry_order["price"]}
            dry_order["stopPrice"] = dry_order["price"]
            dry_order["ft_order_type"] = "stoploss"

        if dry_order["type"] == "market" and not dry_order.get("ft_order_type"):
            average = self.get_dry_market_fill_price(pair, side, amount, rate)
            dry_order.update({
                'average': average,
                'filled': _amount,
                'cost': dry_order['amount'] * average,
            })
            dry_order = self.add_dry_order_fee(pair, dry_order)

        dry_order = self.check_dry_limit_order_filled(dry_order)

        self._dry_run_open_orders[dry_order["id"]] = dry_order
        return dry_order

    def add_dry_order_fee(self, pair: str, dry_order: Dict[str, Any]) -> Dict[str, Any]:
        dry_order.update({
            'fee': {
                'currency': self.get_pair_quote_currency(pair),
                'cost': dry_order['cost'] * self.get_fee(pair),
                'rate': self.get_fee(pair)
            }
        })
        return dry_order

    def get_dry_market_fill_price(self, pair: str, side: str, amount: float, rate: float) -> float:
        """
        시장가 주문 체결 가격 계산
        """
        if self.exchange_has('fetchL2OrderBook'):
            ob = self.fetch_l2_order_book(pair, 20)
            ob_type = 'asks' if side == 'buy' else 'bids'
            slippage = 0.05
            max_slippage_val = rate * ((1 + slippage) if side == 'buy' else (1 - slippage))
            remaining_amount = amount
            filled_amount = 0.0
            book_entry_price = 0.0
            for book_entry in ob[ob_type]:
                book_entry_price = book_entry[0]
                book_entry_coin_volume = book_entry[1]
                if remaining_amount > 0:
                    if remaining_amount < book_entry_coin_volume:

                        filled_amount += remaining_amount * book_entry_price
                        break
                    else:
                        filled_amount += book_entry_coin_volume * book_entry_price
                    remaining_amount -= book_entry_coin_volume
                else:
                    break
            else:

                filled_amount += remaining_amount * book_entry_price
            forecast_avg_filled_price = max(filled_amount, 0) / amount

            if side == 'buy':
                forecast_avg_filled_price = min(forecast_avg_filled_price, max_slippage_val)

            else:
                forecast_avg_filled_price = max(forecast_avg_filled_price, max_slippage_val)

            return self.price_to_precision(pair, forecast_avg_filled_price)

        return rate

    def _is_dry_limit_order_filled(self, pair: str, side: str, limit: float) -> bool:
        if not self.exchange_has('fetchL2OrderBook'):
            return True
        ob = self.fetch_l2_order_book(pair, 1)
        try:
            if side == 'buy':
                price = ob['asks'][0][0]
                if limit >= price:
                    return True
            else:
                price = ob['bids'][0][0]
                if limit <= price:
                    return True
        except IndexError:
            pass
        return False

    def check_dry_limit_order_filled(self, order: Dict[str, Any]) -> Dict[str, Any]:
        # 지정가 주문 채우기
        if (order['status'] != "closed"
                and order['type'] in ["limit"]
                and not order.get('ft_order_type')):
            pair = order['symbol']
            if self._is_dry_limit_order_filled(pair, order['side'], order['price']):
                order.update({
                    'status': 'closed',
                    'filled': order['amount'],
                    'remaining': 0,
                })
                self.add_dry_order_fee(pair, order)

        return order

    def fetch_dry_run_order(self, order_id) -> Dict[str, Any]:
        try:
            order = self._dry_run_open_orders[order_id]
            order = self.check_dry_limit_order_filled(order)
            return order
        except KeyError as e:
            pass

    def create_order(self, pair: str, ordertype: str, side: str, amount: float,
                     rate: float, time_in_force: str = 'gtc') -> Dict:

        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(pair, ordertype, side, amount, rate)
            return dry_order

        params = self._params.copy()
        if time_in_force != 'gtc' and ordertype != 'market':
            param = self._ft_has.get('time_in_force_parameter', '')
            params.update({param: time_in_force})

        try:
            amount = self.amount_to_precision(pair, amount)
            needs_price = (ordertype != 'market'
                           or self._api.options.get("createMarketBuyOrderRequiresPrice", False))
            rate_for_order = self.price_to_precision(pair, rate) if needs_price else None

            order = self._api.create_order(pair, ordertype, side,
                                           amount, rate_for_order, params)
            self._log_exchange_response('create_order', order)
            return order

        except ccxt.InsufficientFunds as e:
            pass

    def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
        pass

    def _get_stop_params(self, ordertype: str, stop_price: float) -> Dict:
        params = self._params.copy()
        params.update({'stopPrice': stop_price})
        return params

    @retrier(retries=0)
    def stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict) -> Dict:

        user_order_type = order_types.get('stoploss', 'market')
        if user_order_type in self._ft_has["stoploss_order_types"].keys():
            ordertype = self._ft_has["stoploss_order_types"][user_order_type]
        else:
            ordertype = list(self._ft_has["stoploss_order_types"].values())[0]
            user_order_type = list(self._ft_has["stoploss_order_types"].keys())[0]

        stop_price_norm = self.price_to_precision(pair, stop_price)
        rate = None
        if user_order_type == 'limit':
            limit_price_pct = order_types.get('stoploss_on_exchange_limit_ratio', 0.99)
            rate = stop_price * limit_price_pct
            rate = self.price_to_precision(pair, rate)

        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(
                pair, ordertype, "sell", amount, stop_price_norm, stop_loss=True)
            return dry_order

        try:
            params = self._get_stop_params(ordertype=ordertype, stop_price=stop_price_norm)

            amount = self.amount_to_precision(pair, amount)

            order = self._api.create_order(symbol=pair, type=ordertype, side='sell',
                                           amount=amount, price=rate, params=params)

            self._log_exchange_response('create_stoploss_order', order)
            return order
        except ccxt.InsufficientFunds as e:
            pass

    @retrier(retries=API_FETCH_ORDER_RETRY_COUNT)
    def fetch_order(self, order_id: str, pair: str) -> Dict:
        if self._config['dry_run']:
            return self.fetch_dry_run_order(order_id)
        try:
            order = self._api.fetch_order(order_id, pair)
            self._log_exchange_response('fetch_order', order)
            return order
        except ccxt.OrderNotFound as e:
            pass

    fetch_stoploss_order = fetch_order

    def fetch_order_or_stoploss_order(self, order_id: str, pair: str,
                                      stoploss_order: bool = False) -> Dict:
        if stoploss_order:
            return self.fetch_stoploss_order(order_id, pair)
        return self.fetch_order(order_id, pair)

    def check_order_canceled_empty(self, order: Dict) -> bool:
        return (order.get('status') in NON_OPEN_EXCHANGE_STATES
                and order.get('filled') == 0.0)

    @retrier
    def cancel_order(self, order_id: str, pair: str) -> Dict:
        if self._config['dry_run']:
            try:
                order = self.fetch_dry_run_order(order_id)

                order.update({'status': 'canceled', 'filled': 0.0, 'remaining': order['amount']})
                return order
            except InvalidOrderException:
                return {}

        try:
            order = self._api.cancel_order(order_id, pair)
            self._log_exchange_response('cancel_order', order)
            return order
        except ccxt.InvalidOrder as e:
            pass

    cancel_stoploss_order = cancel_order

    def is_cancel_order_result_suitable(self, corder) -> bool:
        if not isinstance(corder, dict):
            return False

        required = ('fee', 'status', 'amount')
        return all(k in corder for k in required)

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict:
        """
        결과를 반환하는 주문을 취소
        """
        try:
            corder = self.cancel_order(order_id, pair)
            if self.is_cancel_order_result_suitable(corder):
                return corder
        except InvalidOrderException:
            pass
        try:
            order = self.fetch_order(order_id, pair)
        except InvalidOrderException:
            order = {'fee': {}, 'status': 'canceled', 'amount': amount, 'info': {}}

        return order

    def cancel_stoploss_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict:
        corder = self.cancel_stoploss_order(order_id, pair)
        if self.is_cancel_order_result_suitable(corder):
            return corder
        try:
            order = self.fetch_stoploss_order(order_id, pair)
        except InvalidOrderException:
            order = {'fee': {}, 'status': 'canceled', 'amount': amount, 'info': {}}

        return order

    @retrier
    def get_balances(self) -> dict:

        try:
            balances = self._api.fetch_balance()
            balances.pop("info", None)
            balances.pop("free", None)
            balances.pop("total", None)
            balances.pop("used", None)
            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            pass
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_tickers(self, symbols: List[str] = None, cached: bool = False) -> Dict:
        if cached:
            tickers = self._fetch_tickers_cache.get('fetch_tickers')
            if tickers:
                return tickers
        try:
            tickers = self._api.fetch_tickers(symbols)
            self._fetch_tickers_cache['fetch_tickers'] = tickers
            return tickers
        except ccxt.NotSupported as e:
            pass

    @retrier
    def fetch_ticker(self, pair: str) -> dict:
        try:
            if (pair not in self.markets or
                    self.markets[pair].get('active', False) is False):
                pass
            data = self._api.fetch_ticker(pair)
            return data
        except ccxt.DDoSProtection as e:
            pass

    @staticmethod
    def get_next_limit_in_list(limit: int, limit_range: Optional[List[int]],
                               range_required: bool = True):
        if not limit_range:
            return limit

        result = min([x for x in limit_range if limit <= x] + [max(limit_range)])
        if not range_required and limit > result:
            return None
        return result

    @retrier
    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> dict:
        """
        거래소에서 L2 오더북 받기
        """
        limit1 = self.get_next_limit_in_list(limit, self._ft_has['l2_limit_range'],
                                             self._ft_has['l2_limit_range_required'])
        try:

            return self._api.fetch_l2_order_book(pair, limit1)
        except ccxt.NotSupported as e:
            pass

    def get_rate(self, pair: str, refresh: bool, side: str) -> float:
        """
        입찰/매도 타겟 계산
        """
        cache_rate: TTLCache = self._buy_rate_cache if side == "buy" else self._sell_rate_cache
        [strat_name, name] = ['bid_strategy', 'Buy'] if side == "buy" else ['ask_strategy', 'Sell']

        if not refresh:
            rate = cache_rate.get(pair)
            if rate:
                return rate

        conf_strategy = self._config.get(strat_name, {})

        if conf_strategy.get('use_order_book', False) and ('use_order_book' in conf_strategy):

            order_book_top = conf_strategy.get('order_book_top', 1)
            order_book = self.fetch_l2_order_book(pair, order_book_top)
            try:
                rate = order_book[f"{conf_strategy['price_side']}s"][order_book_top - 1][0]
            except ab as e:
                pass
            price_side = {conf_strategy['price_side'].capitalize()}

        else:
            ticker = self.fetch_ticker(pair)
            ticker_rate = ticker[conf_strategy['price_side']]
            if ticker['last'] and ticker_rate:
                if side == 'buy' and ticker_rate > ticker['last']:
                    balance = conf_strategy.get('ask_last_balance', 0.0)
                    ticker_rate = ticker_rate + balance * (ticker['last'] - ticker_rate)
                elif side == 'sell' and ticker_rate < ticker['last']:
                    balance = conf_strategy.get('bid_last_balance', 0.0)
                    ticker_rate = ticker_rate - balance * (ticker_rate - ticker['last'])
            rate = ticker_rate

        cache_rate[pair] = rate

        return rate


    @retrier
    def get_trades_for_order(self, order_id: str, pair: str, since: datetime,
                             params: Optional[Dict] = None) -> List:
        if self._config['dry_run']:
            return []
        if not self.exchange_has('fetchMyTrades'):
            return []
        try:
            _params = params if params else {}
            my_trades = self._api.fetch_my_trades(
                pair, int((since.replace(tzinfo=timezone.utc).timestamp() - 5) * 1000),
                params=_params)
            matched_trades = [trade for trade in my_trades if trade['order'] == order_id]

            self._log_exchange_response('get_trades_for_order', matched_trades)
            return matched_trades
        except nin as e:
            pass

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        return order['id']

    @retrier
    def get_fee(self, symbol: str, type: str = '', side: str = '', amount: float = 1,
                price: float = 1, taker_or_maker: str = 'maker') -> float:
        try:
            if self._config['dry_run'] and self._config.get('fee', None) is not None:
                return self._config['fee']
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets()

            return self._api.calculate_fee(symbol=symbol, type=type, side=side, amount=amount,
                                           price=price, takerOrMaker=taker_or_maker)['rate']
        except nan as e:
            pass

    @staticmethod
    def order_has_fee(order: Dict) -> bool:
        """
        전달된 순서 dict에 수수료를 추출하는 데 필요한 키가 있는지 확인합니다.
        """
        if not isinstance(order, dict):
            return False
        return ('fee' in order and order['fee'] is not None
                and (order['fee'].keys() >= {'currency', 'cost'})
                and order['fee']['currency'] is not None
                and order['fee']['cost'] is not None
                )

    def calculate_fee_rate(self, order: Dict) -> Optional[float]:

        if order['fee'].get('rate') is not None:
            return order['fee'].get('rate')
        fee_curr = order['fee']['currency']
        if fee_curr in self.get_pair_base_currency(order['symbol']):
            return round(
                order['fee']['cost'] / safe_value_fallback2(order, order, 'filled', 'amount'), 8)
        elif fee_curr in self.get_pair_quote_currency(order['symbol']):
            return round(order['fee']['cost'] / order['cost'], 8) if order['cost'] else None
        else:
            if not order['cost']:
                return None
            try:
                comb = self.get_valid_pair_combination(fee_curr, self._config['stake_currency'])
                tick = self.fetch_ticker(comb)

                fee_to_quote_rate = safe_value_fallback2(tick, tick, 'last', 'ask')
            except ExchangeError:
                fee_to_quote_rate = self._config['exchange'].get('unknown_fee_rate', None)
                if not fee_to_quote_rate:
                    return None
            return round((order['fee']['cost'] * fee_to_quote_rate) / order['cost'], 8)

    def extract_cost_curr_rate(self, order: Dict) -> Tuple[float, str, Optional[float]]:
        """
        비용, 통화, 비율의 튜플 추출
        """
        return (order['fee']['cost'],
                order['fee']['currency'],
                self.calculate_fee_rate(order))

    def get_historic_ohlcv(self, pair: str, timeframe: str,
                           since_ms: int, is_new_pair: bool = False) -> List:

        pair, timeframe, data = self.loop.run_until_complete(
            self._async_get_historic_ohlcv(pair=pair, timeframe=timeframe,
                                           since_ms=since_ms, is_new_pair=is_new_pair))
        return data

    def get_historic_ohlcv_as_df(self, pair: str, timeframe: str,
                                 since_ms: int) -> DataFrame:

        ticks = self.get_historic_ohlcv(pair, timeframe, since_ms=since_ms)
        return ohlcv_to_dataframe(ticks, timeframe, pair=pair, fill_missing=True,
                                  drop_incomplete=self._ohlcv_partial_candle)

    async def _async_get_historic_ohlcv(self, pair: str, timeframe: str,
                                        since_ms: int, is_new_pair: bool = False,
                                        raise_: bool = False
                                        ) -> Tuple[str, str, List]:

        one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(timeframe)
        input_coroutines = [self._async_get_candle_history(
            pair, timeframe, since) for since in
            range(since_ms, arrow.utcnow().int_timestamp * 1000, one_call)]

        data: List = []
        for input_coro in chunks(input_coroutines, 100):

            results = await asyncio.gather(*input_coro, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    if raise_:
                        raise
                    continue
                else:
                    p, _, new_data = res
                    if p == pair:
                        data.extend(new_data)
        data = sorted(data, key=lambda x: x[0])
        return pair, timeframe, data

    def refresh_latest_ohlcv(self, pair_list: ListPairsWithTimeframes, *,
                             since_ms: Optional[int] = None, cache: bool = True
                             ) -> Dict[Tuple[str, str], DataFrame]:
        input_coroutines = []
        cached_pairs = []
        for pair, timeframe in set(pair_list):
            if ((pair, timeframe) not in self._klines or not cache
                    or self._now_is_time_to_refresh(pair, timeframe)):
                if not since_ms and self.required_candle_call_count > 1:
                    one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(timeframe)
                    move_to = one_call * self.required_candle_call_count
                    now = timeframe_to_next_date(timeframe)
                    since_ms = int((now - timedelta(seconds=move_to // 1000)).timestamp() * 1000)

                if since_ms:
                    input_coroutines.append(self._async_get_historic_ohlcv(
                        pair, timeframe, since_ms=since_ms, raise_=True))
                else:
                    input_coroutines.append(self._async_get_candle_history(
                        pair, timeframe, since_ms=since_ms))
            else:
                cached_pairs.append((pair, timeframe))

        results_df = {}
        for input_coro in chunks(input_coroutines, 100):
            async def gather_stuff():
                return await asyncio.gather(*input_coro, return_exceptions=True)

            results = self.loop.run_until_complete(gather_stuff())

            for res in results:
                if isinstance(res, Exception):
                    continue
                pair, timeframe, ticks = res

                if ticks:
                    self._pairs_last_refresh_time[(pair, timeframe)] = ticks[-1][0] // 1000

                ohlcv_df = ohlcv_to_dataframe(
                    ticks, timeframe, pair=pair, fill_missing=True,
                    drop_incomplete=self._ohlcv_partial_candle)
                results_df[(pair, timeframe)] = ohlcv_df
                if cache:
                    self._klines[(pair, timeframe)] = ohlcv_df


        for pair, timeframe in cached_pairs:
            results_df[(pair, timeframe)] = self.klines((pair, timeframe), copy=False)

        return results_df

    def _now_is_time_to_refresh(self, pair: str, timeframe: str) -> bool:

        interval_in_sec = timeframe_to_seconds(timeframe)

        return not ((self._pairs_last_refresh_time.get((pair, timeframe), 0)
                     + interval_in_sec) >= arrow.utcnow().int_timestamp)

    @retrier_async
    async def _async_get_candle_history(self, pair: str, timeframe: str,
                                        since_ms: Optional[int] = None) -> Tuple[str, str, List]:
        try:

            s = '(' + arrow.get(since_ms // 1000).isoformat() + ') ' if since_ms is not None else ''
            params = self._ft_has.get('ohlcv_params', {})
            data = await self._api_async.fetch_ohlcv(pair, timeframe=timeframe,
                                                     since=since_ms,
                                                     limit=self.ohlcv_candle_limit(timeframe),
                                                     params=params)
            try:
                if data and data[0][0] > data[-1][0]:
                    data = sorted(data, key=lambda x: x[0])
            except IndexError:
                pass
            return pair, timeframe, data

        except nan as e:
            pass

    @retrier_async
    async def _async_fetch_trades(self, pair: str,
                                  since: Optional[int] = None,
                                  params: Optional[dict] = None) -> List[List]:
        try:
            if params:
                trades = await self._api_async.fetch_trades(pair, params=params, limit=1000)
            else:
                trades = await self._api_async.fetch_trades(pair, since=since, limit=1000)
            return trades_dict_to_list(trades)
        except nan as e:
            pass

    async def _async_get_trade_history_id(self, pair: str,
                                          until: int,
                                          since: Optional[int] = None,
                                          from_id: Optional[str] = None) -> Tuple[str, List[List]]:
        trades: List[List] = []

        if not from_id:
            t = await self._async_fetch_trades(pair, since=since)

            from_id = t[-1][1]
            trades.extend(t[:-1])
        while True:
            t = await self._async_fetch_trades(pair,
                                               params={self._trades_pagination_arg: from_id})
            if t:
                trades.extend(t[:-1])
                if from_id == t[-1][1] or t[-1][0] > until:
                    trades.extend(t[-1:])
                    break

                from_id = t[-1][1]
            else:
                break

        return (pair, trades)

    async def _async_get_trade_history_time(self, pair: str, until: int,
                                            since: Optional[int] = None) -> Tuple[str, List[List]]:

        trades: List[List] = []
        while True:
            t = await self._async_fetch_trades(pair, since=since)
            if t:
                since = t[-1][0]
                trades.extend(t)
                if until and t[-1][0] > until:
                    break
            else:
                break

        return (pair, trades)

    async def _async_get_trade_history(self, pair: str,
                                       since: Optional[int] = None,
                                       until: Optional[int] = None,
                                       from_id: Optional[str] = None) -> Tuple[str, List[List]]:

        if until is None:
            until = ccxt.Exchange.milliseconds()

        if self._trades_pagination == 'time':
            return await self._async_get_trade_history_time(
                pair=pair, since=since, until=until)
        elif self._trades_pagination == 'id':
            return await self._async_get_trade_history_id(
                pair=pair, since=since, until=until, from_id=from_id
            )
        else:
            pass

    def get_historic_trades(self, pair: str,
                            since: Optional[int] = None,
                            until: Optional[int] = None,
                            from_id: Optional[str] = None) -> Tuple[str, List]:

        return self.loop.run_until_complete(
            self._async_get_trade_history(pair=pair, since=since,
                                          until=until, from_id=from_id))


def is_exchange_known_ccxt(exchange_name: str, ccxt_module: CcxtModuleType = None) -> bool:
    return exchange_name in ccxt_exchanges(ccxt_module)


def is_exchange_officially_supported(exchange_name: str) -> bool:
    return exchange_name in ['binance', 'bittrex', 'ftx', 'gateio', 'huobi', 'kraken', 'okx']


def ccxt_exchanges(ccxt_module: CcxtModuleType = None) -> List[str]:
    return ccxt_module.exchanges if ccxt_module is not None else ccxt.exchanges


def available_exchanges(ccxt_module: CcxtModuleType = None) -> List[str]:
    exchanges = ccxt_exchanges(ccxt_module)
    return [x for x in exchanges if validate_exchange(x)[0]]


def validate_exchange(exchange: str) -> Tuple[bool, str]:
    ex_mod = getattr(ccxt, exchange.lower())()
    if not ex_mod or not ex_mod.has:
        return False, ''
    missing = [k for k in EXCHANGE_HAS_REQUIRED if ex_mod.has.get(k) is not True]
    if missing:
        return False, f"missing: {', '.join(missing)}"

    missing_opt = [k for k in EXCHANGE_HAS_OPTIONAL if not ex_mod.has.get(k)]

    if exchange.lower() in BAD_EXCHANGES:
        return False, BAD_EXCHANGES.get(exchange.lower(), '')
    if missing_opt:
        return True, f"missing opt: {', '.join(missing_opt)}"
    return True, ''


def validate_exchanges(all_exchanges: bool) -> List[Tuple[str, bool, str]]:
    exchanges = ccxt_exchanges() if all_exchanges else available_exchanges()
    exchanges_valid = [
        (e, *validate_exchange(e)) for e in exchanges
    ]
    return exchanges_valid


def timeframe_to_seconds(timeframe: str) -> int:
    return ccxt.Exchange.parse_timeframe(timeframe)


def timeframe_to_minutes(timeframe: str) -> int:
    return ccxt.Exchange.parse_timeframe(timeframe) // 60


def timeframe_to_msecs(timeframe: str) -> int:
    return ccxt.Exchange.parse_timeframe(timeframe) * 1000


def timeframe_to_prev_date(timeframe: str, date: datetime = None) -> datetime:
    if not date:
        date = datetime.now(timezone.utc)

    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, date.timestamp() * 1000,
                                                  ROUND_DOWN) // 1000
    return datetime.fromtimestamp(new_timestamp, tz=timezone.utc)


def timeframe_to_next_date(timeframe: str, date: datetime = None) -> datetime:
    if not date:
        date = datetime.now(timezone.utc)
    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, date.timestamp() * 1000,
                                                  ROUND_UP) // 1000
    return datetime.fromtimestamp(new_timestamp, tz=timezone.utc)


def market_is_active(market: Dict) -> bool:
    return market.get('active', True) is not False
