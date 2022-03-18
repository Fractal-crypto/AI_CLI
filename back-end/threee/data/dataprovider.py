import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pandas import DataFrame

from threee.configuration import TimeRange
from threee.constants import ListPairsWithTimeframes, PairWithTimeframe
from threee.data.history import load_pair_history
from threee.enums import RunMode
from threee.exceptions import ExchangeError, OperationalException
from threee.exchange import Exchange, timeframe_to_seconds

NO_EXCHANGE_EXCEPTION = 'none'
MAX_DATAFRAME_CANDLES = 1000


class DataProvider:
    """
    실제 트레이딩 봇이 데이터를 가져올 변환
    """
    def __init__(self, config: dict, exchange: Optional[Exchange], pairlists=None) -> None:
        self._config = config
        self._exchange = exchange
        self._pairlists = pairlists
        self.__cached_pairs: Dict[PairWithTimeframe, Tuple[DataFrame, datetime]] = {}
        self.__slice_index: Optional[int] = None
        self.__cached_pairs_backtesting: Dict[PairWithTimeframe, DataFrame] = {}

    def _set_dataframe_max_index(self, limit_index: int):
        #분석된 데이터 프레임을 지정된 최대 인덱스로 제한
        self.__slice_index = limit_index

    def _set_cached_df(self, pair: str, timeframe: str, dataframe: DataFrame) -> None:
        #데이터 프레임을 저장
        self.__cached_pairs[(pair, timeframe)] = (dataframe, datetime.now(timezone.utc))

    def add_pairlisthandler(self, pairlists) -> None:
        self._pairlists = pairlists

    def historic_ohlcv(self, pair: str, timeframe: str = None) -> DataFrame:
        #저장된 과거 OHLCV 데이터 가져오기
        saved_pair = (pair, str(timeframe))
        if saved_pair not in self.__cached_pairs_backtesting:
            timerange = TimeRange.parse_timerange(None if self._config.get(
                'timerange') is None else str(self._config.get('timerange')))
            timerange.subtract_start(
                timeframe_to_seconds(str(timeframe)) * self._config.get('startup_candle_count', 0)
            )
            self.__cached_pairs_backtesting[saved_pair] = load_pair_history(
                pair=pair,
                timeframe=timeframe or self._config['timeframe'],
                datadir=self._config['datadir'],
                timerange=timerange,
                data_format=self._config.get('dataformat_ohlcv', 'json')
            )
        return self.__cached_pairs_backtesting[saved_pair].copy()

    def get_pair_dataframe(self, pair: str, timeframe: str = None) -> DataFrame:
        #OHLCV 데이터 반환
        if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            # live 데이터 불러오기
            data = self.ohlcv(pair=pair, timeframe=timeframe)
        else:
            # 데이터 불러오기
            data = self.historic_ohlcv(pair=pair, timeframe=timeframe)
        return data

    def get_analyzed_dataframe(self, pair: str, timeframe: str) -> Tuple[DataFrame, datetime]:
        #분석된 데이터 프레임 검색
        pair_key = (pair, timeframe)
        if pair_key in self.__cached_pairs:
            if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
                df, date = self.__cached_pairs[pair_key]
            else:
                df, date = self.__cached_pairs[pair_key]
                if self.__slice_index is not None:
                    max_index = self.__slice_index
                    df = df.iloc[max(0, max_index - MAX_DATAFRAME_CANDLES):max_index]
            return df, date
        else:
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))

    @property
    def runmode(self) -> RunMode:
        return RunMode(self._config.get('runmode', RunMode.OTHER))

    def current_whitelist(self) -> List[str]:
        if self._pairlists:
            return self._pairlists.whitelist.copy()
        else:
            raise OperationalException()

    def clear_cache(self):
        self.__cached_pairs = {}
        self.__cached_pairs_backtesting = {}
        self.__slice_index = 0

    def refresh(self,
                pairlist: ListPairsWithTimeframes,
                helping_pairs: ListPairsWithTimeframes = None) -> None:

        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        if helping_pairs:
            self._exchange.refresh_latest_ohlcv(pairlist + helping_pairs)
        else:
            self._exchange.refresh_latest_ohlcv(pairlist)

    @property
    def available_pairs(self) -> ListPairsWithTimeframes:
        #튜플 목록을 반환
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        return list(self._exchange._klines.keys())

    def ohlcv(self, pair: str, timeframe: str = None, copy: bool = True) -> DataFrame:
        """
        데이터를 DataFrame으로 가져오기
        """
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            return self._exchange.klines((pair, timeframe or self._config['timeframe']),
                                         copy=copy)
        else:
            return DataFrame()

    def market(self, pair: str) -> Optional[Dict[str, Any]]:
        #마켓 데이터 반환
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        return self._exchange.markets.get(pair)

    def ticker(self, pair: str):
        #마지막 티커 반환
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        try:
            return self._exchange.fetch_ticker(pair)
        except ExchangeError:
            return {}

    def orderbook(self, pair: str, maximum: int) -> Dict[str, List]:
        #마지막 오더북 반환
        if self._exchange is None:
            raise OperationalException(NO_EXCHANGE_EXCEPTION)
        return self._exchange.fetch_l2_order_book(pair, maximum)
