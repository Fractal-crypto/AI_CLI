import logging
import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arrow
from pandas import DataFrame, concat

from threee.configuration import TimeRange
from threee.constants import DEFAULT_DATAFRAME_COLUMNS
from threee.data.converter import (clean_ohlcv_dataframe, ohlcv_to_dataframe,
                                      trades_remove_duplicates, trades_to_ohlcv)
from threee.data.history.idatahandler import IDataHandler, get_datahandler
from threee.exceptions import OperationalException
from threee.exchange import Exchange
from threee.misc import format_ms_time


def load_pair_history(pair: str,
                      timeframe: str,
                      datadir: Path, *,
                      timerange: Optional[TimeRange] = None,
                      fill_up_missing: bool = True,
                      drop_incomplete: bool = True,
                      startup_candles: int = 0,
                      data_format: str = None,
                      data_handler: IDataHandler = None,
                      ) -> DataFrame:
    """
    주어진 종목의 ohlcv 데이터 불러오기 종목, 데이터프레임, 데이터 저장장소, 기간, 시작-끝 데이터
    빠진 데이터로 구성
    """
    data_handler = get_datahandler(datadir, data_format, data_handler)

    return data_handler.ohlcv_load(pair=pair,
                                   timeframe=timeframe,
                                   timerange=timerange,
                                   fill_missing=fill_up_missing,
                                   drop_incomplete=drop_incomplete,
                                   startup_candles=startup_candles,
                                   )


def load_data(datadir: Path,
              timeframe: str,
              pairs: List[str], *,
              timerange: Optional[TimeRange] = None,
              fill_up_missing: bool = True,
              startup_candles: int = 0,
              fail_without_data: bool = False,
              data_format: str = 'json',
              ) -> Dict[str, DataFrame]:
    """
    주어진 종목의 ohlcv 데이터 불러오기 리스트
    """
    result: Dict[str, DataFrame] = {}


    data_handler = get_datahandler(datadir, data_format)

    for pair in pairs:
        hist = load_pair_history(pair=pair, timeframe=timeframe,
                                 datadir=datadir, timerange=timerange,
                                 fill_up_missing=fill_up_missing,
                                 startup_candles=startup_candles,
                                 data_handler=data_handler
                                 )
        if not hist.empty:
            result[pair] = hist

    if fail_without_data and not result:
        raise OperationalException("데이터 없음.. 종료")
    return result


def refresh_data(datadir: Path,
                 timeframe: str,
                 pairs: List[str],
                 exchange: Exchange,
                 data_format: str = None,
                 timerange: Optional[TimeRange] = None,
                 ) -> None:
    """
    ohlcv데이터 다시가져오기
    """
    data_handler = get_datahandler(datadir, data_format)
    for idx, pair in enumerate(pairs):
        process = f'{idx}/{len(pairs)}'
        _download_pair_history(pair=pair, process=process,
                               timeframe=timeframe, datadir=datadir,
                               timerange=timerange, exchange=exchange, data_handler=data_handler)


def _load_cached_data_for_updating(pair: str, timeframe: str, timerange: Optional[TimeRange],
                                   data_handler: IDataHandler) -> Tuple[DataFrame, Optional[int]]:
    """
    데이터 전부다 더 가져오고 이미 있는 데이터는 오버라이드하며 저장
    """
    start = None
    if timerange:
        if timerange.starttype == 'date':
            start = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)

    data = data_handler.ohlcv_load(pair, timeframe=timeframe,
                                   timerange=None, fill_missing=False,
                                   drop_incomplete=True, warn_no_data=False)
    if not data.empty:
        if start and start < data.iloc[0]['date']:

            data = DataFrame(columns=DEFAULT_DATAFRAME_COLUMNS)
        else:
            start = data.iloc[-1]['date']

    start_ms = int(start.timestamp() * 1000) if start else None
    return data, start_ms

def _download_pair_history(pair: str, *,
                           datadir: Path,
                           exchange: Exchange,
                           timeframe: str = '5m',
                           process: str = '',
                           new_pairs_days: int = 30,
                           data_handler: IDataHandler = None,
                           timerange: Optional[TimeRange] = None) -> bool:
    """
    마지막으로 가져온 데이터를 인식하고 다운로드 하면 그다음부터 다운로드 하도록 설계
    """
    data_handler = get_datahandler(datadir, data_handler=data_handler)

    try:
        data, since_ms = _load_cached_data_for_updating(pair, timeframe, timerange,
                                                        data_handler=data_handler)

        # 최소데이터 30일 데이터 가져오도록 설정
        new_data = exchange.get_historic_ohlcv(pair=pair,
                                               timeframe=timeframe,
                                               since_ms=since_ms if since_ms else
                                               arrow.utcnow().shift(
                                                   days=-new_pairs_days).int_timestamp * 1000,
                                               is_new_pair=data.empty
                                               )

        new_dataframe = ohlcv_to_dataframe(new_data, timeframe, pair,
                                           fill_missing=False, drop_incomplete=True)
        if data.empty:
            data = new_dataframe
        else:
            # 존재하는 데이터는 전부 지워버리고 새로 다운로드
            data = clean_ohlcv_dataframe(concat([data, new_dataframe], axis=0), timeframe, pair,
                                         fill_missing=False, drop_incomplete=False)

        data_handler.ohlcv_store(pair, timeframe, data=data)
        return True

    except Exception:
        None
        return False

def refresh_backtest_ohlcv_data(exchange: Exchange, pairs: List[str], timeframes: List[str],
                                datadir: Path, timerange: Optional[TimeRange] = None,
                                new_pairs_days: int = 30, erase: bool = False,
                                data_format: str = None) -> List[str]:
    """
    백테스팅에 필요한 ohlcv 데이터 다시 불러오기
    """
    pairs_not_available = []
    data_handler = get_datahandler(datadir, data_format)
    for idx, pair in enumerate(pairs, start=1):
        if pair not in exchange.markets:
            pairs_not_available.append(pair)
            continue
        for timeframe in timeframes:
            process = f'{idx}/{len(pairs)}'
            _download_pair_history(pair=pair, process=process,
                                   datadir=datadir, exchange=exchange,
                                   timerange=timerange, data_handler=data_handler,
                                   timeframe=str(timeframe), new_pairs_days=new_pairs_days)
    return pairs_not_available

def _download_trades_history(exchange: Exchange,
                             pair: str, *,
                             new_pairs_days: int = 30,
                             timerange: Optional[TimeRange] = None,
                             data_handler: IDataHandler
                             ) -> bool:
    """
    거래 기록 데이터 거래소에서 가져오기
    """
    try:
        until = None
        if (timerange and timerange.starttype == 'date'):
            since = timerange.startts * 1000
            if timerange.stoptype == 'date':
                until = timerange.stopts * 1000
        else:
            since = arrow.utcnow().shift(days=-new_pairs_days).int_timestamp * 1000

        trades = data_handler.trades_load(pair)

        if trades and since < trades[0][0]:

            trades = []

        from_id = trades[-1][1] if trades else None
        if trades and since < trades[-1][0]:
            since = trades[-1][0] - (5 * 1000)


        new_trades = exchange.get_historic_trades(pair=pair,
                                                  since=since,
                                                  until=until,
                                                  from_id=from_id,
                                                  )
        trades.extend(new_trades[1])

        trades = trades_remove_duplicates(trades)
        data_handler.trades_store(pair, data=trades)

        return True

    except Exception:
        None
        return False


def refresh_backtest_trades_data(exchange: Exchange, pairs: List[str], datadir: Path,
                                 timerange: TimeRange, new_pairs_days: int = 30,
                                 erase: bool = False, data_format: str = 'jsongz') -> List[str]:
    """
    백테스팅용 트레이딩 데이터 다시 불러오기
    """
    pairs_not_available = []
    data_handler = get_datahandler(datadir, data_format=data_format)
    for pair in pairs:
        if pair not in exchange.markets:
            pairs_not_available.append(pair)

            continue

        _download_trades_history(exchange=exchange,
                                 pair=pair,
                                 new_pairs_days=new_pairs_days,
                                 timerange=timerange,
                                 data_handler=data_handler)
    return pairs_not_available


def convert_trades_to_ohlcv(pairs: List[str], timeframes: List[str],
                            datadir: Path, timerange: TimeRange, erase: bool = False,
                            data_format_ohlcv: str = 'json',
                            data_format_trades: str = 'jsongz') -> None:
    """
    저장된 트레이딩 데이터 ohlcv 로 변환
    """
    data_handler_trades = get_datahandler(datadir, data_format=data_format_trades)
    data_handler_ohlcv = get_datahandler(datadir, data_format=data_format_ohlcv)

    for pair in pairs:
        trades = data_handler_trades.trades_load(pair)
        for timeframe in timeframes:

            try:
                ohlcv = trades_to_ohlcv(trades, timeframe)
                # 저장
                data_handler_ohlcv.ohlcv_store(pair, timeframe, data=ohlcv)
            except ValueError:
                None


def get_timerange(data: Dict[str, DataFrame]) -> Tuple[datetime, datetime]:
    """
    최대로 백테스팅 데이터 적용해서 데이터 불러오기 예) 100일 데이터 전부 불러와서 백테스팅
    """
    timeranges = [
        (frame['date'].min().to_pydatetime(), frame['date'].max().to_pydatetime())
        for frame in data.values()
    ]
    return (min(timeranges, key=operator.itemgetter(0))[0],
            max(timeranges, key=operator.itemgetter(1))[1])

def validate_backtest_data(data: DataFrame, pair: str, min_date: datetime,
                           max_date: datetime, timeframe_min: int) -> bool:
    """
    만약에 백테스팅 ohlcv 데이터가 부족하게 들어고면 경고 예)20200404의 데이터가 없음
    """
    # 시간프레임 변환
    expected_frames = int((max_date - min_date).total_seconds() // 60 // timeframe_min)
    found_missing = False
    dflen = len(data)

    return found_missing
