"""
Functions to convert data from one format to another
"""
import itertools
import logging
from datetime import datetime, timezone
from operator import itemgetter
from typing import Any, Dict, List

import pandas as pd
from pandas import DataFrame, to_datetime

from threee.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS, TradeList


def ohlcv_to_dataframe(ohlcv: list, timeframe: str, pair: str, *,
                       fill_missing: bool = True, drop_incomplete: bool = True) -> DataFrame:
    """
    OHLCV 데이터가 있는 목록을 변환
    """

    cols = DEFAULT_DATAFRAME_COLUMNS
    df = DataFrame(ohlcv, columns=cols)
    df['date'] = to_datetime(df['date'], unit='ms', utc=True, infer_datetime_format=True)
    df = df.astype(dtype={'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float',
                          'volume': 'float'})
    return clean_ohlcv_dataframe(df, timeframe, pair,
                                 fill_missing=fill_missing,
                                 drop_incomplete=drop_incomplete)


def clean_ohlcv_dataframe(data: DataFrame, timeframe: str, pair: str, *,
                          fill_missing: bool = True,
                          drop_incomplete: bool = True) -> DataFrame:
    """
    모든 데이터 지우기
    """
    data = data.groupby(by='date', as_index=False, sort=True).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'max',
    })

    if drop_incomplete:
        data.drop(data.tail(1).index, inplace=True)
    if fill_missing:
        return ohlcv_fill_up_missing_data(data, timeframe, pair)
    else:
        return data


def ohlcv_fill_up_missing_data(dataframe: DataFrame, timeframe: str, pair: str) -> DataFrame:
    """
    없는 데이터는 모두 0으로 변환

    """
    from threee.exchange import timeframe_to_minutes

    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    timeframe_minutes = timeframe_to_minutes(timeframe)
    df = dataframe.resample(f'{timeframe_minutes}min', on='date').agg(ohlcv_dict)
    df['close'] = df['close'].fillna(method='ffill')
    df.loc[:, ['open', 'high', 'low']] = df[['open', 'high', 'low']].fillna(
        value={'open': df['close'],
               'high': df['close'],
               'low': df['close'],
               })
    df.reset_index(inplace=True)
    len_before = len(dataframe)
    len_after = len(df)
    pct_missing = (len_after - len_before) / len_before if len_before > 0 else 0


    return df


def trim_dataframe(df: DataFrame, timerange, df_date_col: str = 'date',
                   startup_candles: int = 0) -> DataFrame:
    """
    주어진 timerang을 기반으로 데이터 프레임 자르기
    """
    if startup_candles:
        df = df.iloc[startup_candles:, :]
    else:
        if timerange.starttype == 'date':
            start = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)
            df = df.loc[df[df_date_col] >= start, :]
    if timerange.stoptype == 'date':
        stop = datetime.fromtimestamp(timerange.stopts, tz=timezone.utc)
        df = df.loc[df[df_date_col] <= stop, :]
    return df


def trim_dataframes(preprocessed: Dict[str, DataFrame], timerange,
                    startup_candles: int) -> Dict[str, DataFrame]:
    """
    분석된 데이터 프레임에서 시작 기간 다듬기
    """
    processed: Dict[str, DataFrame] = {}

    for pair, df in preprocessed.items():
        trimed_df = trim_dataframe(df, timerange, startup_candles=startup_candles)
        if not trimed_df.empty:
            processed[pair] = trimed_df
        else:
            None
    return processed


def order_book_to_dataframe(bids: list, asks: list) -> DataFrame:
    """
    주문량 과 거래 사이즈
    """
    cols = ['bids', 'b_size']

    bids_frame = DataFrame(bids, columns=cols)
    bids_frame['b_sum'] = bids_frame['b_size'].cumsum()
    cols2 = ['asks', 'a_size']
    asks_frame = DataFrame(asks, columns=cols2)
    asks_frame['a_sum'] = asks_frame['a_size'].cumsum()

    frame = pd.concat([bids_frame['b_sum'], bids_frame['b_size'], bids_frame['bids'],
                       asks_frame['asks'], asks_frame['a_size'], asks_frame['a_sum']], axis=1,
                      keys=['b_sum', 'b_size', 'bids', 'asks', 'a_size', 'a_sum'])
    return frame


def trades_remove_duplicates(trades: List[List]) -> List[List]:
    #거래 목록에서 중복 제거
    return [i for i, _ in itertools.groupby(sorted(trades, key=itemgetter(0)))]


def trades_dict_to_list(trades: List[Dict]) -> TradeList:
    #트레이딩 결과 리스트로 변환
    return [[t[col] for col in DEFAULT_TRADES_COLUMNS] for t in trades]


def trades_to_ohlcv(trades: TradeList, timeframe: str) -> DataFrame:
    """
    ohlcv 데이터 리스트로변환
    """
    from threee.exchange import timeframe_to_minutes
    timeframe_minutes = timeframe_to_minutes(timeframe)
    if not trades:
        raise ValueError('Trade-list empty.')
    df = pd.DataFrame(trades, columns=DEFAULT_TRADES_COLUMNS)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms',
                                     utc=True,)
    df = df.set_index('timestamp')
    df_new = df['price'].resample(f'{timeframe_minutes}min').ohlc()
    df_new['volume'] = df['amount'].resample(f'{timeframe_minutes}min').sum()
    df_new['date'] = df_new.index

    df_new = df_new.dropna()
    return df_new.loc[:, DEFAULT_DATAFRAME_COLUMNS]


def convert_trades_format(config: Dict[str, Any], convert_from: str, convert_to: str, erase: bool):
    """
    거래데이터 다른 형식으로 변환
    """
    from threee.data.history.idatahandler import get_datahandler
    src = get_datahandler(config['datadir'], convert_from)
    trg = get_datahandler(config['datadir'], convert_to)

    if 'pairs' not in config:
        config['pairs'] = src.trades_get_pairs(config['datadir'])

    for pair in config['pairs']:
        data = src.trades_load(pair=pair)
        trg.trades_store(pair, data)
        if erase and convert_from != convert_to:
            src.trades_purge(pair=pair)


def convert_ohlcv_format(config: Dict[str, Any], convert_from: str, convert_to: str, erase: bool):
    """
    ohlcv데이터 다른 형식으로 변환
    """
    from threee.data.history.idatahandler import get_datahandler
    src = get_datahandler(config['datadir'], convert_from)
    trg = get_datahandler(config['datadir'], convert_to)
    timeframes = config.get('timeframes', [config.get('timeframe')])

    if 'pairs' not in config:
        config['pairs'] = []
        for timeframe in timeframes:
            config['pairs'].extend(src.ohlcv_get_pairs(config['datadir'],
                                                       timeframe))

    for timeframe in timeframes:
        for pair in config['pairs']:
            data = src.ohlcv_load(pair=pair, timeframe=timeframe,
                                  timerange=None,
                                  fill_missing=False,
                                  drop_incomplete=False,
                                  startup_candles=0)
            if len(data) > 0:
                trg.ohlcv_store(pair=pair, timeframe=timeframe, data=data)
                if erase and convert_from != convert_to:
                    src.ohlcv_purge(pair=pair, timeframe=timeframe)
