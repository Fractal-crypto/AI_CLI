import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from threee.configuration import TimeRange
from threee.constants import (DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS,
                                 ListPairsWithTimeframes, TradeList)

from .idatahandler import IDataHandler

class HDF5DataHandler(IDataHandler):

    _columns = DEFAULT_DATAFRAME_COLUMNS

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path) -> ListPairsWithTimeframes:
        """
        모든 종목과 각각의 기간 데이터를 출력
        """
        _tmp = [re.search(r'^([a-zA-Z_]+)\-(\d+\S+)(?=.h5)', p.name)
                for p in datadir.glob("*.h5")]
        return [(match[1].replace('_', '/'), match[2]) for match in _tmp
                if match and len(match.groups()) > 1]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        각 종목의 기간데이터에 맞는 ohlcv 데이터를 가져오기
        """
        _tmp = [re.search(r'^(\S+)(?=\-' + timeframe + '.h5)', p.name)
                for p in datadir.glob(f"*{timeframe}.h5")]
        return [match[0].replace('_', '/') for match in _tmp if match]

    def ohlcv_store(self, pair: str, timeframe: str, data: pd.DataFrame) -> None:
        """
        모든 데이터를 hdf5 형식으로 저장
        구성은 종목, 타임프레임 , ohlcv 데이터로 리턴
        """
        key = self._pair_ohlcv_key(pair, timeframe)
        _data = data.copy()
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        _data.loc[:, self._columns].to_hdf(
            filename, key, mode='a', complevel=9, complib='blosc',
            format='table', data_columns=['date']
        )

    def _ohlcv_load(self, pair: str, timeframe: str,
                    timerange: Optional[TimeRange] = None) -> pd.DataFrame:
        """
        저장된 ohlcv 데이터 불러오기
        각각의 타임프레임, 종목에 따라 다르게 데이터 로딩
        """
        key = self._pair_ohlcv_key(pair, timeframe)
        filename = self._pair_data_filename(self._datadir, pair, timeframe)

        if not filename.exists():
            return pd.DataFrame(columns=self._columns)
        where = []
        if timerange:
            if timerange.starttype == 'date':
                where.append(f"date >= Timestamp({timerange.startts * 1e9})")
            if timerange.stoptype == 'date':
                where.append(f"date <= Timestamp({timerange.stopts * 1e9})")

        pairdata = pd.read_hdf(filename, key=key, mode="r", where=where)

        pairdata = pairdata.astype(dtype={'open': 'float', 'high': 'float',
                                          'low': 'float', 'close': 'float', 'volume': 'float'})
        return pairdata

    def ohlcv_append(self, pair: str, timeframe: str, data: pd.DataFrame) -> None:
        """
        저장된 ohlc 파일에 이어서 데이터 붙이기
        """
        raise NotImplementedError()

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        모든 트레이딩 가능한 데이터 불러오기
        """
        _tmp = [re.search(r'^(\S+)(?=\-trades.h5)', p.name)
                for p in datadir.glob("*trades.h5")]
        return [match[0].replace('_', '/') for match in _tmp if match]

    def trades_store(self, pair: str, data: TradeList) -> None:
        """
        데이터를 리스트형식으로 변환하여 저장
        """
        key = self._pair_trades_key(pair)
        pd.DataFrame(data, columns=DEFAULT_TRADES_COLUMNS).to_hdf(
            self._pair_trades_filename(self._datadir, pair), key,
            mode='a', complevel=9, complib='blosc',
            format='table', data_columns=['timestamp']
        )

    def trades_append(self, pair: str, data: TradeList):
        """
        종목데이터 연결해서 붙이기
        """
        raise NotImplementedError()

    def _trades_load(self, pair: str, timerange: Optional[TimeRange] = None) -> TradeList:
        """
        hdf5 파일 불러오고 각종목 리스트 변환
        """
        key = self._pair_trades_key(pair)
        filename = self._pair_trades_filename(self._datadir, pair)

        if not filename.exists():
            return []
        where = []
        if timerange:
            if timerange.starttype == 'date':
                where.append(f"timestamp >= {timerange.startts * 1e3}")
            if timerange.stoptype == 'date':
                where.append(f"timestamp < {timerange.stopts * 1e3}")

        trades: pd.DataFrame = pd.read_hdf(filename, key=key, mode="r", where=where)
        trades[['id', 'type']] = trades[['id', 'type']].replace({np.nan: None})
        return trades.values.tolist()

    @classmethod
    def _get_file_extension(cls):
        return "h5"

    @classmethod
    def _pair_ohlcv_key(cls, pair: str, timeframe: str) -> str:
        return f"{pair}/ohlcv/tf_{timeframe}"

    @classmethod
    def _pair_trades_key(cls, pair: str) -> str:
        return f"{pair}/trades"
