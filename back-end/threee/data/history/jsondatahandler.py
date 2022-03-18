import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from pandas import DataFrame, read_json, to_datetime

from threee import misc
from threee.configuration import TimeRange
from threee.constants import DEFAULT_DATAFRAME_COLUMNS, ListPairsWithTimeframes, TradeList
from threee.data.converter import trades_dict_to_list

from .idatahandler import IDataHandler



class JsonDataHandler(IDataHandler):

    _use_zip = False
    _columns = DEFAULT_DATAFRAME_COLUMNS

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path) -> ListPairsWithTimeframes:
        """
        데이터 리턴가능한 종목 데이터 가져오기
        """
        _tmp = [re.search(r'^([a-zA-Z_]+)\-(\d+\S+)(?=.json)', p.name)
                for p in datadir.glob(f"*.{cls._get_file_extension()}")]
        return [(match[1].replace('_', '/'), match[2]) for match in _tmp
                if match and len(match.groups()) > 1]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        저장된 ohlcv 데이터 불러오기
        """

        _tmp = [re.search(r'^(\S+)(?=\-' + timeframe + '.json)', p.name)
                for p in datadir.glob(f"*{timeframe}.{cls._get_file_extension()}")]
        return [match[0].replace('_', '/') for match in _tmp if match]

    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        value값 데이터 리스트형식으로 변환 해서 가져오기
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        _data = data.copy()
        # 기간을 int형 변환
        _data['date'] = _data['date'].view(np.int64) // 1000 // 1000
        # 리셋후  json 으로 저장
        _data.reset_index(drop=True).loc[:, self._columns].to_json(
            filename, orient="values",
            compression='gzip' if self._use_zip else None)

    def _ohlcv_load(self, pair: str, timeframe: str,
                    timerange: Optional[TimeRange] = None,
                    ) -> DataFrame:
        """
        디스크에 저장된 ohlcv 데이터 로드
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        if not filename.exists():
            return DataFrame(columns=self._columns)
        try:
            pairdata = read_json(filename, orient='values')
            pairdata.columns = self._columns
        except ValueError:
            return DataFrame(columns=self._columns)
        pairdata = pairdata.astype(dtype={'open': 'float', 'high': 'float',
                                          'low': 'float', 'close': 'float', 'volume': 'float'})
        pairdata['date'] = to_datetime(pairdata['date'],
                                       unit='ms',
                                       utc=True,
                                       infer_datetime_format=True)
        return pairdata

    def ohlcv_purge(self, pair: str, timeframe: str) -> bool:
        """
        모든데이터 지우기
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        if filename.exists():
            filename.unlink()
            return True
        return False

    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        존재하는 데이터에 Ohlcv 데이터 붙이기
        """
        raise NotImplementedError()

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        트레이딩 가능한 종목 리턴
        """
        _tmp = [re.search(r'^(\S+)(?=\-trades.json)', p.name)
                for p in datadir.glob(f"*trades.{cls._get_file_extension()}")]
        return [match[0].replace('_', '/') for match in _tmp if match]

    def trades_store(self, pair: str, data: TradeList) -> None:
        """
        가능한 종목을 리스트 형태로 변환하여 저장
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        misc.file_dump_json(filename, data, is_zip=self._use_zip)

    def trades_append(self, pair: str, data: TradeList):
        """
        종목을 리스트형태로 저장
        """
        raise NotImplementedError()

    def _trades_load(self, pair: str, timerange: Optional[TimeRange] = None) -> TradeList:
        """
        json 파일 형태로 저장 된 데이터 로드
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        tradesdata = misc.file_load_json(filename)

        if not tradesdata:
            return []

        if isinstance(tradesdata[0], dict):
            tradesdata = trades_dict_to_list(tradesdata)
            pass
        return tradesdata

    @classmethod
    def _get_file_extension(cls):
        return "json.gz" if cls._use_zip else "json"


class JsonGzDataHandler(JsonDataHandler):

    _use_zip = True
