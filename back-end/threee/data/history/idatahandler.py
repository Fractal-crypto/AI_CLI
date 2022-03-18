import logging
from abc import ABC, abstractclassmethod, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Type

from pandas import DataFrame

from threee import misc
from threee.configuration import TimeRange
from threee.constants import ListPairsWithTimeframes, TradeList
from threee.data.converter import clean_ohlcv_dataframe, trades_remove_duplicates, trim_dataframe
from threee.exchange import timeframe_to_seconds

"""
디스크에 데이터 저장
"""


class IDataHandler(ABC):

    def __init__(self, datadir: Path) -> None:
        self._datadir = datadir

    @classmethod
    def _get_file_extension(cls) -> str:
        raise NotImplementedError()

    @abstractclassmethod
    def ohlcv_get_available_data(cls, datadir: Path) -> ListPairsWithTimeframes:
        """
        저장위치에 ohlcv 데이터 확인
        """
    @abstractclassmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        각종목의 Ohlcv 데이터 확인
        """

    @abstractmethod
    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        데이터 저장
        """

    @abstractmethod
    def _ohlcv_load(self, pair: str, timeframe: str,
                    timerange: Optional[TimeRange] = None,
                    ) -> DataFrame:
        """
        한종목당 한 파일로 저장
        pandas 데이터 프레임 이용해서 변환
        """

    def ohlcv_purge(self, pair: str, timeframe: str) -> bool:
        """
        모든데이터 지우기
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        if filename.exists():
            filename.unlink()
            return True
        return False

    @abstractmethod
    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        데이터 붙이기
        """

    @abstractclassmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        불러올수있는 각 종목데이터 가져오기
        """

    @abstractmethod
    def trades_store(self, pair: str, data: TradeList) -> None:
        """
        리스트와 딕셔너리 형태로 거래데이터 저장
        """

    @abstractmethod
    def trades_append(self, pair: str, data: TradeList):
        """
        존재하는 데이터에 붙이기
        """

    @abstractmethod
    def _trades_load(self, pair: str, timerange: Optional[TimeRange] = None) -> TradeList:
        """
        json 현태로 종목리스트 반환
        """

    def trades_purge(self, pair: str) -> bool:
        """
        지정 종목의 데이터 지우기
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        if filename.exists():
            filename.unlink()
            return True
        return False

    def trades_load(self, pair: str, timerange: Optional[TimeRange] = None) -> TradeList:
        """
        데이터를  json 파일 형식으로 로드
        """
        return trades_remove_duplicates(self._trades_load(pair, timerange=timerange))

    @classmethod
    def _pair_data_filename(cls, datadir: Path, pair: str, timeframe: str) -> Path:
        pair_s = misc.pair_to_filename(pair)
        filename = datadir.joinpath(f'{pair_s}-{timeframe}.{cls._get_file_extension()}')
        return filename

    @classmethod
    def _pair_trades_filename(cls, datadir: Path, pair: str) -> Path:
        pair_s = misc.pair_to_filename(pair)
        filename = datadir.joinpath(f'{pair_s}-trades.{cls._get_file_extension()}')
        return filename

    def ohlcv_load(self, pair, timeframe: str,
                   timerange: Optional[TimeRange] = None,
                   fill_missing: bool = True,
                   drop_incomplete: bool = True,
                   startup_candles: int = 0,
                   warn_no_data: bool = True
                   ) -> DataFrame:
        """
        각 캔들의 ohlcv 데이터 로드
        """
        # 기간 재지정
        timerange_startup = deepcopy(timerange)
        if startup_candles > 0 and timerange_startup:
            timerange_startup.subtract_start(timeframe_to_seconds(timeframe) * startup_candles)

        pairdf = self._ohlcv_load(pair, timeframe,
                                  timerange=timerange_startup)
        if self._check_empty_df(pairdf, pair, timeframe, warn_no_data):
            return pairdf
        else:
            enddate = pairdf.iloc[-1]['date']

            if timerange_startup:
                self._validate_pairdata(pair, pairdf, timeframe, timerange_startup)
                pairdf = trim_dataframe(pairdf, timerange_startup)
                if self._check_empty_df(pairdf, pair, timeframe, warn_no_data):
                    return pairdf

            # 마지막 기간까지 불러오기
            pairdf = clean_ohlcv_dataframe(pairdf, timeframe,
                                           pair=pair,
                                           fill_missing=fill_missing,
                                           drop_incomplete=(drop_incomplete and
                                                            enddate == pairdf.iloc[-1]['date']))
            self._check_empty_df(pairdf, pair, timeframe, warn_no_data)
            return pairdf

    def _check_empty_df(self, pairdf: DataFrame, pair: str, timeframe: str, warn_no_data: bool):
        """
        비어있는 데이터 경고표시
        """
        if pairdf.empty:
            if warn_no_data:
                None
            return True
        return False

    def _validate_pairdata(self, pair, pairdata: DataFrame, timeframe: str, timerange: TimeRange):
        """
        비어있는데 데이터 표시
        """

        if timerange.starttype == 'date':
            start = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)

        if timerange.stoptype == 'date':
            stop = datetime.fromtimestamp(timerange.stopts, tz=timezone.utc)



def get_datahandlerclass(datatype: str) -> Type[IDataHandler]:
    """
    모든 데이터 관리 클래스 불러오기
    """

    if datatype == 'json':
        from .jsondatahandler import JsonDataHandler
        return JsonDataHandler
    elif datatype == 'jsongz':
        from .jsondatahandler import JsonGzDataHandler
        return JsonGzDataHandler
    elif datatype == 'hdf5':
        from .hdf5datahandler import HDF5DataHandler
        return HDF5DataHandler
    else:
        raise ValueError()


def get_datahandler(datadir: Path, data_format: str = None,
                    data_handler: IDataHandler = None) -> IDataHandler:

    if not data_handler:
        HandlerClass = get_datahandlerclass(data_format or 'json')
        data_handler = HandlerClass(datadir)
    return data_handler
