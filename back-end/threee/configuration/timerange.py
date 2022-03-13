import logging
import re
from datetime import datetime
from typing import Optional

import arrow

from threee.exceptions import OperationalException

class TimeRange:
    """
    데이터 분석 기간 지정
    """

    def __init__(self, starttype: Optional[str] = None, stoptype: Optional[str] = None,
                 startts: int = 0, stopts: int = 0):

        self.starttype: Optional[str] = starttype
        self.stoptype: Optional[str] = stoptype
        self.startts: int = startts
        self.stopts: int = stopts

    def __eq__(self, other):
        #오버라이드 형식
        return (self.starttype == other.starttype and self.stoptype == other.stoptype
                and self.startts == other.startts and self.stopts == other.stopts)

    def subtract_start(self, seconds: int) -> None:
        """
        데이터 학습기간 두번째 값을 통해 자르기
        """
        if self.startts:
            self.startts = self.startts - seconds

    def adjust_start_if_necessary(self, timeframe_secs: int, startup_candles: int,
                                  min_date: datetime) -> None:
        """
        데이터 학습 기간과 타임프레임 시간을 재지정 바람
        """
        if (not self.starttype or (startup_candles
                                   and min_date.timestamp() >= self.startts)):
            # 기간 지정안해주면 지정된 자동 기간 학습
            self.startts = int(min_date.timestamp() + timeframe_secs * startup_candles)
            self.starttype = 'date'

    @staticmethod
    def parse_timerange(text: Optional[str]) -> 'TimeRange':
        """
        커멘드에 넣을 기간 지정 예) 20220303-20230303
        """
        if text is None:
            return TimeRange(None, None, 0, 0)
        syntax = [(r'^-(\d{8})$', (None, 'date')),
                  (r'^(\d{8})-$', ('date', None)),
                  (r'^(\d{8})-(\d{8})$', ('date', 'date')),
                  (r'^-(\d{10})$', (None, 'date')),
                  (r'^(\d{10})-$', ('date', None)),
                  (r'^(\d{10})-(\d{10})$', ('date', 'date')),
                  (r'^-(\d{13})$', (None, 'date')),
                  (r'^(\d{13})-$', ('date', None)),
                  (r'^(\d{13})-(\d{13})$', ('date', 'date')),
                  ]
        for rex, stype in syntax:

            match = re.match(rex, text)
            if match:
                rvals = match.groups()
                index = 0
                start: int = 0
                stop: int = 0
                if stype[0]:
                    starts = rvals[index]
                    if stype[0] == 'date' and len(starts) == 8:
                        start = arrow.get(starts, 'YYYYMMDD').int_timestamp
                    elif len(starts) == 13:
                        start = int(starts) // 1000
                    else:
                        start = int(starts)
                    index += 1
                if stype[1]:
                    stops = rvals[index]
                    if stype[1] == 'date' and len(stops) == 8:
                        stop = arrow.get(stops, 'YYYYMMDD').int_timestamp
                    elif len(stops) == 13:
                        stop = int(stops) // 1000
                    else:
                        stop = int(stops)
                if start > stop > 0:
                    raise OperationalException("시작 기간이 끝나는 시간 변경")
                return TimeRange(stype[0], stype[1], start, stop)
        raise OperationalException("형식이 다릅니다")
