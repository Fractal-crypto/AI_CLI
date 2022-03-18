from enum import Enum


class BacktestState(Enum):
    """
     상태 정보
    """
    STARTUP = 1
    DATALOAD = 2
    ANALYZE = 3
    CONVERT = 4
    BACKTEST = 5

    def __str__(self):
        return f"{self.name.lower()}"
