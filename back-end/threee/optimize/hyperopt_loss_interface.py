from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

from pandas import DataFrame


class IHyperOptLoss(ABC):
    """
    hyperopt 손실 함수용 인터페이스
    """
    timeframe: str

    @staticmethod
    @abstractmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:
        """
        더 나은 결과를 위해 더 작은 수를 반환
        """
