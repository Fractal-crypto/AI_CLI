import logging
import math
from abc import ABC
from typing import Dict, List, Union

from sklearn.base import RegressorMixin
from skopt.space import Categorical, Dimension, Integer

from threee.exchange import timeframe_to_minutes
from threee.misc import round_dict
from threee.optimize.space import SKDecimal
from threee.strategy import IStrategy


logger = logging.getLogger(__name__)

EstimatorType = Union[RegressorMixin, str]


class IHyperOpt(ABC):
    """
    hyperopt용 인터페이스
    """
    ticker_interval: str
    timeframe: str
    strategy: IStrategy

    def __init__(self, config: dict) -> None:
        self.config = config
        IHyperOpt.ticker_interval = str(config['timeframe'])
        IHyperOpt.timeframe = str(config['timeframe'])

    def generate_estimator(self, dimensions: List[Dimension], **kwargs) -> EstimatorType:
        """
        base_estimator를 반환
        RegressorMixin에서 상속
        """
        return 'ET'

    def generate_roi_table(self, params: Dict) -> Dict[int, float]:
        roi_table = {}
        roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
        roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
        roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
        roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

        return roi_table

    def roi_space(self) -> List[Dimension]:
        roi_t_alpha = 1.0
        roi_p_alpha = 1.0

        timeframe_min = timeframe_to_minutes(self.timeframe)

        roi_t_scale = timeframe_min / 5
        roi_p_scale = math.log1p(timeframe_min) / math.log1p(5)
        roi_limits = {
            'roi_t1_min': int(10 * roi_t_scale * roi_t_alpha),
            'roi_t1_max': int(120 * roi_t_scale * roi_t_alpha),
            'roi_t2_min': int(10 * roi_t_scale * roi_t_alpha),
            'roi_t2_max': int(60 * roi_t_scale * roi_t_alpha),
            'roi_t3_min': int(10 * roi_t_scale * roi_t_alpha),
            'roi_t3_max': int(40 * roi_t_scale * roi_t_alpha),
            'roi_p1_min': 0.01 * roi_p_scale * roi_p_alpha,
            'roi_p1_max': 0.04 * roi_p_scale * roi_p_alpha,
            'roi_p2_min': 0.01 * roi_p_scale * roi_p_alpha,
            'roi_p2_max': 0.07 * roi_p_scale * roi_p_alpha,
            'roi_p3_min': 0.01 * roi_p_scale * roi_p_alpha,
            'roi_p3_max': 0.20 * roi_p_scale * roi_p_alpha,
        }

        p = {
            'roi_t1': roi_limits['roi_t1_min'],
            'roi_t2': roi_limits['roi_t2_min'],
            'roi_t3': roi_limits['roi_t3_min'],
            'roi_p1': roi_limits['roi_p1_min'],
            'roi_p2': roi_limits['roi_p2_min'],
            'roi_p3': roi_limits['roi_p3_min'],
        }

        p = {
            'roi_t1': roi_limits['roi_t1_max'],
            'roi_t2': roi_limits['roi_t2_max'],
            'roi_t3': roi_limits['roi_t3_max'],
            'roi_p1': roi_limits['roi_p1_max'],
            'roi_p2': roi_limits['roi_p2_max'],
            'roi_p3': roi_limits['roi_p3_max'],
        }

        return [
            Integer(roi_limits['roi_t1_min'], roi_limits['roi_t1_max'], name='roi_t1'),
            Integer(roi_limits['roi_t2_min'], roi_limits['roi_t2_max'], name='roi_t2'),
            Integer(roi_limits['roi_t3_min'], roi_limits['roi_t3_max'], name='roi_t3'),
            SKDecimal(roi_limits['roi_p1_min'], roi_limits['roi_p1_max'], decimals=3,
                      name='roi_p1'),
            SKDecimal(roi_limits['roi_p2_min'], roi_limits['roi_p2_max'], decimals=3,
                      name='roi_p2'),
            SKDecimal(roi_limits['roi_p3_min'], roi_limits['roi_p3_max'], decimals=3,
                      name='roi_p3'),
        ]

    def stoploss_space(self) -> List[Dimension]:
        return [
            SKDecimal(-0.35, -0.02, decimals=3, name='stoploss'),
        ]

    def generate_trailing_params(self, params: Dict) -> Dict:
        return {
            'trailing_stop': params['trailing_stop'],
            'trailing_stop_positive': params['trailing_stop_positive'],
            'trailing_stop_positive_offset': (params['trailing_stop_positive'] +
                                              params['trailing_stop_positive_offset_p1']),
            'trailing_only_offset_is_reached': params['trailing_only_offset_is_reached'],
        }

    def trailing_space(self) -> List[Dimension]:
        return [
            # It was decided to always set trailing_stop is to True if the 'trailing' hyperspace
            # is used. Otherwise hyperopt will vary other parameters that won't have effect if
            # trailing_stop is set False.
            # This parameter is included into the hyperspace dimensions rather than assigning
            # it explicitly in the code in order to have it printed in the results along with
            # other 'trailing' hyperspace parameters.
            Categorical([True], name='trailing_stop'),

            SKDecimal(0.01, 0.35, decimals=3, name='trailing_stop_positive'),

            # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
            # so this intermediate parameter is used as the value of the difference between
            # them. The value of the 'trailing_stop_positive_offset' is constructed in the
            # generate_trailing_params() method.
            # This is similar to the hyperspace dimensions used for constructing the ROI tables.
            SKDecimal(0.001, 0.1, decimals=3, name='trailing_stop_positive_offset_p1'),

            Categorical([True, False], name='trailing_only_offset_is_reached'),
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['timeframe'] = self.timeframe
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        IHyperOpt.ticker_interval = state['timeframe']
        IHyperOpt.timeframe = state['timeframe']
