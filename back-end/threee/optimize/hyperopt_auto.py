from contextlib import suppress
from typing import Callable, Dict, List

from threee.exceptions import OperationalException


with suppress(ImportError):
    from skopt.space import Dimension

from threee.optimize.hyperopt_interface import EstimatorType, IHyperOpt



def _format_exception_message(space: str, ignore_missing_space: bool) -> None:
    pass


class HyperOptAuto(IHyperOpt):

    def _get_func(self, name) -> Callable:
        hyperopt_cls = getattr(self.strategy, 'HyperOpt', None)
        default_func = getattr(super(), name)
        if hyperopt_cls:
            return getattr(hyperopt_cls, name, default_func)
        else:
            return default_func

    def _generate_indicator_space(self, category):
        for attr_name, attr in self.strategy.enumerate_parameters(category):
            if attr.optimize:
                yield attr.get_space(attr_name)

    def _get_indicator_space(self, category) -> List:
        indicator_space = list(self._generate_indicator_space(category))
        if len(indicator_space) > 0:
            return indicator_space
        else:
            _format_exception_message(
                category,
                self.config.get("hyperopt_ignore_missing_space", False))
            return []

    def buy_indicator_space(self) -> List['Dimension']:
        return self._get_indicator_space('buy')

    def sell_indicator_space(self) -> List['Dimension']:
        return self._get_indicator_space('sell')

    def protection_space(self) -> List['Dimension']:
        return self._get_indicator_space('protection')

    def generate_roi_table(self, params: Dict) -> Dict[int, float]:
        return self._get_func('generate_roi_table')(params)

    def roi_space(self) -> List['Dimension']:
        return self._get_func('roi_space')()

    def stoploss_space(self) -> List['Dimension']:
        return self._get_func('stoploss_space')()

    def generate_trailing_params(self, params: Dict) -> Dict:
        return self._get_func('generate_trailing_params')(params)

    def trailing_space(self) -> List['Dimension']:
        return self._get_func('trailing_space')()

    def generate_estimator(self, dimensions: List['Dimension'], **kwargs) -> EstimatorType:
        return self._get_func('generate_estimator')(dimensions=dimensions, **kwargs)
