from pathlib import Path
from typing import Dict

from threee.constants import HYPEROPT_LOSS_BUILTIN, USERPATH_HYPEROPTS
from threee.exceptions import OperationalException
from threee.optimize.hyperopt_loss_interface import IHyperOptLoss
from threee.resolvers import IResolver


class HyperOptLossResolver(IResolver):
    object_type = IHyperOptLoss
    object_type_str = "HyperoptLoss"
    user_subdir = USERPATH_HYPEROPTS
    initial_search_path = Path(__file__).parent.parent.joinpath('optimize').resolve()

    @staticmethod
    def load_hyperoptloss(config: Dict) -> IHyperOptLoss:
        hyperoptloss_name = config.get('hyperopt_loss')

        hyperoptloss = HyperOptLossResolver.load_object(hyperoptloss_name,
                                                        config, kwargs={},
                                                        extra_dir=config.get('hyperopt_path'))

        hyperoptloss.__class__.ticker_interval = str(config['timeframe'])
        hyperoptloss.__class__.timeframe = str(config['timeframe'])

        return hyperoptloss
