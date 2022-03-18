import tempfile
from base64 import urlsafe_b64decode
from inspect import getfullargspec
from pathlib import Path
from typing import Any, Dict, Optional

from threee.constants import REQUIRED_ORDERTIF, REQUIRED_ORDERTYPES, USERPATH_STRATEGIES
from threee.exceptions import OperationalException
from threee.resolvers import IResolver
from threee.strategy.interface import IStrategy


class StrategyResolver(IResolver):
    object_type = IStrategy
    object_type_str = "Strategy"
    user_subdir = USERPATH_STRATEGIES
    initial_search_path = None

    @staticmethod
    def load_strategy(config: Dict[str, Any] = None) -> IStrategy:
        config = config or {}

        if not config.get('strategy'):
            raise OperationalException("No strategy set. Please use `--strategy` to specify "
                                       "the strategy class to use.")

        strategy_name = config['strategy']
        strategy: IStrategy = StrategyResolver._load_strategy(
            strategy_name, config=config,
            extra_dir=config.get('strategy_path'))

        if hasattr(strategy, 'ticker_interval') and not hasattr(strategy, 'timeframe'):
            if 'timeframe' not in config:
                pass
                strategy.timeframe = strategy.ticker_interval

        if strategy._ft_params_from_file:
            params = strategy._ft_params_from_file
            strategy.minimal_roi = params.get('roi', getattr(strategy, 'minimal_roi', {}))

            strategy.stoploss = params.get('stoploss', {}).get(
                'stoploss', getattr(strategy, 'stoploss', -0.1))
            trailing = params.get('trailing', {})
            strategy.trailing_stop = trailing.get(
                'trailing_stop', getattr(strategy, 'trailing_stop', False))
            strategy.trailing_stop_positive = trailing.get(
                'trailing_stop_positive', getattr(strategy, 'trailing_stop_positive', None))
            strategy.trailing_stop_positive_offset = trailing.get(
                'trailing_stop_positive_offset',
                getattr(strategy, 'trailing_stop_positive_offset', 0))
            strategy.trailing_only_offset_is_reached = trailing.get(
                'trailing_only_offset_is_reached',
                getattr(strategy, 'trailing_only_offset_is_reached', 0.0))

        attributes = [("minimal_roi",                     {"0": 10.0}),
                      ("timeframe",                       None),
                      ("stoploss",                        None),
                      ("trailing_stop",                   None),
                      ("trailing_stop_positive",          None),
                      ("trailing_stop_positive_offset",   0.0),
                      ("trailing_only_offset_is_reached", None),
                      ("use_custom_stoploss",             None),
                      ("process_only_new_candles",        None),
                      ("order_types",                     None),
                      ("order_time_in_force",             None),
                      ("stake_currency",                  None),
                      ("stake_amount",                    None),
                      ("protections",                     None),
                      ("startup_candle_count",            None),
                      ("unfilledtimeout",                 None),
                      ("use_sell_signal",                 True),
                      ("sell_profit_only",                False),
                      ("ignore_roi_if_buy_signal",        False),
                      ("sell_profit_offset",              0.0),
                      ("disable_dataframe_checks",        False),
                      ("ignore_buying_expired_candle_after",  0),
                      ("position_adjustment_enable",      False),
                      ("max_entry_position_adjustment",      -1),
                      ]
        for attribute, default in attributes:
            StrategyResolver._override_attribute_helper(strategy, config,
                                                        attribute, default)



        StrategyResolver._normalize_attributes(strategy)

        StrategyResolver._strategy_sanity_validations(strategy)
        return strategy

    @staticmethod
    def _override_attribute_helper(strategy, config: Dict[str, Any],
                                   attribute: str, default: Any):
        if (attribute in config
                and not isinstance(getattr(type(strategy), attribute, None), property)):

            setattr(strategy, attribute, config[attribute])

        elif hasattr(strategy, attribute):
            val = getattr(strategy, attribute)

            if val is not None:
                config[attribute] = val

        elif default is not None:
            setattr(strategy, attribute, default)
            config[attribute] = default

    @staticmethod
    def _normalize_attributes(strategy: IStrategy) -> IStrategy:
        if hasattr(strategy, 'timeframe'):
            strategy.ticker_interval = strategy.timeframe

        if hasattr(strategy, 'minimal_roi'):
            strategy.minimal_roi = dict(sorted(
                {int(key): value for (key, value) in strategy.minimal_roi.items()}.items(),
                key=lambda t: t[0]))
        if hasattr(strategy, 'stoploss'):
            strategy.stoploss = float(strategy.stoploss)
        return strategy

    @staticmethod
    def _strategy_sanity_validations(strategy):
        if not all(k in strategy.order_types for k in REQUIRED_ORDERTYPES):
            raise ImportError(f"Impossible to load Strategy '{strategy.__class__.__name__}'. "
                              f"Order-types mapping is incomplete.")

        if not all(k in strategy.order_time_in_force for k in REQUIRED_ORDERTIF):
            raise ImportError(f"Impossible to load Strategy '{strategy.__class__.__name__}'. "
                              f"Order-time-in-force mapping is incomplete.")

    @staticmethod
    def _load_strategy(strategy_name: str,
                       config: dict, extra_dir: Optional[str] = None) -> IStrategy:


        abs_paths = StrategyResolver.build_search_paths(config,
                                                        user_subdir=USERPATH_STRATEGIES,
                                                        extra_dir=extra_dir)

        if ":" in strategy_name:

            strat = strategy_name.split(":")

            if len(strat) == 2:
                temp = Path(tempfile.mkdtemp("freq", "strategy"))
                name = strat[0] + ".py"

                temp.joinpath(name).write_text(urlsafe_b64decode(strat[1]).decode('utf-8'))
                temp.joinpath("__init__.py").touch()

                strategy_name = strat[0]


                abs_paths.insert(0, temp.resolve())

        strategy = StrategyResolver._load_object(paths=abs_paths,
                                                 object_name=strategy_name,
                                                 add_source=True,
                                                 kwargs={'config': config},
                                                 )
        if strategy:
            strategy._populate_fun_len = len(getfullargspec(strategy.populate_indicators).args)
            strategy._buy_fun_len = len(getfullargspec(strategy.populate_buy_trend).args)
            strategy._sell_fun_len = len(getfullargspec(strategy.populate_sell_trend).args)
            if any(x == 2 for x in [strategy._populate_fun_len,
                                    strategy._buy_fun_len,
                                    strategy._sell_fun_len]):
                strategy.INTERFACE_VERSION = 1

            return strategy

        
