# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
This module contains the hyperopt logic
"""

import logging
import random
import warnings
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

import progressbar
import rapidjson
from colorama import Fore, Style
from colorama import init as colorama_init
from joblib import Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects
from pandas import DataFrame

from threee.constants import DATETIME_PRINT_FORMAT, FTHYPT_FILEVERSION, LAST_BT_RESULT_FN
from threee.data.converter import trim_dataframes
from threee.data.history import get_timerange
from threee.exceptions import OperationalException
from threee.misc import deep_merge_dicts, file_dump_json, plural
from threee.optimize.backtesting import Backtesting
# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
from threee.optimize.hyperopt_auto import HyperOptAuto
from threee.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from threee.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from threee.optimize.hyperopt_tools import HyperoptTools, hyperopt_serializer
from threee.optimize.optimize_reports import generate_strategy_stats
from threee.resolvers.hyperopt_resolver import HyperOptLossResolver


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Dimension

progressbar.streams.wrap_stderr()
progressbar.streams.wrap_stdout()
logger = logging.getLogger(__name__)


INITIAL_POINTS = 30
SKOPT_MODEL_QUEUE_SIZE = 10
MAX_LOSS = 100000


class Hyperopt:
    """
    백테스팅 기능를 이용해 학습
    """
    custom_hyperopt: IHyperOpt

    def __init__(self, config: Dict[str, Any]) -> None:
        self.buy_space: List[Dimension] = []
        self.sell_space: List[Dimension] = []
        self.protection_space: List[Dimension] = []
        self.roi_space: List[Dimension] = []
        self.stoploss_space: List[Dimension] = []
        self.trailing_space: List[Dimension] = []
        self.dimensions: List[Dimension] = []

        self.config = config

        self.backtesting = Backtesting(self.config)
        self.pairlist = self.backtesting.pairlists.whitelist

        if not self.config.get('hyperopt'):
            self.custom_hyperopt = HyperOptAuto(self.config)
        else:
            pass

        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        self.custom_hyperopt.strategy = self.backtesting.strategy

        self.custom_hyperoptloss = HyperOptLossResolver.load_hyperoptloss(self.config)
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        strategy = str(self.config['strategy'])
        self.results_file: Path = (self.config['user_data_dir'] / 'hyperopt_results' /
                                   f'strategy_{strategy}_{time_now}.fthypt')
        self.data_pickle_file = (self.config['user_data_dir'] /
                                 'hyperopt_results' / 'hyperopt_tickerdata.pkl')
        self.total_epochs = config.get('epochs', 0)

        self.current_best_loss = 100

        self.clean_hyperopt()

        self.num_epochs_saved = 0
        self.current_best_epoch: Optional[Dict[str, Any]] = None

        if self.config.get('use_max_market_positions', True):
            self.max_open_trades = self.config['max_open_trades']
        else:
            pass
            self.max_open_trades = 0
        self.position_stacking = self.config.get('position_stacking', False)

        if HyperoptTools.has_space(self.config, 'sell'):
            if 'ask_strategy' not in self.config:
                self.config['ask_strategy'] = {}
            self.config['ask_strategy']['use_sell_signal'] = True

        self.print_all = self.config.get('print_all', False)
        self.hyperopt_table_header = 0
        self.print_colorized = self.config.get('print_colorized', False)
        self.print_json = self.config.get('print_json', False)

    @staticmethod
    def get_lock_filename(config: Dict[str, Any]) -> str:

        return str(config['user_data_dir'] / 'hyperopt.lock')

    def clean_hyperopt(self) -> None:
        """
        저장된 config 파일 리셋
        """
        for f in [self.data_pickle_file, self.results_file]:
            p = Path(f)
            if p.is_file():
                p.unlink()

    def _get_params_dict(self, dimensions: List[Dimension], raw_params: List[Any]) -> Dict:

        if len(raw_params) != len(dimensions):
            raise ValueError('Mismatch in number of search-space dimensions.')

        return {d.name: v for d, v in zip(dimensions, raw_params)}

    def _save_result(self, epoch: Dict) -> None:
        """
        결과 저장
        """
        epoch[FTHYPT_FILEVERSION] = 2
        with self.results_file.open('a') as f:
            rapidjson.dump(epoch, f, default=hyperopt_serializer,
                           number_mode=rapidjson.NM_NATIVE | rapidjson.NM_NAN)
            f.write("\n")

        self.num_epochs_saved += 1

        latest_filename = Path.joinpath(self.results_file.parent, LAST_BT_RESULT_FN)
        file_dump_json(latest_filename, {'latest_hyperopt': str(self.results_file.name)},
                       log=False)

    def _get_params_details(self, params: Dict) -> Dict:
        """
        각 에폭마다 결과 출력
        """
        result: Dict = {}

        if HyperoptTools.has_space(self.config, 'buy'):
            result['buy'] = {p.name: params.get(p.name) for p in self.buy_space}
        if HyperoptTools.has_space(self.config, 'sell'):
            result['sell'] = {p.name: params.get(p.name) for p in self.sell_space}
        if HyperoptTools.has_space(self.config, 'protection'):
            result['protection'] = {p.name: params.get(p.name) for p in self.protection_space}
        if HyperoptTools.has_space(self.config, 'roi'):
            result['roi'] = {str(k): v for k, v in
                             self.custom_hyperopt.generate_roi_table(params).items()}
        if HyperoptTools.has_space(self.config, 'stoploss'):
            result['stoploss'] = {p.name: params.get(p.name) for p in self.stoploss_space}
        if HyperoptTools.has_space(self.config, 'trailing'):
            result['trailing'] = self.custom_hyperopt.generate_trailing_params(params)

        return result

    def _get_no_optimize_details(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        strategy = self.backtesting.strategy
        if not HyperoptTools.has_space(self.config, 'roi'):
            result['roi'] = {str(k): v for k, v in strategy.minimal_roi.items()}
        if not HyperoptTools.has_space(self.config, 'stoploss'):
            result['stoploss'] = {'stoploss': strategy.stoploss}
        if not HyperoptTools.has_space(self.config, 'trailing'):
            result['trailing'] = {
                'trailing_stop': strategy.trailing_stop,
                'trailing_stop_positive': strategy.trailing_stop_positive,
                'trailing_stop_positive_offset': strategy.trailing_stop_positive_offset,
                'trailing_only_offset_is_reached': strategy.trailing_only_offset_is_reached,
            }
        return result

    def print_results(self, results) -> None:
        is_best = results['is_best']

        if self.print_all or is_best:
            print(
                HyperoptTools.get_result_table(
                    self.config, results, self.total_epochs,
                    self.print_all, self.print_colorized,
                    self.hyperopt_table_header
                )
            )
            self.hyperopt_table_header = 2

    def init_spaces(self):
        """
        최적화 공간에 공간 할당
        """
        if HyperoptTools.has_space(self.config, 'protection'):
            self.config['enable_protections'] = True
            self.protection_space = self.custom_hyperopt.protection_space()

        if HyperoptTools.has_space(self.config, 'buy'):
            self.buy_space = self.custom_hyperopt.buy_indicator_space()

        if HyperoptTools.has_space(self.config, 'sell'):
            self.sell_space = self.custom_hyperopt.sell_indicator_space()

        if HyperoptTools.has_space(self.config, 'roi'):
            self.roi_space = self.custom_hyperopt.roi_space()

        if HyperoptTools.has_space(self.config, 'stoploss'):
            self.stoploss_space = self.custom_hyperopt.stoploss_space()

        if HyperoptTools.has_space(self.config, 'trailing'):
            self.trailing_space = self.custom_hyperopt.trailing_space()

        self.dimensions = (self.buy_space + self.sell_space + self.protection_space
                           + self.roi_space + self.stoploss_space + self.trailing_space)

    def assign_params(self, params_dict: Dict, category: str) -> None:
        """
        최적화 가능한 매개변수 할당
        """
        for attr_name, attr in self.backtesting.strategy.enumerate_parameters(category):
            if attr.optimize:
                attr.value = params_dict[attr_name]

    def generate_optimizer(self, raw_params: List[Any], iteration=None) -> Dict:
        """
        구성된 모든 것을 최적화하기 위해 Epoch당 한 번 호출
        """
        backtest_start_time = datetime.now(timezone.utc)
        params_dict = self._get_params_dict(self.dimensions, raw_params)

        if HyperoptTools.has_space(self.config, 'buy'):
            self.assign_params(params_dict, 'buy')

        if HyperoptTools.has_space(self.config, 'sell'):
            self.assign_params(params_dict, 'sell')

        if HyperoptTools.has_space(self.config, 'protection'):
            self.assign_params(params_dict, 'protection')

        if HyperoptTools.has_space(self.config, 'roi'):
            self.backtesting.strategy.minimal_roi = (  # type: ignore
                self.custom_hyperopt.generate_roi_table(params_dict))

        if HyperoptTools.has_space(self.config, 'stoploss'):
            self.backtesting.strategy.stoploss = params_dict['stoploss']

        if HyperoptTools.has_space(self.config, 'trailing'):
            d = self.custom_hyperopt.generate_trailing_params(params_dict)
            self.backtesting.strategy.trailing_stop = d['trailing_stop']
            self.backtesting.strategy.trailing_stop_positive = d['trailing_stop_positive']
            self.backtesting.strategy.trailing_stop_positive_offset = \
                d['trailing_stop_positive_offset']
            self.backtesting.strategy.trailing_only_offset_is_reached = \
                d['trailing_only_offset_is_reached']

        with self.data_pickle_file.open('rb') as f:
            processed = load(f, mmap_mode='r')
        bt_results = self.backtesting.backtest(
            processed=processed,
            start_date=self.min_date,
            end_date=self.max_date,
            max_open_trades=self.max_open_trades,
            position_stacking=self.position_stacking,
            enable_protections=self.config.get('enable_protections', False),
        )
        backtest_end_time = datetime.now(timezone.utc)
        bt_results.update({
            'backtest_start_time': int(backtest_start_time.timestamp()),
            'backtest_end_time': int(backtest_end_time.timestamp()),
        })

        return self._get_results_dict(bt_results, self.min_date, self.max_date,
                                      params_dict,
                                      processed=processed)

    def _get_results_dict(self, backtesting_results, min_date, max_date,
                          params_dict, processed: Dict[str, DataFrame]
                          ) -> Dict[str, Any]:
        params_details = self._get_params_details(params_dict)

        strat_stats = generate_strategy_stats(
            self.pairlist, self.backtesting.strategy.get_strategy_name(),
            backtesting_results, min_date, max_date, market_change=0
        )
        results_explanation = HyperoptTools.format_results_explanation_string(
            strat_stats, self.config['stake_currency'])

        not_optimized = self.backtesting.strategy.get_no_optimize_params()
        not_optimized = deep_merge_dicts(not_optimized, self._get_no_optimize_details())

        trade_count = strat_stats['total_trades']
        total_profit = strat_stats['profit_total']
        loss: float = MAX_LOSS
        if trade_count >= self.config['hyperopt_min_trades']:
            loss = self.calculate_loss(results=backtesting_results['results'],
                                       trade_count=trade_count,
                                       min_date=min_date, max_date=max_date,
                                       config=self.config, processed=processed,
                                       backtest_stats=strat_stats)
        return {
            'loss': loss,
            'params_dict': params_dict,
            'params_details': params_details,
            'params_not_optimized': not_optimized,
            'results_metrics': strat_stats,
            'results_explanation': results_explanation,
            'total_profit': total_profit,
        }

    def get_optimizer(self, dimensions: List[Dimension], cpu_count) -> Optimizer:
        estimator = self.custom_hyperopt.generate_estimator(dimensions=dimensions)

        acq_optimizer = "sampling"
        if isinstance(estimator, str):
            if estimator not in ("GP", "RF", "ET", "GBRT"):
                raise OperationalException(f"Estimator {estimator} not supported.")
            else:
                acq_optimizer = "auto"
        return Optimizer(
            dimensions,
            base_estimator=estimator,
            acq_optimizer=acq_optimizer,
            n_initial_points=INITIAL_POINTS,
            acq_optimizer_kwargs={'n_jobs': cpu_count},
            random_state=self.random_state,
            model_queue_size=SKOPT_MODEL_QUEUE_SIZE,
        )

    def run_optimizer_parallel(self, parallel, asked, i) -> List:
        return parallel(delayed(
                        wrap_non_picklable_objects(self.generate_optimizer))(v, i) for v in asked)

    def _set_random_state(self, random_state: Optional[int]) -> int:
        return random_state or random.randint(1, 2**16 - 1)

    def prepare_hyperopt_data(self) -> None:
        data, timerange = self.backtesting.load_bt_data()

        preprocessed = self.backtesting.strategy.advise_all_indicators(data)

        processed = trim_dataframes(preprocessed, timerange, self.backtesting.required_startup)
        self.min_date, self.max_date = get_timerange(processed)

        dump(preprocessed, self.data_pickle_file)

    def start(self) -> None:
        self.random_state = self._set_random_state(self.config.get('hyperopt_random_state', None))
        self.hyperopt_table_header = -1
        self.init_spaces()

        self.prepare_hyperopt_data()

        self.backtesting.exchange.close()
        self.backtesting.exchange._api = None
        self.backtesting.exchange._api_async = None
        self.backtesting.exchange.loop = None

        cpus = cpu_count()
        config_jobs = self.config.get('hyperopt_jobs', -1)

        self.opt = self.get_optimizer(self.dimensions, config_jobs)

        if self.print_colorized:
            colorama_init(autoreset=True)

        try:
            with Parallel(n_jobs=config_jobs) as parallel:
                jobs = parallel._effective_n_jobs()

                if self.print_colorized:
                    widgets = [
                        ' [Epoch ', progressbar.Counter(), ' of ', str(self.total_epochs),
                        ' (', progressbar.Percentage(), ')] ',
                        progressbar.Bar(marker=progressbar.AnimatedMarker(
                            fill='\N{FULL BLOCK}',
                            fill_wrap=Fore.GREEN + '{}' + Fore.RESET,
                            marker_wrap=Style.BRIGHT + '{}' + Style.RESET_ALL,
                        )),
                        ' [', progressbar.ETA(), ', ', progressbar.Timer(), ']',
                    ]
                else:
                    widgets = [
                        ' [Epoch ', progressbar.Counter(), ' of ', str(self.total_epochs),
                        ' (', progressbar.Percentage(), ')] ',
                        progressbar.Bar(marker=progressbar.AnimatedMarker(
                            fill='\N{FULL BLOCK}',
                        )),
                        ' [', progressbar.ETA(), ', ', progressbar.Timer(), ']',
                    ]
                with progressbar.ProgressBar(
                    max_value=self.total_epochs, redirect_stdout=False, redirect_stderr=False,
                    widgets=widgets
                ) as pbar:
                    EVALS = ceil(self.total_epochs / jobs)
                    for i in range(EVALS):
                        n_rest = (i + 1) * jobs - self.total_epochs
                        current_jobs = jobs - n_rest if n_rest > 0 else jobs

                        asked = self.opt.ask(n_points=current_jobs)
                        f_val = self.run_optimizer_parallel(parallel, asked, i)
                        self.opt.tell(asked, [v['loss'] for v in f_val])

                        for j, val in enumerate(f_val):
                            current = i * jobs + j + 1
                            val['current_epoch'] = current
                            val['is_initial_point'] = current <= INITIAL_POINTS

                            is_best = HyperoptTools.is_best_loss(val, self.current_best_loss)
                            val['is_best'] = is_best
                            self.print_results(val)

                            if is_best:
                                self.current_best_loss = val['loss']
                                self.current_best_epoch = val

                            self._save_result(val)

                            pbar.update(current)

        except KeyboardInterrupt:
            print('중지')

        if self.current_best_epoch:
            HyperoptTools.try_export_params(
                self.config,
                self.backtesting.strategy.get_strategy_name(),
                self.current_best_epoch)

            HyperoptTools.show_epoch_details(self.current_best_epoch, self.total_epochs,
                                             self.print_json)
        else:
            pass
