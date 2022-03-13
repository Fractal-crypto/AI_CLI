"""
This module contains the configuration class
"""
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from threee import constants
from threee.configuration.check_exchange import check_exchange

from threee.configuration.directory_operations import create_datadir, create_userdata_dir
from threee.configuration.environment_vars import enironment_vars_to_dict
from threee.configuration.load_config import load_config_file, load_file
from threee.enums import NON_UTIL_MODES, TRADING_MODES, RunMode
from threee.exceptions import OperationalException
from threee.loggers import setup_logging
from threee.misc import deep_merge_dicts, parse_db_uri_for_logging



class Configuration:
    """
    config파일에 있는 모든 데이터를 가공하며 트레이딩, 백테스팅, 최적화 부분에서 모두 사용
    """

    def __init__(self, args: Dict[str, Any], runmode: RunMode = None) -> None:
        self.args = args
        self.config: Optional[Dict[str, Any]] = None
        self.runmode = runmode

    def get_config(self) -> Dict[str, Any]:
        """
        config 파일 불러오기
        """
        if self.config is None:
            self.config = self.load_config()
        return self.config

    @staticmethod
    def from_files(files: List[str]) -> Dict[str, Any]:
        """
        모든 데이터를 불러오고 다시 통합
        같은 데이터는 오버라이트 하며
        Runs through the whole Configuration initialization, so all expected config entries
        딕셔너리 형태로 반환
        """
        c = Configuration({'config': files}, RunMode.OTHER)
        return c.get_config()

    def load_from_files(self, files: List[str]) -> Dict[str, Any]:
        #로드
        config: Dict[str, Any] = {}

        if not files:
            return deepcopy(constants.MINIMAL_CONFIG)

        for path in files:
            #오버라이드
            config = deep_merge_dicts(load_config_file(path), config)

        # 각 환경 불러오기
        env_data = enironment_vars_to_dict()
        config = deep_merge_dicts(env_data, config)

        config['config_files'] = files

        if 'internals' not in config:
            config['internals'] = {}
        if 'ask_strategy' not in config:
            config['ask_strategy'] = {}

        if 'pairlists' not in config:
            config['pairlists'] = []

        return config

    def load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        # 데이터 전부 로드
        config: Dict[str, Any] = self.load_from_files(self.args.get("config", []))

        config['original_config'] = deepcopy(config)
        self._process_runmode(config)
        self._process_common_options(config)
        self._process_trading_options(config)
        self._process_optimize_options(config)
        self._process_plot_options(config)
        self._process_data_options(config)

        # 거래소 확인 후 실제 가능한지 확인
        check_exchange(config, config.get('experimental', {}).get('block_bad_exchanges', True))

        self._resolve_pairs_list(config)


        return config



    def _process_trading_options(self, config: Dict[str, Any]) -> None:
        if config['runmode'] not in TRADING_MODES:
            return

        if config.get('dry_run', False):

            if config.get('db_url') in [None, constants.DEFAULT_DB_PROD_URL]:
                config['db_url'] = constants.DEFAULT_DB_DRYRUN_URL
        else:
            if not config.get('db_url', None):
                config['db_url'] = constants.DEFAULT_DB_PROD_URL


    def _process_common_options(self, config: Dict[str, Any]) -> None:

        # 기본적으로 지정 되지 않은 전략 가져오기
        if self.args.get('strategy') or not config.get('strategy'):
            config.update({'strategy': self.args.get('strategy')})

        self._args_to_config(config, argname='strategy_path',
                             logstring='{}를 통해 위치 재지정')

        if ('db_url' in self.args and self.args['db_url'] and
                self.args['db_url'] != constants.DEFAULT_DB_PROD_URL):
            config.update({'db_url': self.args['db_url']})

        if config.get('forcebuy_enable', False):
            None

    def _process_datadir_options(self, config: Dict[str, Any]) -> None:
        """
        유저데이터 파일 정보 각각 정리후 가져오기
        """
        # 거래소 정보 여기서 재지정.
        if 'exchange' in self.args and self.args['exchange']:
            config['exchange']['name'] = self.args['exchange']
            logger.info(f"Using exchange {config['exchange']['name']}")

        if 'pair_whitelist' not in config['exchange']:
            config['exchange']['pair_whitelist'] = []

        if 'user_data_dir' in self.args and self.args['user_data_dir']:
            config.update({'user_data_dir': self.args['user_data_dir']})
        elif 'user_data_dir' not in config:
            config.update({'user_data_dir': str(Path.cwd() / 'user_data')})

        # 위치 재지정
        config['user_data_dir'] = create_userdata_dir(config['user_data_dir'], create_dir=False)
        config.update({'datadir': create_datadir(config, self.args.get('datadir', None))})

        if self.args.get('exportfilename'):
            self._args_to_config(config, argname='exportfilename',
                                 logstring='백테스팅 정보 저장')
            config['exportfilename'] = Path(config['exportfilename'])
        else:
            config['exportfilename'] = (config['user_data_dir']
                                        / 'backtest_results')

    def _process_optimize_options(self, config: Dict[str, Any]) -> None:

        # 최적화후 오버라이트를 통해서 전략 바로 수정
        self._args_to_config(config, argname='timeframe',
                             logstring='타임프레임 재지정')

        self._args_to_config(config, argname='position_stacking',
                             logstring='파라미터 재지정')

        self._args_to_config(
            config, argname='enable_protections',
            logstring='데이터 오류 파악중')

        if 'use_max_market_positions' in self.args and not self.args["use_max_market_positions"]:
            config.update({'use_max_market_positions': False})

        elif 'max_open_trades' in self.args and self.args['max_open_trades']:
            config.update({'max_open_trades': self.args['max_open_trades']})

        elif config['runmode'] in NON_UTIL_MODES:
            None
        # 최대 거래 지정
        if config.get('max_open_trades') == -1:
            config['max_open_trades'] = float('inf')

        if self.args.get('stake_amount', None):
            # cli  값 다시 지정
            try:
                self.args['stake_amount'] = float(self.args['stake_amount'])
            except ValueError:
                pass



        self._args_to_config(config, argname='timerange',
                             logstring='기간 지정 오버라이드')

        self._process_datadir_options(config)

        self._args_to_config(config, argname='timeframe',
                             logstring='타임프레임 오버라이드')

        # Edge section:
        if 'stoploss_range' in self.args and self.args["stoploss_range"]:
            txt_range = eval(self.args["stoploss_range"])
            config['edge'].update({'stoploss_range_min': txt_range[0]})
            config['edge'].update({'stoploss_range_max': txt_range[1]})
            config['edge'].update({'stoploss_range_step': txt_range[2]})


        # Hyperopt section
        self._args_to_config(config, argname='hyperopt',
                             logstring='Using Hyperopt class name: {}')

        self._args_to_config(config, argname='hyperopt_path',
                             logstring='Using additional Hyperopt lookup path: {}')

        self._args_to_config(config, argname='hyperoptexportfilename',
                             logstring='Using hyperopt file: {}')

        self._args_to_config(config, argname='epochs',
                             logstring='Parameter --epochs detected ... '
                             'Will run Hyperopt with for {} epochs ...'
                             )

        self._args_to_config(config, argname='spaces',
                             logstring='Parameter -s/--spaces detected: {}')

        self._args_to_config(config, argname='print_all',
                             logstring='Parameter --print-all detected ...')

        if 'print_colorized' in self.args and not self.args["print_colorized"]:

            config.update({'print_colorized': False})
        else:
            config.update({'print_colorized': True})



        self._args_to_config(config, argname='hyperopt_min_trades',
                             logstring='Parameter --min-trades detected: {}')

        self._args_to_config(config, argname='hyperopt_loss',
                             logstring='Using Hyperopt loss class name: {}')



        self._args_to_config(config, argname='hyperopt_list_min_trades',
                             logstring='Parameter --min-trades detected: {}')



    def _process_plot_options(self, config: Dict[str, Any]) -> None:

        self._args_to_config(config, argname='pairs',
                             logstring='Using pairs {}')

        self._args_to_config(config, argname='timeframes',
                             logstring='timeframes --timeframes: {}')

        self._args_to_config(config, argname='days',
                             logstring='Detected --days: {}')

        self._args_to_config(config, argname='download_trades',
                             logstring='Detected --dl-trades: {}')


    def _process_data_options(self, config: Dict[str, Any]) -> None:
        self._args_to_config(config, argname='new_pairs_days',
                             logstring='Detected --new-pairs-days: {}')

    def _process_runmode(self, config: Dict[str, Any]) -> None:

        self._args_to_config(config, argname='dry_run',
                             logstring='가상모드')

        if not self.runmode:
            # 실제 거래 모드
            self.runmode = RunMode.DRY_RUN if config.get('dry_run', True) else RunMode.LIVE
        config.update({'runmode': self.runmode})

    def _args_to_config(self, config: Dict[str, Any], argname: str,
                        logstring: str, logfun: Optional[Callable] = None,
                        deprecated_msg: Optional[str] = None) -> None:
        """
        딕셔너리 형태로 저장하고
        config파일에서 불러옴
        """
        if (argname in self.args and self.args[argname] is not None
           and self.args[argname] is not False):

            config.update({argname: self.args[argname]})

    def _resolve_pairs_list(self, config: Dict[str, Any]) -> None:
        """
        데이터 다운로드 커멘드 지원(사용 안하면 필요없는 기능)
        """

        if "pairs" in config:
            config['exchange']['pair_whitelist'] = config['pairs']
            return

        if "pairs_file" in self.args and self.args["pairs_file"]:
            pairs_file = Path(self.args["pairs_file"])
            # 명확히 지정 안하면
            if not pairs_file.exists():
                raise OperationalException(f'No pairs file found with path "{pairs_file}".')
            config['pairs'] = load_file(pairs_file)
            config['pairs'].sort()
            return

        if 'config' in self.args and self.args['config']:
            config['pairs'] = config.get('exchange', {}).get('pair_whitelist')
        else:
            pairs_file = config['datadir'] / 'pairs.json'
            if pairs_file.exists():
                config['pairs'] = load_file(pairs_file)
                if 'pairs' in config:
                    config['pairs'].sort()
