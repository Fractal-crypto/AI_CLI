"""
argparse 이용해서 쉘 에서 커멘드 가능하도록 지원
"""
import argparse
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from threee.commands.cli_options import AVAILABLE_CLI_OPTIONS
from threee.constants import DEFAULT_CONFIG


ARGS_COMMON = [  "config"]

ARGS_STRATEGY = ["strategy"]

ARGS_TRADE = [ ]

ARGS_WEBSERVER: List[str] = []

ARGS_COMMON_OPTIMIZE = ["timeframe", "timerange",
                        "pairs"]

ARGS_BACKTEST = ARGS_COMMON_OPTIMIZE + []

ARGS_HYPEROPT = ARGS_COMMON_OPTIMIZE + ["hyperopt",


                                        "epochs", "spaces", "print_all",
                                        "print_colorized",
                                         "hyperopt_min_trades",
                                        "hyperopt_loss",
                                        ]

ARGS_EDGE = ARGS_COMMON_OPTIMIZE + []

ARGS_LIST_STRATEGIES = [ "print_colorized"]

ARGS_LIST_HYPEROPTS = [  "print_colorized"]

ARGS_BACKTEST_SHOW = []

ARGS_LIST_EXCHANGES = []

ARGS_LIST_TIMEFRAMES = ["exchange"]

ARGS_LIST_PAIRS = ["exchange"]

ARGS_TEST_PAIRLIST = ["config"]

ARGS_CREATE_USERDIR = []

ARGS_BUILD_CONFIG = ["config"]

ARGS_BUILD_STRATEGY = ["strategy"]

ARGS_CONVERT_DATA = ["pairs"]
ARGS_CONVERT_DATA_OHLCV = ARGS_CONVERT_DATA + ["timeframes"]

ARGS_CONVERT_TRADES = ["pairs", "timeframes", "exchange"]

ARGS_LIST_DATA = ["exchange", "pairs"]

ARGS_DOWNLOAD_DATA = ["pairs", "days",
                      "timerange", "exchange", "timeframes"]



ARGS_INSTALL_UI = []

ARGS_SHOW_TRADES = [ ]

ARGS_HYPEROPT_LIST = [
                      "print_colorized"]

ARGS_HYPEROPT_SHOW = [
                       ]

NO_CONF_REQURIED = ["convert-data", "convert-trade-data", "download-data", "list-timeframes",
                    "list-markets", "list-pairs", "list-strategies", "list-data",
                    "hyperopt-list", "hyperopt-show", "backtest-filter",
                    "plot-dataframe", "plot-profit", "show-trades", "trades-to-ohlcv"]

NO_CONF_ALLOWED = ["create-userdir", "list-exchanges", "new-strategy"]


class Arguments:
    """
    cli 에서 받은 커멘드 동작하도록 관리하는 클래스
    """

    def __init__(self, args: Optional[List[str]]) -> None:
        self.args = args
        self._parsed_arg: Optional[argparse.Namespace] = None

    def get_parsed_arg(self) -> Dict[str, Any]: # 받은 값 리스트형태로 리턴 함수

        if self._parsed_arg is None:
            self._build_subcommands()
            self._parsed_arg = self._parse_args()

        return vars(self._parsed_arg)

    def _parse_args(self) -> argparse.Namespace: # namespace로 리턴 하는 함수
        """
        유저데이터에서 config.josn 파일 불러오기
        """

        parsed_arg = self.parser.parse_args(self.args)

        if ('config' in parsed_arg and parsed_arg.config is None):
            conf_required = ('command' in parsed_arg and parsed_arg.command in NO_CONF_REQURIED)

            if 'user_data_dir' in parsed_arg and parsed_arg.user_data_dir is not None:
                user_dir = parsed_arg.user_data_dir
            else:
                user_dir = 'user_data'

            cfgfile = Path(user_dir) / DEFAULT_CONFIG
            if cfgfile.is_file():
                parsed_arg.config = [str(cfgfile)]
            else:
                cfgfile = Path.cwd() / DEFAULT_CONFIG
                if cfgfile.is_file() or not conf_required:
                    parsed_arg.config = [DEFAULT_CONFIG]

        return parsed_arg

    def _build_args(self, optionlist, parser):

        for val in optionlist:
            opt = AVAILABLE_CLI_OPTIONS[val]
            parser.add_argument(*opt.cli, dest=val, **opt.kwargs)

    def _build_subcommands(self) -> None:
        """
        common + _____ + ______ 각각의 명령어 연결
        """
        _common_parser = argparse.ArgumentParser(add_help=False)
        group = _common_parser.add_argument_group("Common arguments")
        self._build_args(optionlist=ARGS_COMMON, parser=group)

        _strategy_parser = argparse.ArgumentParser(add_help=False)
        strategy_group = _strategy_parser.add_argument_group("Strategy arguments")
        self._build_args(optionlist=ARGS_STRATEGY, parser=strategy_group)

        self.parser = argparse.ArgumentParser(description='hello this is TUK')


        from threee.commands import (start_backtesting, start_backtesting_show,
                                        start_convert_data, start_convert_trades,
                                        start_create_userdir, start_download_data, start_edge,
                                        start_hyperopt, start_hyperopt_list, start_hyperopt_show,
                                        start_install_ui, start_list_data, start_list_exchanges,
                                        start_list_markets, start_list_strategies,
                                        start_list_timeframes, start_new_strategy,
                                         start_show_trades,
                                        start_test_pairlist, start_trading, start_webserver)

        subparsers = self.parser.add_subparsers(dest='command',

                                                )

        # 트레이드 시작
        trade_cmd = subparsers.add_parser('trade', help='Trade module.',
                                          parents=[_common_parser, _strategy_parser])
        trade_cmd.set_defaults(func=start_trading)
        self._build_args(optionlist=ARGS_TRADE, parser=trade_cmd)


        # 데이터 다운로드
        download_data_cmd = subparsers.add_parser(
            'download-data',
            help='Download backtesting data.',
            parents=[_common_parser],
        )
        download_data_cmd.set_defaults(func=start_download_data)
        self._build_args(optionlist=ARGS_DOWNLOAD_DATA, parser=download_data_cmd)


        # 백테스팅
        backtesting_cmd = subparsers.add_parser('backtesting', help='Backtesting module.',
                                                parents=[_common_parser, _strategy_parser])
        backtesting_cmd.set_defaults(func=start_backtesting)
        self._build_args(optionlist=ARGS_BACKTEST, parser=backtesting_cmd)


        # 베이지안 옵티마져
        hyperopt_cmd = subparsers.add_parser('hyperopt', help='Hyperopt module.',
                                             parents=[_common_parser, _strategy_parser],
                                             )
        hyperopt_cmd.set_defaults(func=start_hyperopt)
        self._build_args(optionlist=ARGS_HYPEROPT, parser=hyperopt_cmd)
