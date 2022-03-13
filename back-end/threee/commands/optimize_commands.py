import logging
from typing import Any, Dict

from threee import constants
from threee.configuration import setup_utils_configuration
from threee.enums import RunMode
from threee.exceptions import OperationalException
from threee.misc import round_coin_value


def setup_optimize_configuration(args: Dict[str, Any], method: RunMode) -> Dict[str, Any]:
    """
    베이지안 최적화 위해 config파일 기본 설정
    """
    config = setup_utils_configuration(args, method)

    no_unlimited_runmodes = {
        RunMode.BACKTEST: 'backtesting',
        RunMode.HYPEROPT: 'hyperoptimization',
    }
    if method in no_unlimited_runmodes.keys():
        wallet_size = config['dry_run_wallet'] * config['tradable_balance_ratio']

        if (config['stake_amount'] != constants.UNLIMITED_STAKE_AMOUNT
                and config['stake_amount'] > wallet_size):
            wallet = round_coin_value(wallet_size, config['stake_currency'])
            stake = round_coin_value(config['stake_amount'], config['stake_currency'])
            raise OperationalException("...")

    return config


def start_backtesting(args: Dict[str, Any]) -> None:
    """
    백테스팅 시작
    """
    from threee.optimize.backtesting import Backtesting

    config = setup_optimize_configuration(args, RunMode.BACKTEST)
    backtesting = Backtesting(config)
    backtesting.start()


def start_backtesting_show(args: Dict[str, Any]) -> None:
    """
    백테스팅 데이터 불러오기
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from threee.data.btanalysis import load_backtest_stats
    from threee.optimize.optimize_reports import show_backtest_results, show_sorted_pairlist

    results = load_backtest_stats(config['exportfilename'])

    show_backtest_results(config, results)
    show_sorted_pairlist(config, results)


def start_hyperopt(args: Dict[str, Any]) -> None:
    """
    베이지안 최적화 커멘드 실행 함수
    """
    try:
        from filelock import FileLock, Timeout

        from threee.optimize.hyperopt import Hyperopt
    except ImportError as e:
        raise OperationalException("...") from e
    config = setup_optimize_configuration(args, RunMode.HYPEROPT)

    lock = FileLock(Hyperopt.get_lock_filename(config))

    try:
        with lock.acquire(timeout=1):

            # 백테스팅 실행
            hyperopt = Hyperopt(config)
            hyperopt.start()

    except Timeout:
        None
