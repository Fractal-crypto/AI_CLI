import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List
from threee.configuration import TimeRange, setup_utils_configuration
from threee.data.converter import convert_ohlcv_format, convert_trades_format
from threee.data.history import (convert_trades_to_ohlcv, refresh_backtest_ohlcv_data,refresh_backtest_trades_data)
from threee.enums import RunMode
from threee.exceptions import OperationalException
from threee.exchange import timeframe_to_minutes
from threee.exchange.exchange import market_is_active
from threee.plugins.pairlist.pairlist_helpers import expand_pairlist
from threee.resolvers import ExchangeResolver



def start_download_data(args: Dict[str, Any]) -> None:
    """
    데이터 다운로드
    """
    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    if 'days' in config and 'timerange' in config:
        raise OperationalException("--이후 days")
    timerange = TimeRange()
    if 'days' in config:
        time_since = (datetime.now() - timedelta(days=config['days'])).strftime("%Y%m%d")
        timerange = TimeRange.parse_timerange(f'{time_since}-')

    if 'timerange' in config:
        timerange = timerange.parse_timerange(config['timerange'])

    # config 파일에서 다운로드 종목 확인
    config['stake_currency'] = ''

    if 'pairs' not in config:
        raise OperationalException("종목을 추가해주세요")

    pairs_not_available: List[str] = []

    # config 에서 거래소 확인
    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)
    markets = [p for p, m in exchange.markets.items() if market_is_active(m)
               or config.get('include_inactive')]
    expanded_pairs = expand_pairlist(config['pairs'], markets)

    # config 에서 기본 정보 확인
    if not config['exchange'].get('skip_pair_validation', False):
        exchange.validate_pairs(expanded_pairs)


    for timeframe in config['timeframes']:
        exchange.validate_timeframes(timeframe)

    try:

        if config.get('download_trades'):
            pairs_not_available = refresh_backtest_trades_data(
                exchange, pairs=expanded_pairs, datadir=config['datadir'],
                timerange=timerange, new_pairs_days=config['new_pairs_days'],
                erase=bool(config.get('erase')), data_format=config['dataformat_trades'])

            # 데이터 시간 확인
            convert_trades_to_ohlcv(
                pairs=expanded_pairs, timeframes=config['timeframes'],
                datadir=config['datadir'], timerange=timerange, erase=bool(config.get('erase')),
                data_format_ohlcv=config['dataformat_ohlcv'],
                data_format_trades=config['dataformat_trades'],
            )
        else:
            pairs_not_available = refresh_backtest_ohlcv_data(
                exchange, pairs=expanded_pairs, timeframes=config['timeframes'],
                datadir=config['datadir'], timerange=timerange,
                new_pairs_days=config['new_pairs_days'],
                erase=bool(config.get('erase')), data_format=config['dataformat_ohlcv'])

    except KeyboardInterrupt:
        sys.exit("취소...")



def start_convert_trades(args: Dict[str, Any]) -> None:

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    timerange = TimeRange()

    # config 파일에서 필요없는 종목은 예외 처리
    config['stake_currency'] = ''

    if 'pairs' not in config:
        raise OperationalException(
            "Downloading data requires a list of pairs. "
            "Please check the documentation on how to configure this.")

    # 거래소
    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)
    # config 에서 기본 정보 확안
    if not config['exchange'].get('skip_pair_validation', False):
        exchange.validate_pairs(config['pairs'])
    expanded_pairs = expand_pairlist(config['pairs'], list(exchange.markets))



    for timeframe in config['timeframes']:
        exchange.validate_timeframes(timeframe)
    # 데이터 시간 변환
    convert_trades_to_ohlcv(
        pairs=expanded_pairs, timeframes=config['timeframes'],
        datadir=config['datadir'], timerange=timerange, erase=bool(config.get('erase')),
        data_format_ohlcv=config['dataformat_ohlcv'],
        data_format_trades=config['dataformat_trades'],
    )


def start_convert_data(args: Dict[str, Any], ohlcv: bool = True) -> None:
    """
    다운로드 받은 데이터 변환
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    if ohlcv:
        convert_ohlcv_format(config,
                             convert_from=args['format_from'], convert_to=args['format_to'],
                             erase=args['erase'])
    else:
        convert_trades_format(config,
                              convert_from=args['format_from'], convert_to=args['format_to'],
                              erase=args['erase'])

def start_list_data(args: Dict[str, Any]) -> None:
    """
    다운로드 완료된 데이터 확인
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from tabulate import tabulate
    from threee.data.history.idatahandler import get_datahandler

    dhc = get_datahandler(config['datadir'], config['dataformat_ohlcv'])
    paircombs = dhc.ohlcv_get_available_data(config['datadir'])

    if args['pairs']:
        paircombs = [comb for comb in paircombs if comb[0] in args['pairs']]
    print(f"Found {len(paircombs)} pair / timeframe combinations.")
    groupedpair = defaultdict(list)
    for pair, timeframe in sorted(paircombs, key=lambda x: (x[0], timeframe_to_minutes(x[1]))):
        groupedpair[pair].append(timeframe)

    if groupedpair:
        print(tabulate([(pair, ', '.join(timeframes)) for pair, timeframes in groupedpair.items()],
                       headers=("Pair", "Timeframe"),
                       tablefmt='psql', stralign='right'))
