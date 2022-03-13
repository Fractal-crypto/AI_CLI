"""
arguments.py 에서 사용하는 실질적 cli 커멘드
"""
from argparse import SUPPRESS, ArgumentTypeError

from threee import __version__, constants
from threee.constants import HYPEROPT_LOSS_BUILTIN


def check_int_positive(value: str) -> int:
    try:
        uint = int(value)
        if uint <= 0:
            raise ValueError
    except ValueError:
        raise ArgumentTypeError(
            f"{value} 양수로 변환"
        )
    return uint


def check_int_nonzero(value: str) -> int:
    try:
        uint = int(value)
        if uint == 0:
            raise ValueError
    except ValueError:
        raise ArgumentTypeError(
            f"{value} 0 이상으로 변환"
        )
    return uint


class Arg:
    """
    명령어로 들어갈 커멘드 클래스  ex) --days --timeframe
    """
    def __init__(self, *args, **kwargs):
        self.cli = args
        self.kwargs = kwargs

AVAILABLE_CLI_OPTIONS = {

    "config": Arg(
        '-c', '--config',
        help='기본 config 정보',
        action='append',
        metavar='PATH',
    ),

    "strategy": Arg(
        '-s', '--strategy',
        help='트레이딩 실행',
        metavar='NAME',
    ),

    "timeframe": Arg(
        '-i', '--timeframe', '--ticker-interval',
        help='데이터 시간 5m 15m 4h',
    ),
    "timerange": Arg(
        '--timerange',
        help='데이터 학습 지기간',
    ),

    "hyperopt": Arg(
        '--hyperopt',
        help='베이지안 최적화',
        metavar='NAME',
        required=False,
    ),

    "epochs": Arg(
        '-e', '--epochs',
        help='epoch수 지정',
        type=check_int_positive,
        metavar='INT',
        default=constants.HYPEROPT_EPOCH,
    ),
    "spaces": Arg(
        '--spaces',
        help='파라미터 구체적 지정',
        choices=['all', 'buy', 'sell', 'roi', 'stoploss', 'trailing', 'protection', 'default'],
        nargs='+',
        default='default',
    ),
    "print_all": Arg(
        '--print-all',
        help='전체 트레이닝 과정 출력',
        action='store_true',
        default=False,
    ),
    "print_colorized": Arg(
        '--no-color',
        help='무색'
        'redirecting output to a file.',
        action='store_false',
        default=True,
    ),

    "hyperopt_min_trades": Arg(
        '--min-trades',
        help="최소거래량 지정"
        "optimization path (default: 1).",
        type=check_int_positive,
        metavar='INT',
        default=1,
    ),
    "hyperopt_loss": Arg(
        '--hyperopt-loss', '--hyperoptloss',
        help='loss function 적용  '
        ,
        metavar='NAME',
    ),

    "pairs": Arg(
        '-p', '--pairs',
        help='필요한 종목',
        nargs='+',
    ),

    "days": Arg(
        '--days',
        help='데이터 다운로드',
        type=check_int_positive,
        metavar='INT',
    ),

    "exchange": Arg(
        '--exchange',
        help='거래소 선택',
    ),
    "timeframes": Arg(
        '-t', '--timeframes',
        help='데이터 시간'
        'Default: `1m 5m`.',
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
                 '6h', '8h', '12h', '1d', '3d', '1w', '2w', '1M', '1y'],
        default=['1m', '5m'],
        nargs='+',
    ),

    "hyperopt_list_profitable": Arg(
        '--profitable',
        help='Select only profitable epochs.',
        action='store_true',
    ),

}
