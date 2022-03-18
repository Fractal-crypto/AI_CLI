from enum import Enum


class SignalType(Enum):
    """
    매수 매도 신호 구별
    """
    BUY = "buy"
    SELL = "sell"


class SignalTagType(Enum):
    """
    신호 열 구분
    """
    BUY_TAG = "buy_tag"
    EXIT_TAG = "exit_tag"
