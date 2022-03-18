import logging
from typing import Any, Dict

from threee.enums import RunMode
from threee.exceptions import OperationalException
from threee.exchange import (available_exchanges, is_exchange_known_ccxt,
                                is_exchange_officially_supported, validate_exchange)


def check_exchange(config: Dict[str, Any], check_for_bad: bool = True) -> bool:
    """
    confgit 파일에서 거래소 정보 확인
    """

    if (config['runmode'] in [RunMode.PLOT, RunMode.UTIL_NO_EXCHANGE, RunMode.OTHER]
       and not config.get('exchange', {}).get('name')):
        return True

    exchange = config.get('exchange', {}).get('name').lower()
    

    return True
