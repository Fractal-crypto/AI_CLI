import logging
from copy import deepcopy
from typing import Any, Dict

from jsonschema import Draft4Validator, validators
from jsonschema.exceptions import ValidationError, best_match

from threee import constants
from threee.enums import RunMode
from threee.exceptions import OperationalException


def _extend_validator(validator_class):
    """
    json 스키마 이용해서  config.json 데이터 검증
    """
    validate_properties = validator_class.VALIDATORS['properties']

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if 'default' in subschema:
                instance.setdefault(prop, subschema['default'])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {'properties': set_defaults})

threee = _extend_validator(Draft4Validator)

def validate_config_schema(conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    지속적으로 실행전 config파일에서 정보 확인
    """
    conf_schema = deepcopy(constants.CONF_SCHEMA)
    if conf.get('runmode', RunMode.OTHER) in (RunMode.DRY_RUN, RunMode.LIVE):
        conf_schema['required'] = constants.SCHEMA_TRADE_REQUIRED
    elif conf.get('runmode', RunMode.OTHER) in (RunMode.BACKTEST, RunMode.HYPEROPT):
        conf_schema['required'] = constants.SCHEMA_BACKTEST_REQUIRED
    else:
        conf_schema['required'] = constants.SCHEMA_MINIMAL_REQUIRED
    try:
        threee(conf_schema).validate(conf)
        return conf
    except ValidationError as e:

        raise ValidationError('config 오류')

def validate_config_consistency(conf: Dict[str, Any]) -> None:
    """
    실행시킬때마다 각 함수들 정보 불러오
    """

    _validate_trailing_stoploss(conf)
    _validate_price_config(conf)
    _validate_edge(conf)
    _validate_whitelist(conf)
    _validate_protections(conf)
    _validate_unlimited_amount(conf)
    _validate_ask_orderbook(conf)
    validate_config_schema(conf)


def _validate_unlimited_amount(conf: Dict[str, Any]) -> None:
    """
    종목수랑 매매 거래량 재지정
    """
    if (not conf.get('edge', {}).get('enabled')
       and conf.get('max_open_trades') == float('inf')
       and conf.get('stake_amount') == constants.UNLIMITED_STAKE_AMOUNT):
        raise OperationalException('종목수 다시 지정')

def _validate_price_config(conf: Dict[str, Any]) -> None:
    """
    시장가 매수 할때 각 종목의 주문을 넣는 방향
    """
    if (conf.get('order_types', {}).get('buy') == 'market'
            and conf.get('bid_strategy', {}).get('price_side') != 'ask'):
        raise OperationalException('ask로 수정')

    if (conf.get('order_types', {}).get('sell') == 'market'
            and conf.get('ask_strategy', {}).get('price_side') != 'bid'):
        raise OperationalException('bid로 수')


def _validate_trailing_stoploss(conf: Dict[str, Any]) -> None:

    if conf.get('stoploss') == 0.0:
        raise OperationalException("종목이 일정이상 올라가면 손실률 측정과 기본 설정값이 너무 높습니"
        )
    if not conf.get('trailing_stop', False):
        return

    tsl_positive = float(conf.get('trailing_stop_positive', 0))
    tsl_offset = float(conf.get('trailing_stop_positive_offset', 0))
    tsl_only_offset = conf.get('trailing_only_offset_is_reached', False)

    if tsl_only_offset:
        if tsl_positive == 0.0:
            raise OperationalException("0보다 높은 값을 필요")
    if tsl_positive > 0 and 0 < tsl_offset <= tsl_positive:
        raise OperationalException("config 파일과 실제 loss값이 음수입니다")
    if 'trailing_stop_positive' in conf and float(conf['trailing_stop_positive']) == 0.0:
        raise OperationalException('0보다 높은 값을 필요')


def _validate_edge(conf: Dict[str, Any]) -> None:
    """
    edge 기능 동시작동 불가
    """
    if not conf.get('edge', {}).get('enabled'):
        return
    if not conf.get('use_sell_signal', True):
        raise OperationalException("Sell 기능 필요")


def _validate_whitelist(conf: Dict[str, Any]) -> None:
    """
    config파일에서 종목 정보
    """
    if conf.get('runmode', RunMode.OTHER) in [RunMode.OTHER, RunMode.PLOT,
                                              RunMode.UTIL_NO_EXCHANGE, RunMode.UTIL_EXCHANGE]:
        return
    for pl in conf.get('pairlists', [{'method': 'StaticPairList'}]):
        if (pl.get('method') == 'StaticPairList'
                and not conf.get('exchange', {}).get('pair_whitelist')):
            raise OperationalException("종목을 추가해주세요")

def _validate_protections(conf: Dict[str, Any]) -> None:
    """
    데이터 정보 오류시 확인
    """

    for prot in conf.get('protections', []):
        if ('stop_duration' in prot and 'stop_duration_candles' in prot):
            raise OperationalException('켄들데이터 오류')

        if ('lookback_period' in prot and 'lookback_period_candles' in prot):
            raise OperationalException("켄들데이터 오류")


def _validate_ask_orderbook(conf: Dict[str, Any]) -> None:
    ask_strategy = conf.get('ask_strategy', {})
    ob_min = ask_strategy.get('order_book_min')
    ob_max = ask_strategy.get('order_book_max')
    if ob_min is not None and ob_max is not None and ask_strategy.get('use_order_book'):
        if ob_min != ob_max:
            raise OperationalException("지정가 주문 지원하지 않습니다"
            )
        else:
            None
