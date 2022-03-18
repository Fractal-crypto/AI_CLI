from typing import Any, Callable, NamedTuple, Optional, Union

from pandas import DataFrame

from threee.exceptions import OperationalException
from threee.strategy.strategy_helper import merge_informative_pair


PopulateIndicators = Callable[[Any, DataFrame, dict], DataFrame]


class InformativeData(NamedTuple):
    asset: Optional[str]
    timeframe: str
    fmt: Union[str, Callable[[Any], str], None]
    ffill: bool


def informative(timeframe: str, asset: str = '',
                fmt: Optional[Union[str, Callable[[Any], str]]] = None,
                ffill: bool = True) -> Callable[[PopulateIndicators], PopulateIndicators]:

    _asset = asset
    _timeframe = timeframe
    _fmt = fmt
    _ffill = ffill

    def decorator(fn: PopulateIndicators):
        informative_pairs = getattr(fn, '_ft_informative', [])
        informative_pairs.append(InformativeData(_asset, _timeframe, _fmt, _ffill))
        setattr(fn, '_ft_informative', informative_pairs)
        return fn
    return decorator


def _format_pair_name(config, pair: str) -> str:
    return pair.format(stake_currency=config['stake_currency'],
                       stake=config['stake_currency']).upper()


def _create_and_merge_informative_pair(strategy, dataframe: DataFrame, metadata: dict,
                                       inf_data: InformativeData,
                                       populate_indicators: PopulateIndicators):
    asset = inf_data.asset or ''
    timeframe = inf_data.timeframe
    fmt = inf_data.fmt
    config = strategy.config

    if asset:
        asset = _format_pair_name(config, asset)
    else:
        asset = metadata['pair']

    market = strategy.dp.market(asset)
    if market is None:
        raise OperationalException(f'Market {asset} is not available.')
    base = market['base']
    quote = market['quote']

    if not fmt:
        fmt = '{column}_{timeframe}'
        if inf_data.asset:
            fmt = '{base}_{quote}_' + fmt

    inf_metadata = {'pair': asset, 'timeframe': timeframe}
    inf_dataframe = strategy.dp.get_pair_dataframe(asset, timeframe)
    inf_dataframe = populate_indicators(strategy, inf_dataframe, inf_metadata)

    formatter: Any = None
    if callable(fmt):
        formatter = fmt
    else:
        formatter = fmt.format

    fmt_args = {
        'BASE': base.upper(),
        'QUOTE': quote.upper(),
        'base': base.lower(),
        'quote': quote.lower(),
        'asset': asset,
        'timeframe': timeframe,
    }
    inf_dataframe.rename(columns=lambda column: formatter(column=column, **fmt_args),
                         inplace=True)

    date_column = formatter(column='date', **fmt_args)
    if date_column in dataframe.columns:
        pass
    dataframe = merge_informative_pair(dataframe, inf_dataframe, strategy.timeframe, timeframe,
                                       ffill=inf_data.ffill, append_timeframe=False,
                                       date_column=date_column)
    return dataframe
