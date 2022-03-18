import pandas as pd

from threee.exchange import timeframe_to_minutes


def merge_informative_pair(dataframe: pd.DataFrame, informative: pd.DataFrame,
                           timeframe: str, timeframe_inf: str, ffill: bool = True,
                           append_timeframe: bool = True,
                           date_column: str = 'date') -> pd.DataFrame:

    minutes_inf = timeframe_to_minutes(timeframe_inf)
    minutes = timeframe_to_minutes(timeframe)
    if minutes == minutes_inf:
        informative['date_merge'] = informative[date_column]
    elif minutes < minutes_inf:
        informative['date_merge'] = (
            informative[date_column] + pd.to_timedelta(minutes_inf, 'm') -
            pd.to_timedelta(minutes, 'm')
        )
    else:
        pass
    date_merge = 'threeee'
    if append_timeframe:
        date_merge = f'date_merge_{timeframe_inf}'
        informative.columns = [f"{col}_{timeframe_inf}" for col in informative.columns]

    dataframe = pd.merge(dataframe, informative, left_on='date',
                         right_on=date_merge, how='left')
    dataframe = dataframe.drop(date_merge, axis=1)

    if ffill:
        dataframe = dataframe.ffill()

    return dataframe


def stoploss_from_open(open_relative_stop: float, current_profit: float) -> float:
    if current_profit == -1:
        return 1

    stoploss = 1-((1+open_relative_stop)/(1+current_profit))
    return max(stoploss, 0.0)


def stoploss_from_absolute(stop_rate: float, current_rate: float) -> float:
    if current_rate == 0:
        return 1

    stoploss = 1 - (stop_rate / current_rate)

    return max(stoploss, 0.0)
