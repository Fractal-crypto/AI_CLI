import math
from datetime import datetime

from pandas import DataFrame, date_range

from threee.optimize.hyperopt import IHyperOptLoss


class SharpeHyperOptLossDaily(IHyperOptLoss):


    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        resample_freq = '1D'
        slippage_per_trade_ratio = 0.0005
        days_in_year = 365
        annual_risk_free_rate = 0.0
        risk_free_rate = annual_risk_free_rate / days_in_year

        results.loc[:, 'profit_ratio_after_slippage'] = \
            results['profit_ratio'] - slippage_per_trade_ratio


        t_index = date_range(start=min_date, end=max_date, freq=resample_freq,
                             normalize=True)

        sum_daily = (
            results.resample(resample_freq, on='close_date').agg(
                {"profit_ratio_after_slippage": sum}).reindex(t_index).fillna(0)
        )

        total_profit = sum_daily["profit_ratio_after_slippage"] - risk_free_rate
        expected_returns_mean = total_profit.mean()
        up_stdev = total_profit.std()

        if up_stdev != 0:
            sharp_ratio = expected_returns_mean / up_stdev * math.sqrt(days_in_year)
        else:
            sharp_ratio = -20.

        return -sharp_ratio
