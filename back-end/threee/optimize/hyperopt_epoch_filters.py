from typing import List

from threee.exceptions import OperationalException

def hyperopt_filter_epochs(epochs: List, filteroptions: dict, log: bool = True) -> List:
    """
    hyperopt 결과 목록에서 항목 필터링
    """
    if filteroptions['only_best']:
        epochs = [x for x in epochs if x['is_best']]
    if filteroptions['only_profitable']:
        epochs = [x for x in epochs
                  if x['results_metrics'].get('profit_total', 0) > 0]

    epochs = _hyperopt_filter_epochs_trade_count(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_duration(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_profit(epochs, filteroptions)

    epochs = _hyperopt_filter_epochs_objective(epochs, filteroptions)

    return epochs


def _hyperopt_filter_epochs_trade(epochs: List, trade_count: int):
    return [
        x for x in epochs if x['results_metrics'].get('total_trades', 0) > trade_count
    ]


def _hyperopt_filter_epochs_trade_count(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_trades'] > 0:
        epochs = _hyperopt_filter_epochs_trade(epochs, filteroptions['filter_min_trades'])

    if filteroptions['filter_max_trades'] > 0:
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('total_trades') < filteroptions['filter_max_trades']
        ]
    return epochs


def _hyperopt_filter_epochs_duration(epochs: List, filteroptions: dict) -> List:

    def get_duration_value(x):
        # Duration in minutes ...
        if 'holding_avg_s' in x['results_metrics']:
            avg = x['results_metrics']['holding_avg_s']
            return avg // 60
        pass

    if filteroptions['filter_min_avg_time'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if get_duration_value(x) > filteroptions['filter_min_avg_time']
        ]
    if filteroptions['filter_max_avg_time'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if get_duration_value(x) < filteroptions['filter_max_avg_time']
        ]

    return epochs


def _hyperopt_filter_epochs_profit(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_avg_profit'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('profit_mean', 0) * 100
            > filteroptions['filter_min_avg_profit']
        ]
    if filteroptions['filter_max_avg_profit'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('profit_mean', 0) * 100
            < filteroptions['filter_max_avg_profit']
        ]
    if filteroptions['filter_min_total_profit'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('profit_total_abs', 0)
            > filteroptions['filter_min_total_profit']
        ]
    if filteroptions['filter_max_total_profit'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)
        epochs = [
            x for x in epochs
            if x['results_metrics'].get('profit_total_abs', 0)
            < filteroptions['filter_max_total_profit']
        ]
    return epochs


def _hyperopt_filter_epochs_objective(epochs: List, filteroptions: dict) -> List:

    if filteroptions['filter_min_objective'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)

        epochs = [x for x in epochs if x['loss'] < filteroptions['filter_min_objective']]
    if filteroptions['filter_max_objective'] is not None:
        epochs = _hyperopt_filter_epochs_trade(epochs, 0)

        epochs = [x for x in epochs if x['loss'] > filteroptions['filter_max_objective']]

    return epochs
