import asyncio
import logging
from copy import deepcopy

from fastapi import APIRouter, BackgroundTasks, Depends

from threee.configuration.config_validation import validate_config_consistency
from threee.enums import BacktestState
from threee.exceptions import DependencyException
from threee.rpc.api_server.api_schemas import BacktestRequest, BacktestResponse
from threee.rpc.api_server.deps import get_config, is_webserver_mode
from threee.rpc.api_server.webserver import ApiServer
from threee.rpc.rpc import RPCException


router = APIRouter()


@router.post('/backtest', response_model=BacktestResponse, tags=['webserver', 'backtest'])
async def api_start_backtest(bt_settings: BacktestRequest, background_tasks: BackgroundTasks,
                             config=Depends(get_config), ws_mode=Depends(is_webserver_mode)):


    btconfig = deepcopy(config)
    settings = dict(bt_settings)
    for setting in settings.keys():
        if settings[setting] is not None:
            btconfig[setting] = settings[setting]
    try:
        btconfig['stake_amount'] = float(btconfig['stake_amount'])
    except ValueError:
        pass

    btconfig['dry_run'] = True

    def run_backtest():
        from threee.optimize.optimize_reports import (generate_backtest_stats,
                                                         store_backtest_stats)
        from threee.resolvers import StrategyResolver
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:

            lastconfig = ApiServer._bt_last_config
            strat = StrategyResolver.load_strategy(btconfig)
            validate_config_consistency(btconfig)

            if (
                not ApiServer._bt
                or lastconfig.get('timeframe') != strat.timeframe
                or lastconfig.get('timeframe_detail') != btconfig.get('timeframe_detail')
                or lastconfig.get('timerange') != btconfig['timerange']
            ):
                from threee.optimize.backtesting import Backtesting
                ApiServer._bt = Backtesting(btconfig)
                ApiServer._bt.load_bt_data_detail()
            else:
                ApiServer._bt.config = btconfig
                ApiServer._bt.init_backtest()
            if (
                not ApiServer._bt_data
                or not ApiServer._bt_timerange
                or lastconfig.get('timeframe') != strat.timeframe
                or lastconfig.get('timerange') != btconfig['timerange']
            ):
                ApiServer._bt_data, ApiServer._bt_timerange = ApiServer._bt.load_bt_data()

            lastconfig['timerange'] = btconfig['timerange']
            lastconfig['timeframe'] = strat.timeframe
            lastconfig['protections'] = btconfig.get('protections', [])
            lastconfig['enable_protections'] = btconfig.get('enable_protections')
            lastconfig['dry_run_wallet'] = btconfig.get('dry_run_wallet')

            ApiServer._bt.results = {}
            ApiServer._bt.load_prior_backtest()

            ApiServer._bt.abort = False
            if (ApiServer._bt.results and
                    strat.get_strategy_name() in ApiServer._bt.results['strategy']):
                pass
            else:
                min_date, max_date = ApiServer._bt.backtest_one_strategy(
                    strat, ApiServer._bt_data, ApiServer._bt_timerange)

                ApiServer._bt.results = generate_backtest_stats(
                    ApiServer._bt_data, ApiServer._bt.all_results,
                    min_date=min_date, max_date=max_date)

            if btconfig.get('export', 'none') == 'trades':
                store_backtest_stats(btconfig['exportfilename'], ApiServer._bt.results)

        except DependencyException as e:
            pass
        finally:
            ApiServer._bgtask_running = False

    background_tasks.add_task(run_backtest)
    ApiServer._bgtask_running = True

    return {
        "status": "running",
        "running": True,
        "progress": 0,
        "step": str(BacktestState.STARTUP),
        "status_msg": "Backtest started",
    }


@router.get('/backtest', response_model=BacktestResponse, tags=['webserver', 'backtest'])
def api_get_backtest(ws_mode=Depends(is_webserver_mode)):
    from threee.persistence import LocalTrade
    if ApiServer._bgtask_running:
        return {
            "status": "running",
            "running": True,
            "step": ApiServer._bt.progress.action if ApiServer._bt else str(BacktestState.STARTUP),
            "progress": ApiServer._bt.progress.progress if ApiServer._bt else 0,
            "trade_count": len(LocalTrade.trades),
            "status_msg": "Backtest running",
        }

    if not ApiServer._bt:
        return {
            "status": "not_started",
            "running": False,
            "step": "",
            "progress": 0,
            "status_msg": "Backtest not yet executed"
        }

    return {
        "status": "ended",
        "running": False,
        "status_msg": "Backtest ended",
        "step": "finished",
        "progress": 1,
        "backtest_result": ApiServer._bt.results,
    }


@router.delete('/backtest', response_model=BacktestResponse, tags=['webserver', 'backtest'])
def api_delete_backtest(ws_mode=Depends(is_webserver_mode)):
    """Reset backtesting"""
    if ApiServer._bgtask_running:
        return {
            "status": "running",
            "running": True,
            "step": "",
            "progress": 0,
            "status_msg": "Backtest running",
        }
    if ApiServer._bt:
        del ApiServer._bt
        ApiServer._bt = None
        del ApiServer._bt_data
        ApiServer._bt_data = None
    
    return {
        "status": "reset",
        "running": False,
        "step": "",
        "progress": 0,
        "status_msg": "Backtest reset",
    }


@router.get('/backtest/abort', response_model=BacktestResponse, tags=['webserver', 'backtest'])
def api_backtest_abort(ws_mode=Depends(is_webserver_mode)):
    if not ApiServer._bgtask_running:
        return {
            "status": "not_running",
            "running": False,
            "step": "",
            "progress": 0,
            "status_msg": "Backtest ended",
        }
    ApiServer._bt.abort = True
    return {
        "status": "stopping",
        "running": False,
        "step": "",
        "progress": 0,
        "status_msg": "Backtest ended",
    }
