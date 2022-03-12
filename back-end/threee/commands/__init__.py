# flake8: noqa: F401
"""
Commands module.
Contains all start-commands, subcommands and CLI Interface creation.

Note: Be careful with file-scoped imports in these subfiles.
    as they are parsed on startup, nothing containing optional modules should be loaded.
"""
from threee.commands.arguments import Arguments
# from threee.commands.build_config_commands import start_new_config
from threee.commands.data_commands import (start_convert_data, start_convert_trades,
                                              start_download_data, start_list_data)
from threee.commands.deploy_commands import (start_create_userdir, start_install_ui,
                                                start_new_strategy)
from threee.commands.hyperopt_commands import start_hyperopt_list, start_hyperopt_show
from threee.commands.list_commands import (start_list_exchanges, start_list_markets,
                                              start_list_strategies, start_list_timeframes,
                                              start_show_trades)
from threee.commands.optimize_commands import (start_backtesting, start_backtesting_show,
                                                  start_edge, start_hyperopt)
from threee.commands.pairlist_commands import start_test_pairlist
# from threee.commands.plot_commands import start_plot_dataframe, start_plot_profit
from threee.commands.trade_commands import start_trading
from threee.commands.webserver_commands import start_webserver
