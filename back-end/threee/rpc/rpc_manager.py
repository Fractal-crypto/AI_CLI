"""
This module contains class to manage RPC communications (Telegram, API, ...)
"""
import logging
from typing import Any, Dict, List

from threee.enums import RPCMessageType
from threee.rpc import RPC, RPCHandler


class RPCManager:
    def __init__(self, freqtrade) -> None:
        self.registered_modules: List[RPCHandler] = []
        self._rpc = RPC(freqtrade)
        config = freqtrade.config
        if config.get('telegram', {}).get('enabled', False):
            from threee.rpc.telegram import Telegram
            self.registered_modules.append(Telegram(self._rpc, config))

        if config.get('webhook', {}).get('enabled', False):
            from threee.rpc.webhook import Webhook
            self.registered_modules.append(Webhook(self._rpc, config))


        if config.get('api_server', {}).get('enabled', False):

            from threee.rpc.api_server import ApiServer
            apiserver = ApiServer(config)
            apiserver.add_rpc_handler(self._rpc)
            self.registered_modules.append(apiserver)

    def cleanup(self) -> None:


        while self.registered_modules:
            mod = self.registered_modules.pop()

            mod.cleanup()
            del mod

    def send_msg(self, msg: Dict[str, Any]) -> None:
        if 'pair' in msg:
            msg.update({
                'base_currency': self._rpc._threee.exchange.get_pair_base_currency(msg['pair'])
                })
        for mod in self.registered_modules:
            try:
                mod.send_msg(msg)
            except NotImplementedError:
                pass

    def startup_messages(self, config: Dict[str, Any], pairlist, protections) -> None:
        if config['dry_run']:
            self.send_msg({
                'type': RPCMessageType.WARNING,
                'status': 'Dry run is enabled. All trades are simulated.'
            })
        stake_currency = config['stake_currency']
        stake_amount = config['stake_amount']
        minimal_roi = config['minimal_roi']
        stoploss = config['stoploss']
        trailing_stop = config['trailing_stop']
        timeframe = config['timeframe']
        exchange_name = config['exchange']['name']
        strategy_name = config.get('strategy', '')
        pos_adjust_enabled = 'On' if config['position_adjustment_enable'] else 'Off'
        self.send_msg({
            'type': RPCMessageType.STARTUP,
            'status': f'*Exchange:* `{exchange_name}`\n'
                      f'*Stake per trade:* `{stake_amount} {stake_currency}`\n'
                      f'*Minimum ROI:* `{minimal_roi}`\n'
                      f'*{"Trailing " if trailing_stop else ""}Stoploss:* `{stoploss}`\n'
                      f'*Position adjustment:* `{pos_adjust_enabled}`\n'
                      f'*Timeframe:* `{timeframe}`\n'
                      f'*Strategy:* `{strategy_name}`'
        })
        self.send_msg({
            'type': RPCMessageType.STARTUP,
            'status': f'Searching for {stake_currency} pairs to buy and sell '
                      f'based on {pairlist.short_desc()}'
        })
        if len(protections.name_list) > 0:
            prots = '\n'.join([p for prot in protections.short_desc() for k, p in prot.items()])
            self.send_msg({
                'type': RPCMessageType.STARTUP,
                'status': f'Using Protections: \n{prots}'
            })
