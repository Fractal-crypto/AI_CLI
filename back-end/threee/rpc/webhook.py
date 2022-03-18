
import logging
import time
from typing import Any, Dict

from requests import RequestException, post

from threee.enums import RPCMessageType
from threee.rpc import RPC, RPCHandler


class Webhook(RPCHandler):

    def __init__(self, rpc: RPC, config: Dict[str, Any]) -> None:
        super().__init__(rpc, config)

        self._url = self._config['webhook']['url']
        self._format = self._config['webhook'].get('format', 'form')
        self._retries = self._config['webhook'].get('retries', 0)
        self._retry_delay = self._config['webhook'].get('retry_delay', 0.1)

    def cleanup(self) -> None:
        pass

    def send_msg(self, msg: Dict[str, Any]) -> None:
        try:

            if msg['type'] == RPCMessageType.BUY:
                valuedict = self._config['webhook'].get('webhookbuy', None)
            elif msg['type'] == RPCMessageType.BUY_CANCEL:
                valuedict = self._config['webhook'].get('webhookbuycancel', None)
            elif msg['type'] == RPCMessageType.BUY_FILL:
                valuedict = self._config['webhook'].get('webhookbuyfill', None)
            elif msg['type'] == RPCMessageType.SELL:
                valuedict = self._config['webhook'].get('webhooksell', None)
            elif msg['type'] == RPCMessageType.SELL_FILL:
                valuedict = self._config['webhook'].get('webhooksellfill', None)
            elif msg['type'] == RPCMessageType.SELL_CANCEL:
                valuedict = self._config['webhook'].get('webhooksellcancel', None)
            elif msg['type'] in (RPCMessageType.STATUS,
                                 RPCMessageType.STARTUP,
                                 RPCMessageType.WARNING):
                valuedict = self._config['webhook'].get('webhookstatus', None)
            else:
                raise NotImplementedError('Unknown message type: {}'.format(msg['type']))
            if not valuedict:

                return

            payload = {key: value.format(**msg) for (key, value) in valuedict.items()}
            self._send_msg(payload)
        except KeyError as exc:
            pass

    def _send_msg(self, payload: dict) -> None:

        success = False
        attempts = 0
        while not success and attempts <= self._retries:
            if attempts:
                if self._retry_delay:
                    time.sleep(self._retry_delay)

            attempts += 1

            try:
                if self._format == 'form':
                    response = post(self._url, data=payload)
                elif self._format == 'json':
                    response = post(self._url, json=payload)
                elif self._format == 'raw':
                    response = post(self._url, data=payload['data'],
                                    headers={'Content-Type': 'text/plain'})
                else:
                    raise NotImplementedError('Unknown format: {}'.format(self._format))

                response.raise_for_status()
                success = True

            except RequestException as exc:
                pass
