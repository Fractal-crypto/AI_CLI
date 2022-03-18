import logging
from ipaddress import IPv4Address
from typing import Any, Dict

import rapidjson
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from threee.exceptions import OperationalException
from threee.rpc.api_server.uvicorn_threaded import UvicornServer
from threee.rpc.rpc import RPC, RPCException, RPCHandler

class FTJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return rapidjson.dumps(content).encode("utf-8")


class ApiServer(RPCHandler):

    __instance = None
    __initialized = False

    _rpc: RPC
    _bt = None
    _bt_data = None
    _bt_timerange = None
    _bt_last_config: Dict[str, Any] = {}
    _has_rpc: bool = False
    _bgtask_running: bool = False
    _config: Dict[str, Any] = {}
    _exchange = None

    def __new__(cls, *args, **kwargs):
        if ApiServer.__instance is None:
            ApiServer.__instance = object.__new__(cls)
            ApiServer.__initialized = False
        return ApiServer.__instance

    def __init__(self, config: Dict[str, Any], standalone: bool = False) -> None:
        ApiServer._config = config
        if self.__initialized and (standalone or self._standalone):
            return
        self._standalone: bool = standalone
        self._server = None
        ApiServer.__initialized = True

        api_config = self._config['api_server']

        self.app = FastAPI(title="threee API",
                           docs_url='/docs' if api_config.get('enable_openapi', False) else None,
                           redoc_url=None,
                           default_response_class=FTJSONResponse,
                           )
        self.configure_app(self.app, self._config)

        self.start_api()

    def add_rpc_handler(self, rpc: RPC):
        if not self._has_rpc:
            ApiServer._rpc = rpc
            ApiServer._has_rpc = True
        else:
            raise OperationalException('RPC Handler already attached.')

    def cleanup(self) -> None:
        ApiServer._has_rpc = False
        del ApiServer._rpc
        if self._server and not self._standalone:
            self._server.cleanup()

    @classmethod
    def shutdown(cls):
        cls.__initialized = False
        del cls.__instance
        cls.__instance = None
        cls._has_rpc = False
        cls._rpc = None

    def send_msg(self, msg: Dict[str, str]) -> None:
        pass

    def handle_rpc_exception(self, request, exc):
        return JSONResponse(
            status_code=502,
            content={'error': f"Error querying {request.url.path}: {exc.message}"}
        )

    def configure_app(self, app: FastAPI, config):
        from threee.rpc.api_server.api_auth import http_basic_or_jwt_token, router_login
        from threee.rpc.api_server.api_backtest import router as api_backtest
        from threee.rpc.api_server.api_v1 import router as api_v1
        from threee.rpc.api_server.api_v1 import router_public as api_v1_public
        from threee.rpc.api_server.web_ui import router_ui

        app.include_router(api_v1_public, prefix="/api/v1")

        app.include_router(api_v1, prefix="/api/v1",
                           dependencies=[Depends(http_basic_or_jwt_token)],
                           )
        app.include_router(api_backtest, prefix="/api/v1",
                           dependencies=[Depends(http_basic_or_jwt_token)],
                           )
        app.include_router(router_login, prefix="/api/v1", tags=["auth"])
        app.include_router(router_ui, prefix='')

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config['api_server'].get('CORS_origins', []),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.add_exception_handler(RPCException, self.handle_rpc_exception)

    def start_api(self):
        """
        Start API ... should be run in thread.
        """
        rest_ip = self._config['api_server']['listen_ip_address']
        rest_port = self._config['api_server']['listen_port']


        verbosity = self._config['api_server'].get('verbosity', 'error')

        uvconfig = uvicorn.Config(self.app,
                                  port=rest_port,
                                  host=rest_ip,
                                  use_colors=False,
                                  log_config=None,
                                  access_log=True if verbosity != 'error' else False,
                                  )
        try:
            self._server = UvicornServer(uvconfig)
            if self._standalone:
                self._server.run()
            else:
                self._server.run_in_thread()
        except Exception:
            pass
