import contextlib
import threading
import time

import uvicorn


def asyncio_setup() -> None:
    import sys

    if sys.version_info >= (3, 8) and sys.platform == "win32":
        import asyncio
        import selectors
        selector = selectors.SelectSelector()
        loop = asyncio.SelectorEventLoop(selector)
        asyncio.set_event_loop(loop)


class UvicornServer(uvicorn.Server):

    def run(self, sockets=None):
        import asyncio

        try:
            import uvloop
        except ImportError:

            asyncio_setup()
        else:
            asyncio.set_event_loop(uvloop.new_event_loop())
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        loop.run_until_complete(self.serve(sockets=sockets))

    @contextlib.contextmanager
    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        while not self.started:
            time.sleep(1e-3)

    def cleanup(self):
        self.should_exit = True
        self.thread.join()
