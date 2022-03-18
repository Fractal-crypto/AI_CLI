import logging
import time
import traceback
from os import getpid
from typing import Any, Callable, Dict, Optional

import sdnotify

from threee import __version__, constants
from threee.configuration import Configuration
from threee.enums import State
from threee.exceptions import OperationalException, TemporaryError
from threee.threeebot import threeeBot


class Worker:

    def __init__(self, args: Dict[str, Any], config: Dict[str, Any] = None) -> None:
        """
        워커를 통해 거래 시작
        """


        self._args = args
        self._config = config
        self._init(False)

        self.last_throttle_start_time: float = 0
        self._heartbeat_msg: float = 0

        # Tell systemd that we completed initialization phase
        self._notify("READY=1")

    def _init(self, reconfig: bool) -> None:
        """
        config에서도 사용가능
        """
        if reconfig or self._config is None:

            self._config = Configuration(self._args, None).get_config()
        self.threee = threeeBot(self._config)
        internals_config = self._config.get('internals', {})
        self._throttle_secs = internals_config.get('process_throttle_secs',
                                                   constants.PROCESS_THROTTLE_SECS)
        self._heartbeat_interval = internals_config.get('heartbeat_interval', 60)

        self._sd_notify = sdnotify.SystemdNotifier() if \
            self._config.get('internals', {}).get('sd_notify', False) else None

    def _notify(self, message: str) -> None:
        """
        사용가능하면 리턴
        """
        if self._sd_notify:
            self._sd_notify.notify(message)

    def run(self) -> None:
        state = None
        while True:
            state = self._worker(old_state=state)
            if state == State.RELOAD_CONFIG:
                self._reconfigure()

    def _worker(self, old_state: Optional[State]) -> State:
        """
        각 쓰로틀이 각자 워커 사용
        """
        state = self.threee.state

        if state != old_state:

            if old_state != State.RELOAD_CONFIG:
                self.threee.notify_status(f'{state.name.lower()}')

            if state == State.RUNNING:
                self.threee.startup()

            if state == State.STOPPED:
                self.threee.check_for_open_trades()

            self._heartbeat_msg = 0

        if state == State.STOPPED:
            # 프로그램 멈출때 핑
            self._notify("멈춤")
            self._throttle(func=self._process_stopped, throttle_secs=self._throttle_secs)

        elif state == State.RUNNING:
            # 시작전에 러닝 표시
            self._notify("시작")

            self._throttle(func=self._process_running, throttle_secs=self._throttle_secs)

        if self._heartbeat_interval:
            now = time.time()
            if (now - self._heartbeat_msg) > self._heartbeat_interval:
                version = __version__
                strategy_version = self.threee.strategy.version()
                if (strategy_version is not None):
                    version += ', strategy_version: ' + strategy_version
                self._heartbeat_msg = now

        return state

    def _throttle(self, func: Callable[..., Any], throttle_secs: float, *args, **kwargs) -> Any:
        """
        쉘위에서 각 실행 결과 표시
        """
        self.last_throttle_start_time = time.time()

        result = func(*args, **kwargs)
        time_passed = time.time() - self.last_throttle_start_time
        sleep_duration = max(throttle_secs - time_passed, 0.0)

        time.sleep(sleep_duration)
        return result

    def _process_stopped(self) -> None:
        self.threee.process_stopped()

    def _process_running(self) -> None:
        try:
            self.threee.process()
        except TemporaryError as error:
            None
        except OperationalException:
            tb = traceback.format_exc()
            hint = 'None'

            self.threee.notify_status()


            self.threee.state = State.STOPPED

    def _reconfigure(self) -> None:
        """
        새로운 config 리로드
        """

        self._notify("다시시작")

        self.threee.cleanup()

        self._init(True)

        self.threee.notify_status('재설정')

        self._notify("준비")

    def exit(self) -> None:

        self._notify("멈춤")

        if self.threee:
            self.threee.notify_status('프로세스 죵료')
            self.threee.cleanup()
