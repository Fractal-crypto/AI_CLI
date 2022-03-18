from enum import Enum


class State(Enum):
    """
    트레이딩 상태 정보
    """
    RUNNING = 1
    STOPPED = 2
    RELOAD_CONFIG = 3

    def __str__(self):
        return f"{self.name.lower()}"
