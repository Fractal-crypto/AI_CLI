import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


def start_trading(args: Dict[str, Any]) -> int:
    """
    트레이딩 명령
    """

    from threee.worker import Worker #워커실행
    worker = None
    try:
        worker = Worker(args)
        worker.run()
    except Exception as e:
        logger.error(str(e))
        logger.exception("Fatal exception!")
    except KeyboardInterrupt:
        logger.info('SIGINT received, aborting ...')
    finally:
        if worker:
            logger.info("worker found ... calling exit")
            worker.exit()
    return 0
