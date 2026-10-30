from __future__ import annotations

import logging
import logging.handlers
import queue
from typing import Optional


class _QueueListener:
    def __init__(self, log_queue: queue.Queue):
        self._queue = log_queue
        self._listener: Optional[logging.handlers.QueueListener] = None

    def start(self) -> None:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self._listener = logging.handlers.QueueListener(self._queue, handler)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()


def setup_queue_listener(log_queue: queue.Queue) -> _QueueListener:
    return _QueueListener(log_queue)


def setup_logger(name: str, level: int, log_queue: queue.Queue) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    qh = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(qh)
    return logger
