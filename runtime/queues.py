import queue
from typing import Any, Optional

class PrefetchQueue:
    """
    Thread-safe producer-consumer queue for background layer prefetching.
    """
    def __init__(self, maxsize: int = 16):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        return self._queue.get(block=block, timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()

    def clear(self) -> None:
        with self._queue.mutex:
            self._queue.queue.clear()
