import threading
from typing import Any, Optional

class LayerFuture:
    """
    Represents an in-flight or completed asynchronous layer prefetch operation.
    """
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self._event = threading.Event()
        self._result: Any = None
        self._exception: Optional[Exception] = None

    def set_result(self, result: Any) -> None:
        self._result = result
        self._event.set()

    def set_exception(self, exc: Exception) -> None:
        self._exception = exc
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> Any:
        signaled = self._event.wait(timeout=timeout)
        if not signaled:
            raise TimeoutError(f"LayerFuture for layer {self.layer_id} timed out after {timeout} seconds.")
        if self._exception is not None:
            raise self._exception
        return self._result

    def is_done(self) -> bool:
        return self._event.is_set()
