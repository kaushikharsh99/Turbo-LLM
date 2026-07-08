import torch


class DeviceManager:
    def __init__(self, device=None):
        if device is None or device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    @property
    def is_mps(self):
        return self.device.type == "mps"

    @property
    def is_cpu(self):
        return self.device.type == "cpu"

    def synchronize(self):
        if self.is_cuda:
            torch.cuda.synchronize()
        elif self.is_mps:
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

    def empty_cache(self):
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def max_memory_allocated(self):
        if self.is_cuda:
            return torch.cuda.max_memory_allocated()
        elif self.is_mps:
            if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                return torch.mps.current_allocated_memory()
        return 0

    def memory_allocated(self):
        if self.is_cuda:
            return torch.cuda.memory_allocated()
        elif self.is_mps:
            if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                return torch.mps.current_allocated_memory()
        return 0

    def memory_reserved(self):
        if self.is_cuda:
            return torch.cuda.memory_reserved()
        return 0

    def reset_peak_memory_stats(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def create_stream(self):
        if self.is_cuda:
            return torch.cuda.Stream()
        return None

    def create_event(self, **kwargs):
        if self.is_cuda:
            return torch.cuda.Event(**kwargs)
        return None

    def supports_async_copy(self):
        return self.is_cuda
