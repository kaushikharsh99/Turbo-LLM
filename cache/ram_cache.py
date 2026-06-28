from collections import OrderedDict

import psutil


class RAMCache:

    def __init__(self, config=None):

        self.cache = OrderedDict()
        self.current = 0

        self.config = config
        self.max_bytes = self.compute_limit()

    def compute_limit(self):

        available = psutil.virtual_memory().available

        ram_percent = 35

        if (
            self.config
            and "memory" in self.config
            and "max_ram_percent" in self.config["memory"]
        ):
            ram_percent = self.config["memory"]["max_ram_percent"]

        return int(available * (ram_percent / 100.0))

    def get(self, key):

        if key not in self.cache:
            return None

        value, size = self.cache.pop(key)

        self.cache[key] = (value, size)

        return value

    def put(self, key, value):

        if isinstance(value, (list, tuple)):
            size = sum(
                t.numel() * t.element_size()
                for t in value
                if t is not None
            )
        else:
            size = value.numel() * value.element_size()

        # Don't cache tensors larger than the cache itself.
        if size > self.max_bytes:
            return

        # Update existing entry.
        if key in self.cache:
            _, old_size = self.cache.pop(key)
            self.current -= old_size

        # Evict LRU entries until there is enough room.
        while self.current + size > self.max_bytes and self.cache:

            _, (old, old_size) = self.cache.popitem(last=False)

            if isinstance(old, (list, tuple)):
                for tensor in old:
                    del tensor
            else:
                del old

            self.current -= old_size

        self.cache[key] = (value, size)
        self.current += size

    def clear(self):

        self.cache.clear()
        self.current = 0

    def stats(self):

        return {
            "entries": len(self.cache),
            "current_bytes": self.current,
            "max_bytes": self.max_bytes,
            "usage_percent": (
                100 * self.current / self.max_bytes
                if self.max_bytes
                else 0
            ),
        }

    def __contains__(self, key):

        return key in self.cache

    def __len__(self):

        return len(self.cache)