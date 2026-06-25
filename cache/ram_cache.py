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
        if self.config and "memory" in self.config and "max_ram_percent" in self.config["memory"]:
            ram_percent = self.config["memory"]["max_ram_percent"]

        return int(
            available * (ram_percent / 100.0)
        )

    def get(self, key):

        if key not in self.cache:

            return None

        value, size = self.cache.pop(key)

        self.cache[key] = (value, size)

        return value

    def put(
        self,
        key,
        value
    ):

        if isinstance(value, (list, tuple)):
            size = sum(t.numel() * t.element_size() for t in value if t is not None)
        else:
            size = value.numel() * value.element_size()

        # Scale down tracked size by 3 to prevent cache thrashing
        size = size // 3

        while (
            self.current
            +
            size
            >
            self.max_bytes
        ):

            _, (
                old,
                old_size
            ) = (
                self.cache
                .popitem(
                    last=False
                )
            )

            if isinstance(old, (list, tuple)):
                for t in old:
                    if t is not None:
                        del t
            del old

            self.current -= old_size

        self.cache[key] = (
            value,
            size
        )

        self.current += size
