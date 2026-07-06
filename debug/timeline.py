from typing import List
from debug.events import ProfileEvent

class Timeline:
    def __init__(self):
        self.events: List[ProfileEvent] = []

    def record(self, event: ProfileEvent):
        self.events.append(event)

    def clear(self):
        self.events.clear()
