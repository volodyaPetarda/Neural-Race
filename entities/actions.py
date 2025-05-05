from dataclasses import dataclass
from abc import ABC, abstractmethod

class BaseAction(ABC):
    pass

@dataclass
class AccelerateAction(BaseAction):
    pass

@dataclass
class RotateAction(BaseAction):
    def __init__(self, direction: str):
        if direction not in ["left", "right"]:
            raise ValueError(f"Direction {direction} is not valid")
        self.direction = direction

class NitroAction(BaseAction):
    def __init__(self):
        pass

class BackAction(BaseAction):
    pass