from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from actions import *

class BaseBrainType(Enum):
    BotBrain = 0
    UserBrain = 1

class BaseBrain(ABC):
    @abstractmethod
    def get_actions(self) -> List[BaseAction]:
        pass

    @abstractmethod
    def get_brain_type(self) -> BaseBrainType:
        pass

