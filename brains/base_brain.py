from enum import Enum
from typing import List
from entities.actions import *
from entities.car_state import CarState
from entities.player_state import PlayerState


class BaseBrainType(Enum):
    BotBrain = 0
    UserBrain = 1

class BaseBrain(ABC):
    @abstractmethod
    def get_actions(self, player_state: PlayerState, car_state: CarState, rays_dists: List[float]) -> List[BaseAction]:
        pass

    @abstractmethod
    def get_brain_type(self) -> BaseBrainType:
        pass

