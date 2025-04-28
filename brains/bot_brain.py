from typing import List
from actions import BaseAction
from base_brain import BaseBrain
from brains.base_brain import BaseBrainType


class BotBrain(BaseBrain):
    def get_brain_type(self) -> BaseBrainType:
        return BaseBrainType.BotBrain

    def get_actions(self) -> List[BaseAction]:
        pass