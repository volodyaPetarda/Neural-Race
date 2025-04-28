from typing import List

import pygame

from actions import BaseAction, AccelerateAction, RotateAction, NitroAction, BackAction
from brains.base_brain import BaseBrain, BaseBrainType


class UserBrain(BaseBrain):
    def get_brain_type(self) -> BaseBrainType:
        return BaseBrainType.UserBrain

    def get_actions(self) -> List[BaseAction]:
        keys = pygame.key.get_pressed()
        actions = []
        if keys[pygame.K_w]:
            actions.append(AccelerateAction())
        if keys[pygame.K_a]:
            actions.append(RotateAction("left"))
        if keys[pygame.K_d]:
            actions.append(RotateAction("right"))
        if keys[pygame.K_s]:
            actions.append(BackAction())
        if keys[pygame.K_LSHIFT]:
            actions.append(NitroAction())
        return actions