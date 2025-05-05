from typing import List

from entities.actions import BaseAction
from brains.base_brain import BaseBrain
from entities.car_state import CarState
from entities.figures import Vector2D, Rectangle
from entities.player_state import PlayerState


class Car:
    def __init__(self, brain: BaseBrain, position: Vector2D, velocity: Vector2D, angle: float, width=10, length=30):
        self.brain = brain
        self._position = position
        self._velocity = velocity
        self._angle = angle
        self.width = width
        self.length = length

    def get_actions(self, player_state: PlayerState, car_state: CarState, rays_dists: List[float]) -> List[BaseAction]:
        return self.brain.get_actions(player_state, car_state, rays_dists)

    @property
    def position(self) -> Vector2D:
        return self._position

    @position.setter
    def position(self, value: Vector2D):
        self._position = value

    @property
    def velocity(self) -> Vector2D:
        return self._velocity

    @velocity.setter
    def velocity(self, value: Vector2D):
        self._velocity = value

    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, angle: float):
        self._angle = angle

    @property
    def collider(self) -> Rectangle:
        return Rectangle(
            self.position,
            self.length,
            self.width,
            self.angle,
        )