from typing import List

from figures import Vector2D, Segment


class Track:
    def __init__(self, spawn_point: Vector2D, spawn_rotation: float, walls: List[Segment], rewards: List[Segment]):
        self.spawn_point = spawn_point
        self.spawn_rotation = spawn_rotation
        self.walls = walls
        self.rewards = rewards

    def get_reward_segment(self, ind) -> Segment:
        return self.rewards[ind % len(self.rewards)]


