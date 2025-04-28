from typing import List

from car import Car
from track import Track


class PhysicsEngineContext:
    def __init__(self, cars: List[Car], track: Track, cars_reward_ind: dict[Car, int], rays_count: dict[Car, int], delta_time: float):
        self.cars = cars
        self.track = track
        self.cars_reward_ind = cars_reward_ind
        self.rays_count = rays_count
        self.delta_time = delta_time