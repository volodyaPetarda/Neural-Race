from typing import List, Tuple

import numpy as np

from entities.car import Car
from entities.track import Track


class PhysicsEngineContext:
    def __init__(self, step: int, cars: List[Car], track: Track, numpy_walls: np.array, precomputed_numpy_walls: np.array, cars_reward_ind: dict[Car, int], rays_count: dict[Car, int], cars_deaths: dict[Car, int], last_reward_collected: dict[Car, float], time_elapsed: float, delta_time: float, think_every: dict[Car, Tuple[int, int]], prev_action: dict[Car, list]):
        self.step = step
        self.cars = cars
        self.track = track
        self.numpy_walls = numpy_walls
        self.precomputed_numpy_walls = precomputed_numpy_walls
        self.cars_reward_ind = cars_reward_ind
        self.rays_count = rays_count
        self.cars_deaths = cars_deaths
        self.last_reward_collected = last_reward_collected
        self.time_elapsed = time_elapsed
        self.delta_time = delta_time
        self.think_every = think_every
        self.prev_action = prev_action
