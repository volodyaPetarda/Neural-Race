from typing import List

from car import Car
from track import Track


class RenderEngineContext:
    def __init__(self, cars: List[Car], track: Track, cars_reward_ind: dict[Car, int], draw_rewards: dict[Car, bool]):
        self.cars = cars
        self.track = track
        self.cars_reward_ind = cars_reward_ind
        self.draw_rewards = draw_rewards

