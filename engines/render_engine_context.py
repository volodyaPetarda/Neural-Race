from typing import List

import numpy as np
import pygame

from entities.car import Car
from entities.track import Track


class RenderEngineContext:
    def __init__(self, cars: List[Car], track: Track, precomputed_numpy_walls: np.array, cars_reward_ind: dict[Car, int], draw_rewards: dict[Car, bool], rays_count: dict[Car, int], draw_rays: dict[Car, bool], car_person_images: dict[Car, pygame.surface]):
        self.cars = cars
        self.track = track
        self.precomputed_numpy_walls = precomputed_numpy_walls
        self.cars_reward_ind = cars_reward_ind
        self.draw_rewards = draw_rewards
        self.rays_count = rays_count
        self.draw_rays = draw_rays
        self.car_person_images = car_person_images
