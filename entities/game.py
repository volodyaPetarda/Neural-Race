from typing import List, Tuple

import numpy as np
import pygame

from engines.physics_engine import PhysicsEngine
from engines.physics_engine_context import PhysicsEngineContext
from engines.render_engine import RenderEngine
from entities.car import Car
from engines.render_engine_context import RenderEngineContext
from entities.track import Track
from utils.geometry import _precompute_segment_data

class GameContext:
    def __init__(self, physics_engine: PhysicsEngine, render_engine: RenderEngine, cars: List[Car], draw_rewards: dict[Car, bool], draw_rays: dict[Car, bool], rays_count: dict[Car, int], track: Track, think_every: dict[Car, Tuple[int, int]], car_person_images: dict[Car, pygame.surface]):
        self.physics_engine = physics_engine
        self.render_engine = render_engine
        self.cars = cars
        self.draw_rewards = draw_rewards
        for car in cars:
            if car not in draw_rewards:
                self.draw_rewards[car] = False
        self.draw_rays = draw_rays
        for car in cars:
            if car not in draw_rays:
                self.draw_rays[car] = False
        self.rays_count = rays_count
        self.track = track
        self.think_every = think_every
        for car in cars:
            if car not in think_every:
                self.think_every[car] = (0, 1)
        self.car_person_images = car_person_images

class Game:
    def __init__(self, context: GameContext):
        self.context = context
        cars = context.cars
        self.cars_reward_ind = {car : 0 for car in cars}
        self.last_reward_collected = {car: 0 for car in cars}

        spawn_position = self.context.track.spawn_point
        spawn_rotation = self.context.track.spawn_rotation
        for car in cars:
            car.velocity *= 0
            car.position = spawn_position
            car.angle = spawn_rotation
        self.cars_deaths = {car : 0 for car in cars}
        self.time_elapsed = 0

        self.numpy_walls = np.array([[(w.start.x, w.start.y), (w.end.x, w.end.y)] for w in context.track.walls])
        self.precomputed_numpy_walls = _precompute_segment_data(context.track.walls)

        self.prev_action = {car: [] for car in cars}
        self.step = 0

    def next_frame(self, delta_time: float):
        self.time_elapsed += delta_time
        render_context = RenderEngineContext(
            cars = self.context.cars,
            track = self.context.track,
            precomputed_numpy_walls=self.precomputed_numpy_walls,
            cars_reward_ind = self.cars_reward_ind,
            draw_rewards = self.context.draw_rewards,
            rays_count = self.context.rays_count,
            draw_rays = self.context.draw_rays,
            car_person_images = self.context.car_person_images
        )
        physics_context = PhysicsEngineContext(
            step=self.step,
            cars = self.context.cars,
            track = self.context.track,
            numpy_walls=self.numpy_walls,
            precomputed_numpy_walls=self.precomputed_numpy_walls,
            cars_reward_ind=self.cars_reward_ind,
            delta_time = delta_time,
            rays_count = self.context.rays_count,
            cars_deaths = self.cars_deaths,
            last_reward_collected = self.last_reward_collected,
            time_elapsed = self.time_elapsed,
            think_every=self.context.think_every,
            prev_action=self.prev_action,
        )
        self.context.physics_engine.next_frame(physics_context)
        self.context.render_engine.next_frame(render_context)

        self.step += 1