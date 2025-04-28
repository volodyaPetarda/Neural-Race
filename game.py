from typing import List

from engines.physics_engine import PhysicsEngine
from engines.physics_engine_context import PhysicsEngineContext
from engines.render_engine import RenderEngine
from car import Car
from engines.render_engine_context import RenderEngineContext
from track import Track

class GameContext:
    def __init__(self, physics_engine: PhysicsEngine, render_engine: RenderEngine, cars: List[Car], draw_rewards: dict[Car, bool], draw_rays: dict[Car, bool], rays_count: dict[Car, int], track: Track):
        self.physics_engine = physics_engine
        self.render_engine = render_engine
        self.cars = cars
        self.draw_rewards = draw_rewards
        self.draw_rays = draw_rays
        self.rays_count = rays_count
        self.track = track

class Game:
    def __init__(self, context: GameContext):
        self.context = context
        cars = context.cars
        self.cars_reward_ind = {car : 0 for car in cars}

    def next_frame(self, delta_time: float):
        render_context = RenderEngineContext(
            cars = self.context.cars,
            track = self.context.track,
            cars_reward_ind = self.cars_reward_ind,
            draw_rewards = self.context.draw_rewards,
            rays_count = self.context.rays_count,
            draw_rays = self.context.draw_rays,
        )
        physics_context = PhysicsEngineContext(
            cars = self.context.cars,
            track = self.context.track,
            cars_reward_ind=self.cars_reward_ind,
            delta_time = delta_time,
            rays_count = self.context.rays_count
        )
        self.context.physics_engine.next_frame(physics_context)
        self.context.render_engine.next_frame(render_context)
