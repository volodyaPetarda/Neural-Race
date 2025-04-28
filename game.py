from typing import List

from engines.physics_engine import PhysicsEngine
from engines.physics_engine_context import PhysicsEngineContext
from engines.render_engine import RenderEngine
from car import Car
from engines.render_engine_context import RenderEngineContext
from track import Track


class Game:
    def __init__(self, physics_engine: PhysicsEngine, render_engine: RenderEngine, cars: List[Car], draw_rewards: dict[Car, bool], track: Track):
        self.physics_engine = physics_engine
        self.render_engine = render_engine
        self.cars = cars
        self.draw_rewards = draw_rewards
        self.cars_reward_ind = {car : 0 for car in self.cars}
        self.track = track

    def next_frame(self, delta_time: float):
        render_context = RenderEngineContext(
            cars = self.cars,
            track = self.track,
            cars_reward_ind = self.cars_reward_ind,
            draw_rewards = self.draw_rewards,
        )
        physics_context = PhysicsEngineContext(
            cars = self.cars,
            track = self.track,
            cars_reward_ind=self.cars_reward_ind,
            delta_time = delta_time
        )
        self.physics_engine.next_frame(physics_context)
        self.render_engine.next_frame(render_context)
