from typing import List

from car import Car
from actions import AccelerateAction, RotateAction, NitroAction, BackAction
from engines.physics_engine_context import PhysicsEngineContext
from figures import Vector2D, dot
from geometry import is_segment_rectangle_intersection


class PhysicsEngine:
    def __init__(self):
        pass

    def next_frame(self, context: PhysicsEngineContext):
        self._handle_cars(context)

    def _handle_cars(self, context: PhysicsEngineContext):
        cars = context.cars
        for car in cars:
            self._handle_actions(context, car)
            self._handle_passive_car(context, car)
            self._handle_car_walls_collisions(context, car)
            self._handle_car_rewards_collision(context, car)

    def _handle_passive_car(self, context: PhysicsEngineContext, car: Car):
        delta_time = context.delta_time
        car.position += car.velocity * delta_time
        ang = dot(car.velocity.normalized(), Vector2D.one_rotated(car.angle)) ** 6
        car.velocity *= (0.5 * ang + (1 - ang) * 0.01) ** delta_time

    def _handle_actions(self, context: PhysicsEngineContext, car: Car):
        delta_time = context.delta_time
        actions = car.get_actions()

        have_nitro = any(isinstance(action, NitroAction) for action in actions)

        for action in actions:
            if isinstance(action, AccelerateAction):
                delta_speed = 1000 if have_nitro else 500
                car.velocity += Vector2D.one_rotated(car.angle) * delta_time * delta_speed

            if isinstance(action, BackAction):
                car.velocity *= 0.2 ** delta_time
                delta_speed = 300 if have_nitro else 200
                car.velocity += Vector2D.one_rotated(car.angle) * -1 * delta_time * delta_speed

            if isinstance(action, RotateAction):
                speed = Vector2D.magnitude(car.velocity)
                if speed < 150:
                    angle_velocity_delta = speed / 15
                else:
                    angle_velocity_delta = min(max(0.5, 1500 / speed), 10)
                print(angle_velocity_delta)
                drift = max(0, 1 - speed / 1500)
                rotation_amount = delta_time * angle_velocity_delta

                if action.direction == "left":
                    car.angle -= rotation_amount
                    car.velocity = car.velocity.rotate(-rotation_amount * drift)
                elif action.direction == "right":
                    car.angle += rotation_amount
                    car.velocity = car.velocity.rotate(rotation_amount * drift)


    def _handle_car_walls_collisions(self, context: PhysicsEngineContext, car: Car):
        collider = car.collider
        track = context.track
        for wall in track.walls:
            if is_segment_rectangle_intersection(wall, collider):
                car.velocity *= 0
                car.angle = track.spawn_rotation
                car.position = track.spawn_point
                break

    def _handle_car_rewards_collision(self, context: PhysicsEngineContext, car: Car):
        collider = car.collider
        track = context.track
        current_ind = context.cars_reward_ind[car]
        reward_collider = track.get_reward_segment(current_ind)
        if is_segment_rectangle_intersection(reward_collider, collider):
            context.cars_reward_ind[car] += 1