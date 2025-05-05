import copy

from entities.car import Car
from entities.actions import AccelerateAction, RotateAction, NitroAction, BackAction
from engines.physics_engine_context import PhysicsEngineContext
from entities.car_state import CarState
from entities.figures import Vector2D, dot, get_batch_rays, Segment, get_numpy_batch_rays
from entities.player_state import PlayerState
from utils.geometry import is_segment_rectangle_intersection, is_batch_segments_rectangle_intersection, \
    get_batch_ray_intersect_segments, segment_batch_intersect, segment_intersect, get_rays_intersect_segment, \
    _precompute_segment_data

class PhysicsEngine:
    def __init__(self):
        pass

    def next_frame(self, context: PhysicsEngineContext):

        self._handle_cars(context)

    def _handle_cars(self, context: PhysicsEngineContext):
        cars = context.cars
        for car in cars:
            prev_car_pos = copy.deepcopy(car.position)
            self._handle_actions(context, car)
            self._handle_passive_car(context, car)
            next_car_pos = copy.deepcopy(car.position)
            self._handle_car_walls_collisions(context, car, prev_car_pos, next_car_pos)
            self._handle_car_rewards_collision(context, car, prev_car_pos, next_car_pos)

    def _handle_passive_car(self, context: PhysicsEngineContext, car: Car):
        delta_time = context.delta_time
        car.position += car.velocity * delta_time
        ang = dot(car.velocity.normalized(), Vector2D.one_rotated(car.angle)) ** 6
        car.velocity *= (0.5 * ang + (1 - ang) * 0.01) ** delta_time

    def _handle_actions(self, context: PhysicsEngineContext, car: Car):
        delta_time = context.delta_time

        player_state = PlayerState(
            context.cars_reward_ind[car],
            context.cars_deaths[car],
            context.time_elapsed,
            context.delta_time,
            context.last_reward_collected[car]
        )
        car_state = CarState(
            velocity=car.velocity,
            angle=car.angle
        )

        rays_count = context.rays_count
        numpy_rays = get_numpy_batch_rays(car.position, car.angle, rays_count[car])
        walls_intersects = get_batch_ray_intersect_segments(numpy_rays, context.precomputed_numpy_walls)
        reward_ind = context.cars_reward_ind[car]
        reward_segment = context.track.get_reward_segment(reward_ind)
        rewards_intersects = get_batch_ray_intersect_segments(numpy_rays, _precompute_segment_data([reward_segment]))
        intersects = walls_intersects + rewards_intersects
        rays_dists = [
            (intersect - car.position).magnitude() if intersect else 1000 for intersect in intersects
        ]

        actions = car.get_actions(player_state, car_state, rays_dists)

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
                    angle_velocity_delta = speed / 30
                else:
                    angle_velocity_delta = min(max(0.5, 1500 / speed), 5)
                drift = max(0, 1 - speed / 1500)
                rotation_amount = delta_time * angle_velocity_delta

                if action.direction == "left":
                    car.angle -= rotation_amount
                    car.velocity = car.velocity.rotate(-rotation_amount * drift)
                elif action.direction == "right":
                    car.angle += rotation_amount
                    car.velocity = car.velocity.rotate(rotation_amount * drift)


    def _handle_car_walls_collisions(self, context: PhysicsEngineContext, car: Car, prev_car_pos: Vector2D, next_car_pos: Vector2D):
        collider = car.collider
        track = context.track
        numpy_walls = context.numpy_walls
        precomputed_numpy_walls = context.precomputed_numpy_walls
        if is_batch_segments_rectangle_intersection(numpy_walls, collider) or segment_batch_intersect(Segment(prev_car_pos, next_car_pos), precomputed_numpy_walls):
            car.velocity *= 0
            car.angle = track.spawn_rotation
            car.position = track.spawn_point
            context.cars_reward_ind[car] = 0
            context.cars_deaths[car] += 1
            context.last_reward_collected[car] = context.time_elapsed

    def _handle_car_rewards_collision(self, context: PhysicsEngineContext, car: Car, prev_car_pos: Vector2D, next_car_pos: Vector2D):
        collider = car.collider
        track = context.track
        current_ind = context.cars_reward_ind[car]
        reward_collider = track.get_reward_segment(current_ind)
        if is_segment_rectangle_intersection(reward_collider, collider) or segment_intersect(Segment(prev_car_pos, next_car_pos), reward_collider):
            context.cars_reward_ind[car] += 1
            context.last_reward_collected[car] = context.time_elapsed
            self._handle_car_rewards_collision(context, car, prev_car_pos, next_car_pos)