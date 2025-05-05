import math

import pygame

from engines.render_engine_context import RenderEngineContext
from entities.figures import get_batch_rays, get_numpy_batch_rays
from utils.geometry import get_batch_ray_intersect_segments, get_rays_intersect_segment, _precompute_segment_data


class RenderEngine:
    def __init__(self, screen):
        self.screen = screen
        pygame.display.set_caption("Car Race Game")

    def next_frame(self, context: RenderEngineContext):
        self.screen.fill((0, 0, 0))
        self._draw_rays(context)
        self._draw_walls(context)
        self._draw_rewards(context)
        self._draw_cars(context)

        pygame.display.flip()

    def _draw_cars(self, context: RenderEngineContext):
        cars = context.cars
        for car in cars:
            car_surface = pygame.Surface((car.length, car.width), pygame.SRCALPHA)
            pygame.draw.rect(car_surface, (255, 0, 0), (0, 0, car.length, car.width))

            rotated_surface = pygame.transform.rotate(car_surface, -car.angle * 180 / math.pi)

            rotated_rect = rotated_surface.get_rect(center=(int(car.position.x), int(car.position.y)))

            self.screen.blit(rotated_surface, rotated_rect.topleft)

    def _draw_walls(self, context: RenderEngineContext):
        track = context.track
        for wall in track.walls:
            pygame.draw.line(
                self.screen,
                (128, 128, 128),
                (wall.x1, wall.y1),
                (wall.x2, wall.y2),
                2
            )

    def _draw_rewards(self, context: RenderEngineContext):
        cars = context.cars
        draw_rewards = context.draw_rewards
        track = context.track
        cars_reward_ind = context.cars_reward_ind
        for car in cars:
            if draw_rewards[car]:
                bonus_segment = track.get_reward_segment(cars_reward_ind[car])
                pygame.draw.line(
                    self.screen,
                    (255, 255, 0),
                    (bonus_segment.x1, bonus_segment.y1),
                    (bonus_segment.x2, bonus_segment.y2),
                    3
                )

    def _draw_rays(self, context: RenderEngineContext):
        cars = context.cars
        draw_rays = context.draw_rays
        rays_count = context.rays_count
        precomputed_numpy_walls = context.precomputed_numpy_walls
        for car in cars:
            if not draw_rays[car]:
                continue

            precomputed_rays = get_numpy_batch_rays(car.position, car.angle, rays_count[car])
            walls_intersects = get_batch_ray_intersect_segments(precomputed_rays, precomputed_numpy_walls)

            for wall_intersect in walls_intersects:
                if wall_intersect is None:
                    continue
                pygame.draw.line(
                    self.screen,
                    (150, 150, 150),
                    (car.position.x, car.position.y),
                    (wall_intersect.x, wall_intersect.y),
                1
                )

            reward = context.track.get_reward_segment(context.cars_reward_ind[car])
            reward_intersects = get_batch_ray_intersect_segments(precomputed_rays, _precompute_segment_data([reward]))

            for reward_intersect in reward_intersects:
                if reward_intersect is None:
                    continue
                pygame.draw.line(
                    self.screen,
                    (255, 255, 150),
                    (car.position.x, car.position.y),
                    (reward_intersect.x, reward_intersect.y),
                1
                )