import math
from typing import List

import pygame

from car import Car
from engines.render_engine_context import RenderEngineContext
from track import Track


class RenderEngine:
    def __init__(self, screen_size=(1920, 1080)):
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Car Race Game")

    def next_frame(self, context: RenderEngineContext):
        self.screen.fill((0, 0, 0))

        self._draw_walls(context)
        self._draw_rewards(context)
        self._draw_cars(context)

        pygame.display.flip()

    def _draw_cars(self, context: RenderEngineContext):
        cars = context.cars
        for car in cars:
            car_rect = pygame.Rect(
                int(car.position.x),
                int(car.position.y),
                car.width,
                car.length
            )
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
                    (255, 255, 0),  # Желтый цвет
                    (bonus_segment.x1, bonus_segment.y1),
                    (bonus_segment.x2, bonus_segment.y2),
                    3  # Толщина линии
                )

