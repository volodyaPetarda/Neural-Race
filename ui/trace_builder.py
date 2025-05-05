import math
from typing import List

import pygame

from entities.track import Track
from ui.button import Button
from ui.click_manager import ClickManager
from entities.figures import Segment, Vector2D
from ui.text_input import TextInput
from utils.track_serializer import TrackSerializer


class TraceBuilder:
    def __init__(self, screen):
        self.screen = screen
        self._saving = False
        self._exited = False

        self.click_manager = ClickManager()
        self.walls = []
        self.rewards = []
        self.start_point = Vector2D(1920 // 2, 1080 // 2)
        self.start_rotation = 0
        self.last_clicked = None

        image_save = pygame.image.load("data/images/build_trace.jpg")
        image_save_pressed = pygame.image.load("data/images/build_trace_pressed.jpg")
        self.save_trace_button = Button(600, 600, image_save, image_save_pressed, 600, 150, self.screen)
        self.save_name_input = TextInput(
            600, 300, 600, 150, pygame.font.Font(None, 64), self.screen
        )

        self.track_serializer = TrackSerializer('data/races')

        #walls, rewards, start_point
        self._current_building_type = "walls"

    def next_frame(self, delta_time: float, events: List[pygame.event.Event]):
        self._manage_events(events, delta_time)
        self._manage_building()

        self._draw_walls()
        self._draw_rewards()
        self._draw_start_point()
        self._draw_start_rotation()
        self._draw_ghost_building()
        self._manage_saving(delta_time, events)

    def _manage_events(self, events: List[pygame.event.Event], delta_time: float):
        self.click_manager.next_frame(events)
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_s and (event.mod & pygame.KMOD_LCTRL):
                    if not self._saving:
                        self._saving = True

                elif event.key == pygame.K_d and (event.mod & pygame.KMOD_LCTRL):
                    if not self._exited:
                        self._exited = True
                        return

            if not self._exited and not self._saving:
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        self.last_clicked = None
                        if self._current_building_type == "walls":
                            self._current_building_type = "rewards"
                        elif self._current_building_type == "rewards":
                            self._current_building_type = "start_point"
                        elif self._current_building_type == "start_point":
                            self._current_building_type = "walls"
                    if event.key == pygame.K_f:
                        self.last_clicked = None

                if event.type == pygame.MOUSEMOTION:
                    self.last_motion = event.pos

                keys = pygame.key.get_pressed()
                if keys[pygame.K_r]:
                    self.start_rotation += math.pi / 2 * delta_time


    def _manage_building(self):
        if self.click_manager.clicked and not self._exited and not self._saving:
            pos = self.click_manager.clicked_pos
            if self._current_building_type == "walls":
                if self.last_clicked is not None:
                    self.walls.append(Segment(Vector2D(self.last_clicked[0], self.last_clicked[1]), Vector2D(pos[0], pos[1])))
                self.last_clicked = pos
            if self._current_building_type == "rewards":
                if self.last_clicked is not None:
                    self.rewards.append(Segment(Vector2D(self.last_clicked[0], self.last_clicked[1]), Vector2D(pos[0], pos[1])))
                    self.last_clicked = None
                else:
                    self.last_clicked = pos
            if self._current_building_type == "start_point":
                self.start_point = Vector2D(pos[0], pos[1])

    def _draw_walls(self):
        for wall in self.walls:
            pygame.draw.line(
                self.screen,
                (128, 128, 128),
                (wall.x1, wall.y1),
                (wall.x2, wall.y2),
                2
            )

    def _draw_rewards(self):
        for bonus_segment in self.rewards:
            pygame.draw.line(
                self.screen,
                (255, 255, 0),
                (bonus_segment.x1, bonus_segment.y1),
                (bonus_segment.x2, bonus_segment.y2),
                3
            )

    def _draw_start_point(self):
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (self.start_point.x, self.start_point.y),
            5
        )

    def _draw_start_rotation(self):
        center = self.start_point
        end = center + Vector2D.one_rotated(self.start_rotation) * 30
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            start_pos=(center.x, center.y),
            end_pos=(end.x, end.y)
        )

    def _draw_ghost_building(self):
        if self.last_clicked and not self._exited and not self._saving:
            color = None
            if self._current_building_type == "walls":
                color = (128, 128, 128)
            elif self._current_building_type == "rewards":
                color = (255, 255, 0)
            if color:
                pygame.draw.line(
                    self.screen,
                    color,
                    (self.last_clicked[0], self.last_clicked[1]),
                    (self.last_motion[0], self.last_motion[1])
                )

    def _manage_saving(self, delta_time: float, events: List[pygame.event.Event]):
        if self._saving:
            self.save_name_input.next_frame(delta_time, events)
            clicked = self.save_trace_button.next_frame(events)
            if clicked:
                track = Track(
                    spawn_point=self.start_point,
                    spawn_rotation=self.start_rotation,
                    walls=self.walls,
                    rewards=self.rewards,
                )
                name = self.save_name_input.text
                self.track_serializer.save(track, name)
                self._saving = False

    @property
    def exited(self):
        return self._exited