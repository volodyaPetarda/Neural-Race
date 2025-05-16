from enum import Enum
from typing import List

import pygame
import time

from brains.bot_brain import BotBrain, Trainer
from brains.user_brain import UserBrain
from engines.physics_engine import PhysicsEngine
from engines.render_engine import RenderEngine
from entities.car import Car
from entities.figures import Vector2D
from entities.game import Game, GameContext
from ui.images_generator import ImageGenerator
from ui.text_input import TextInput
from ui.trace_builder import TraceBuilder
from ui.button import Button
from utils.track_serializer import TrackSerializer


class State(Enum):
    START_MENU = 1,
    GAME = 2,
    TRACE_BUILDER = 3,
    IMAGE_GENERATOR = 4,

class Menu:
    def __init__(self, screen):
        self.screen = screen

        trace_button_image = pygame.image.load("data/images/build_trace.png")
        trace_button_pressed = pygame.image.load("data/images/build_trace_pressed.png")
        self.build_trace_button = Button(300, 300, trace_button_image, trace_button_pressed, 600, 200, self.screen)


        start_button_image = pygame.image.load("data/images/start_trace.png")
        start_button_pressed = pygame.image.load("data/images/start_trace_pressed.png")
        self.start_trace_button = Button(1000, 300, start_button_image, start_button_pressed, 600, 200, self.screen)

        self.game = None
        self.trace_builder = None
        self.image_generator = None
        self.state = State.START_MENU

    def run(self):
        TARGET_FPS = 60
        frames = 0
        gstart_time = time.perf_counter()
        running = True
        last_time = time.perf_counter()
        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
            start_time = time.perf_counter()
            delta_time = max(1/TARGET_FPS, time.perf_counter() - last_time)
            delta_time = min(delta_time, 2/TARGET_FPS)
            self.screen.fill((0, 0, 0))

            if self.state == State.START_MENU:
                self._handle_start_menu(events)
            elif self.state == State.TRACE_BUILDER:
                self._handle_builder(delta_time, events)
            elif self.state == State.IMAGE_GENERATOR:
                self._handle_image_generator(delta_time, events)
            elif self.state == State.GAME:
                self._handle_game(delta_time)

            pygame.display.flip()

            # game.next_frame(max(1 / TARGET_FPS, time.perf_counter() - last_time))
            end_time = time.perf_counter()
            if end_time - start_time < 1 / TARGET_FPS:
                time.sleep(1 / TARGET_FPS - (end_time - start_time))

            last_time = start_time
            frames += 1
            # print(frames / (time.perf_counter() - gstart_time))


    def _handle_start_menu(self, events: List[pygame.event.Event]):
        if self.build_trace_button.next_frame(events):
            self.trace_builder = TraceBuilder(self.screen)
            self.state = State.TRACE_BUILDER
        if self.start_trace_button.next_frame(events):
            self.image_generator = ImageGenerator(self.screen)
            self.state = State.IMAGE_GENERATOR

    def _handle_builder(self, delta_time, events: List[pygame.event.Event]):
        self.trace_builder.next_frame(delta_time, events)
        if self.trace_builder.exited:
            self.state = State.START_MENU

    def _handle_image_generator(self, delta_time, events: List[pygame.event.Event]):
        self.image_generator.next_frame(delta_time, events)
        if self.image_generator.exited:
            filename = self.image_generator.trace_name_input.text
            person_images = self.image_generator.get_images(128)

            track_serializer = TrackSerializer(base_path='data/races')
            track = track_serializer.load(filename)

            images = [
                pygame.image.load('data/images/car_' + str(i) + '.png') for i in range(1, 6)
            ]
            user_car = Car(
                UserBrain(),
                Vector2D(0, 0),
                Vector2D(0, 0),
                0,
                images[0]
            )

            trainer = Trainer(32)
            bot_cars = [
                Car(
                    BotBrain(trainer),
                    Vector2D(0, 0),
                    Vector2D(0, 0),
                    0,
                    images[i % 4 + 1]
                ) for i in range(15)
            ]
            think_every = {
                car: (i % 5, 5) for i, car in enumerate(bot_cars)
            }
            car_person_images = {
                user_car: person_images[0]
            }
            for car, person_image in zip(bot_cars, person_images[1:]):
                car_person_images[car] = person_image
            cars = [user_car] + bot_cars

            rays_count = {car: 16 for car in bot_cars}
            rays_count[user_car] = 0
            game_context = GameContext(PhysicsEngine(), RenderEngine(self.screen), cars, {user_car: True}, {}, rays_count, track, think_every, car_person_images)
            self.game = Game(game_context)

            self.state = State.GAME

    def _handle_game(self, delta_time: float):
        self.game.next_frame(delta_time)