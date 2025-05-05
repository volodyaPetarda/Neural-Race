from enum import Enum
from typing import List

import pygame
import time

from ui.text_input import TextInput
from ui.trace_builder import TraceBuilder
from ui.button import Button


class State(Enum):
    START_MENU = 1,
    GAME = 2,
    TRACE_BUILDER = 3,

class Menu:
    def __init__(self, screen):
        self.screen = screen

        trace_button_image = pygame.image.load("data/images/build_trace.jpg")
        trace_button_pressed = pygame.image.load("data/images/build_trace_pressed.jpg")
        self.build_trace_button = Button(300, 300, trace_button_image, trace_button_pressed, 200, 200, self.screen)
        self.game = None
        self.trace_builder = None
        self.state = State.START_MENU

        self.text_input = TextInput(
            1500, 500, 200, 50, pygame.font.Font(None, 32), self.screen
        )

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
            self.screen.fill((0, 0, 0))

            if self.state == State.START_MENU:
                self._handle_start_menu(events)
            elif self.state == State.TRACE_BUILDER:
                self._handle_builder(delta_time, events)

            self.text_input.next_frame(delta_time, events)

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

    def _handle_builder(self, delta_time, events: List[pygame.event.Event]):
        self.trace_builder.next_frame(delta_time, events)
        if self.trace_builder.exited:
            self.state = State.START_MENU