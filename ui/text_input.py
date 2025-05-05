from typing import List

import pygame
import time


class TextInput:
    def __init__(self, x: int, y: int, width: int, height: int,
                 font: pygame.font.Font, screen: pygame.Surface,
                 text_color=(0, 0, 0),
                 bg_color=(255, 255, 255),
                 active_color=(200, 200, 255),
                 max_length=20):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.screen = screen
        self.text_color = text_color
        self.bg_color = bg_color
        self.active_color = active_color
        self.max_length = max_length

        self.text = ""
        self.active = False
        self.cursor_visible = True
        self.last_cursor_toggle = 0
        self.now = 0
        self.cursor_blink_interval = 0.5

    def next_frame(self, delta_time: float, events: List[pygame.event.Event]) -> str:
        self.handle_events(events)
        self.update(delta_time)
        self.draw()
        return self.text

    def draw(self):
        bg_color = self.active_color if self.active else self.bg_color
        pygame.draw.rect(self.screen, bg_color, self.rect)

        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(midleft=(self.rect.x + 5, self.rect.centery))
        self.screen.blit(text_surface, text_rect)

        if self.active and self.cursor_visible:
            cursor_x = text_rect.right + 2
            cursor_rect = pygame.Rect(cursor_x, self.rect.y + 5, 2, self.rect.height - 10)
            pygame.draw.rect(self.screen, self.text_color, cursor_rect)

    def update(self, delta_time: float):
        self.now += delta_time
        if self.now - self.last_cursor_toggle > self.cursor_blink_interval:
            self.cursor_visible = not self.cursor_visible
            self.last_cursor_toggle = self.now

    def handle_events(self, events: List[pygame.event.Event]):
        for event in events:
            if event.type == pygame.KEYDOWN and self.active:
                if event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif event.key == pygame.K_RETURN:
                    self.active = False
                elif len(self.text) < self.max_length:
                    self.text += event.unicode
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                self.active = self.rect.collidepoint(pos)