from typing import List

import pygame


class ClickManager:
    def __init__(self):
        self._pressed = False
        self._clicked = False
        self._clicked_pos = None

    def next_frame(self, events: List[pygame.event.Event]):
        self._clicked = False
        self._clicked_pos = None
        self.handle_events(events)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._pressed = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._clicked = True
                self._pressed = False
                self._clicked_pos = pygame.mouse.get_pos()

    @property
    def clicked(self):
        return self._clicked

    @property
    def clicked_pos(self):
        return self._clicked_pos