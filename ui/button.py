from typing import List

import pygame


class Button:
    def __init__(self, x: int, y: int, image, pressed_image, width: int, height: int, screen):
        self.image = pygame.transform.scale(image, (width, height))
        self.pressed_image = pygame.transform.scale(pressed_image, (width, height))
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self._pressed = False
        self._clicked = False
        self.screen = screen

    def next_frame(self, events: List[pygame.event.Event]) -> bool:
        self.handle_events(events)
        self.draw()
        return self.clicked

    def draw(self):
        image_to_show = self.pressed_image if self._pressed else self.image
        self.screen.blit(image_to_show, self.rect)

    def handle_events(self, events: List[pygame.event.Event]):
        self._clicked = False
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if self.rect.collidepoint(pos):
                    self._pressed = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._pressed:
                self._pressed = False
                self._clicked = True

    @property
    def clicked(self):
        return self._clicked