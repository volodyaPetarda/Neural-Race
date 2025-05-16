import pygame


class ProgressBar:
    def __init__(self, x_left_down, y_left_down, x_right_up, y_right_up, steps, screen):
        self.x_left_down = x_left_down
        self.y_left_down = y_left_down
        self.x_right_up = x_right_up
        self.y_right_up = y_right_up
        self.steps = steps
        self.current_step = 0
        self.screen = screen

    def next_frame(self):
        bar_width = self.x_right_up - self.x_left_down
        bar_height = self.y_left_down - self.y_right_up

        full_rect = pygame.Rect(
            self.x_left_down,
            self.y_right_up,
            bar_width,
            bar_height
        )
        pygame.draw.rect(self.screen, "grey", full_rect)

        progress_width = (self.current_step / self.steps) * bar_width

        progress_rect = pygame.Rect(
            self.x_left_down,
            self.y_right_up,
            progress_width,
            bar_height
        )
        pygame.draw.rect(self.screen, "green", progress_rect)

    def step(self):
        self.current_step += 1
        self.current_step = min(self.current_step, self.steps)