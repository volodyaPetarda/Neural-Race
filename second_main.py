import math
import time

import pygame

from brains.user_brain import UserBrain
from entities.car import Car
from engines.physics_engine import PhysicsEngine
from engines.render_engine import RenderEngine
from entities.game import Game, GameContext
from entities.figures import Vector2D, Segment
from entities.track import Track
from ui.button import Button
from ui.menu import Menu

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    menu = Menu(screen)

    menu.run()

    pygame.quit()