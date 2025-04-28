import math
import time

import pygame

from brains.user_brain import UserBrain
from car import Car
from engines.physics_engine import PhysicsEngine
from engines.render_engine import RenderEngine
from game import Game
from figures import Vector2D, Segment
from track import Track

if __name__ == "__main__":
    pygame.init()
    user_brain = UserBrain()
    user_car = Car(user_brain, Vector2D(1000, 500), Vector2D(0, 0), math.pi/2)
    physics_engine = PhysicsEngine()
    render_engine = RenderEngine()
    track = Track(
        Vector2D(1000, 500),
        math.pi/2,
        [Segment(Vector2D(100, 100), Vector2D(400, 200)), Segment(Vector2D(400, 500), Vector2D(500, 500))],
        [Segment(Vector2D(200, 200), Vector2D(300, 300)), Segment(Vector2D(500, 600), Vector2D(600, 600))],
    )
    game = Game(physics_engine, render_engine, [user_car], {user_car: True}, track)
    TARGET_FPS = 60
    last_time = time.perf_counter()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        start_time = time.perf_counter()
        game.next_frame(max(1/TARGET_FPS, time.perf_counter() - last_time))
        end_time = time.perf_counter()
        if end_time - start_time < 1 / TARGET_FPS:
            time.sleep(1 / TARGET_FPS - (end_time - start_time))

        last_time = start_time

