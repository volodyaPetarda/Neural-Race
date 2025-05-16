import math
import time

import pygame

from brains.bot_brain import BotBrain, Trainer
from brains.user_brain import UserBrain
from entities.car import Car
from engines.physics_engine import PhysicsEngine
from engines.render_engine import RenderEngine
from entities.game import Game, GameContext
from entities.figures import Vector2D, Segment
from entities.track import Track
from ui.button import Button
from utils.track_serializer import TrackSerializer

if __name__ == "__main__":
    pygame.init()
    user_brain = UserBrain()
    images = [
        pygame.image.load('data/images/car_' + str(i) + '.png') for i in range(1, 6)
    ]
    user_car = Car(user_brain, Vector2D(1000, 500), Vector2D(0, 0), math.pi/2, images[0])

    trainer = Trainer(32)
    bot_brain = [BotBrain(trainer) for _ in range(20)]
    bot_cars = [Car(bot_brain[_], Vector2D(1000, 500), Vector2D(0, 0), math.pi/2, images[_ % 4 + 1]) for _ in range(15)]

    physics_engine = PhysicsEngine()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    render_engine = RenderEngine(screen)

    track_serializer = TrackSerializer('data/races')
    track = track_serializer.load('saving')

    rays_count = {user_car: 0}
    rays_count.update({bot_car: 16 for bot_car in bot_cars})

    think_every = {car: (i % 5, 5) for i, car in enumerate(bot_cars)}

    game_context = GameContext(
        physics_engine=physics_engine,
        render_engine=render_engine,
        cars=[user_car] + bot_cars,
        draw_rewards={user_car: True},
        draw_rays={user_car: False},
        rays_count=rays_count,
        track=track,
        think_every=think_every,
    )
    game = Game(game_context)
    TARGET_FPS = 60
    frames = 0
    gstart_time = time.monotonic()
    last_time = time.monotonic()
    running = True

    max_delta = 0
    max_diff = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        start_time = time.monotonic()
        next_frame_time = max(1/TARGET_FPS, time.monotonic() - last_time)
        if next_frame_time > max_delta:
            print('max_next', next_frame_time)
            max_delta = next_frame_time
        next_frame_time = min(next_frame_time, 1 / TARGET_FPS * 2)
        game.next_frame(next_frame_time)

        end_time = time.monotonic()
        if end_time - start_time < 1 / TARGET_FPS:
            time.sleep(1 / TARGET_FPS - (end_time - start_time))


        last_time = start_time
        frames += 1
        if frames % 10 == 0:
            print('frames', frames / (time.monotonic() - gstart_time))

    pygame.quit()