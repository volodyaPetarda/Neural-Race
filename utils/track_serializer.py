import os

from entities.figures import Vector2D, Segment
from entities.track import Track


class TrackSerializer:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def save(self, track: Track, name: str):
        with open(os.path.join(self.base_path, name), 'w') as file:
            spawn_point = track.spawn_point
            spawn_rotation = track.spawn_rotation
            walls = track.walls
            rewards = track.rewards

            print(f"spawn_point: {spawn_point.x} {spawn_point.y}", file=file)
            print(f"spawn_rotation: {spawn_rotation}", file=file)
            print(f"len_walls: {len(walls)}", file=file)
            for wall in walls:
                print(f"wall: {wall.x1} {wall.y1} {wall.x2} {wall.y2}", file=file)
            print(f"len_rewards: {len(rewards)}", file=file)
            for reward in rewards:
                print(f"reward: {reward.x1} {reward.y1} {reward.x2} {reward.y2}", file=file)

    def load(self, name: str) -> Track:
        with open(os.path.join(self.base_path, name), 'r') as file:
            input_spawn_point = file.readline().split()
            spawn_point = Vector2D(float(input_spawn_point[1]), float(input_spawn_point[2]))

            input_spawn_rotation = file.readline().split()
            spawn_rotation = float(input_spawn_rotation[1])

            input_len_walls = file.readline().split()
            len_walls = int(input_len_walls[1])
            walls = []
            for _ in range(len_walls):
                input_wall = file.readline().split()
                wall = Segment(
                    Vector2D(float(input_wall[1]), float(input_wall[2])),
                    Vector2D(float(input_wall[3]), float(input_wall[4]))
                )
                walls.append(wall)

            len_rewards_input = file.readline().split()
            len_rewards = int(len_rewards_input[1])
            rewards = []
            for _ in range(len_rewards):
                input_reward = file.readline().split()
                reward = Segment(
                    Vector2D(float(input_reward[1]), float(input_reward[2])),
                    Vector2D(float(input_reward[3]), float(input_reward[4]))
                )
                rewards.append(reward)

            return Track(
                spawn_point=spawn_point,
                spawn_rotation=spawn_rotation,
                walls=walls,
                rewards=rewards
            )