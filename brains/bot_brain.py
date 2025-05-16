import copy
import math
import random
from typing import List

import torch
from torch import optim

from entities.actions import BaseAction, RotateAction, AccelerateAction, NitroAction, BackAction
from brains.base_brain import BaseBrain
from brains.base_brain import BaseBrainType
from entities.car_state import CarState
from entities.player_state import PlayerState
from entities.q_network import QNetwork


class Trainer:
    def __init__(self, num_rays: int):

        self.q_net = QNetwork([num_rays + 4, 128, 128, 64, 32])
        self.target_net = QNetwork([num_rays + 4, 128, 128, 64, 32])
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()


        self.optimizer = optim.Adam(self.q_net.parameters(), lr=5e-4)

        self.batch_size = 64
        self.min_size_for_train = 5000
        self.train_every = 3
        self.copy_every = 2048 * 4
        self.gamma = 0.95
        self.step = 0
        self.max_len = 1000000
        self.delta_len = 1000
        self.start_ind = 0
        self.end_ind = 0
        self.replay_buffer = [None] * self.max_len

    def add_dataset(self, example):
        self.replay_buffer[self.end_ind % self.max_len] = copy.deepcopy(example)
        self.end_ind += 1
        if self.end_ind - self.start_ind >= self.max_len:
            self.start_ind += self.delta_len


    def predict(self, input_data):
        self.q_net.eval()
        with torch.no_grad():
            prediction = self.q_net(input_data)
            return prediction

    def train(self):
        if self.step % self.copy_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.step += 1
        if self.step % self.train_every != 0:
            return
        self.q_net.train()
        if self.end_ind - self.start_ind < self.min_size_for_train:
            return

        inds = random.choices(range(self.start_ind, self.end_ind), k=self.batch_size)
        for i in range(len(inds)):
            inds[i] %= self.max_len
        player_states = [self.replay_buffer[i][0] for i in inds]
        next_player_states = [self.replay_buffer[i][3] for i in inds]
        car_states = [self.replay_buffer[i][1] for i in inds]
        next_car_states = [self.replay_buffer[i][4] for i in inds]
        rays_dists = [self.replay_buffer[i][2] for i in inds]
        next_rays_dists = [self.replay_buffer[i][5] for i in inds]
        actions = [self.replay_buffer[i][6] for i in inds]
        done = torch.tensor([next_player_state.deaths - player_state.deaths for player_state, next_player_state in
                             zip(player_states, next_player_states)])

        rewards = torch.tensor([self.reward(player_state, car_state, next_player_state, next_car_state) for player_state, next_player_state, car_state, next_car_state in
                                zip(player_states, next_player_states, car_states, next_car_states)])
        first_input_data = self.concat_to_input(rays_dists, car_states, player_states)
        second_input_data = self.concat_to_input(next_rays_dists, next_car_states, next_player_states)
        output = self.q_net(first_input_data)
        with torch.no_grad():
            # vanilla dqn
            # target_output = (1 - done) * self.target_net(second_input_data).detach().max(dim=-1).values * self.gamma + rewards

            next_actions = self.q_net(second_input_data).argmax(dim=-1).detach()
            target_next_q = self.target_net(second_input_data).detach()
            next_q_values = target_next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
            target_output = ((1 - done) * next_q_values * self.gamma + rewards).detach()

        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        q_values = output.gather(1, actions_tensor).squeeze(1)
        loss = ((q_values - target_output) ** 2).mean()
        if loss.item() > 1e8:
            print('warning')
            print('q_values', q_values)
            print('target', target_output)
        # loss = torch.nn.functional.smooth_l1_loss(q_values, target_output)
        with torch.no_grad():
            stupid_loss = ((target_output - target_output.mean())**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if random.random() < 0.01:
            print(self.step, f'loss={loss.item()}, stupid_loss={(stupid_loss.item())}')
            print(f'q_values {q_values[0:5]}')
            print(f'target_output {target_output[0:5]}')
        self.optimizer.step()

    def reward(self, prev_player_state: PlayerState|None, prev_car_state: CarState|None, player_state: PlayerState, car_state: CarState):
        if prev_player_state is None:
            return 0
        reward = 0
        reward += max(0, player_state.rewards_collected - prev_player_state.rewards_collected) * 0.5
        reward += (player_state.deaths - prev_player_state.deaths) * -5
        delta_time = (player_state.time_elapsed - prev_player_state.time_elapsed)
        reward += (player_state.time_elapsed - player_state.last_reward_collected + 0.5) * -0.25 * delta_time
        return reward


    def concat_to_input(self, rays_dists: List[List[float]], car_states: List[CarState], player_states: List[PlayerState]):
        rays_dists = torch.tensor(rays_dists)
        velocities = [car_state.velocity.rotate(-car_state.angle) for car_state in car_states]
        car_states = torch.tensor([[velocity.x, velocity.y] for velocity in velocities])
        player_states = torch.tensor([[player_state.delta_time, player_state.time_elapsed - player_state.last_reward_collected] for player_state in player_states])
        return torch.cat((rays_dists, car_states, player_states), dim=1) / torch.tensor([1000.0 for _ in range(len(rays_dists[0]))] + [200, 200] + [1/60, 1])


class BotBrain(BaseBrain):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.eps = 1.0
        self.prev_player_state = None
        self.prev_car_state = None
        self.prev_rays_dists = None
        self.prev_action = None
        self.step = 0

    def get_brain_type(self) -> BaseBrainType:
        return BaseBrainType.BotBrain

    def get_actions(self, player_state: PlayerState, car_state: CarState, rays_dists: List[float]) -> List[BaseAction]:
        self.trainer.train()
        input_data = self.trainer.concat_to_input([rays_dists], [car_state], [player_state])
        output = self.trainer.predict(input_data)
        self.eps = math.exp(-(player_state.time_elapsed - 30) / 150)
        self.eps = max(self.eps, 0.01)
        self.eps = min(self.eps, 1)


        if random.random() < 0.0001:
            print(f"eps={self.eps}")

        action = output[0].argmax()
        if random.random() < self.eps:
            action = random.randint(0, 31)

        result = []
        if action & 1:
            result.append(RotateAction("left"))
        if action & 2:
            result.append(RotateAction("right"))
        if action & 4:
            result.append(AccelerateAction())
        if action & 8:
            result.append(NitroAction())
        if action & 16:
            result.append(BackAction())
        if self.prev_player_state is not None:
            self.trainer.add_dataset((self.prev_player_state, self.prev_car_state, self.prev_rays_dists, player_state, car_state, rays_dists, self.prev_action))
        self.prev_rays_dists = rays_dists
        self.prev_player_state = player_state
        self.prev_car_state = car_state
        self.prev_action = action
        return result