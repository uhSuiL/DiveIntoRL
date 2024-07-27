import util
import random

import numpy as np
import gymnasium as gym
import torch as th

from torch import nn
from ch7 import DQN, run_plot_dqn, Qnet, ReplayBuffer


class DoubleDQN(DQN):
	def q_targets(self, r, s_, done):
		next_values = self.target_qnet(s_) * (1 - done)
		_, best_a = self.qnet(s_).max(dim=-1, keepdim=True)
		max_next_values = next_values.gather(dim=-1, index=best_a)
		targets = r + self.gam * max_next_values
		return targets


def valid_double_dqn():
	SEED = 0

	random.seed(SEED)
	np.random.seed(SEED)
	th.manual_seed(SEED)

	env = gym.make('CartPole-v0')
	qnet = Qnet(
		num_state=env.observation_space.shape[0],
		hidden_dims=[128],
		num_action=env.action_space.n,
		device='cuda' if th.cuda.is_available() else 'cpu',
	)
	double_dqn = DoubleDQN(
		qnet,
		learning_rate=2e-3,
		epsilon=0.01,
		gamma=0.98,
		update_period=10,
	)
	replay_buffer = ReplayBuffer(
		min_size=500,
		capacity=10000,
	)

	run_plot_dqn(
		env,
		replay_buffer,
		double_dqn,
		batch_size=64,
		num_episode=500,
		name='double_dqn'
	)


@util.lazy_init
class VAnet(nn.Module):
	def __init__(self, state_dim: int, num_action: int, shared_net_params: list, device):
		super().__init__()
		self.num_action = num_action
		self.device = device

		shared_net_layers = [nn.Linear(state_dim, shared_net_params[0]), nn.ReLU()]
		for i in range(len(shared_net_params) - 1):
			shared_net_layers += [nn.Linear(shared_net_params[i], shared_net_params[i+1]), nn.ReLU()]
		self.shared_net = nn.Sequential(*shared_net_layers)

		self.v_head = nn.Linear(shared_net_params[-1], 1)
		self.a_head = nn.Linear(shared_net_params[-1], num_action)

		self.to(device)

	def forward(self, state: th.tensor):
		basic_patterns = self.shared_net(state)
		V = self.v_head(basic_patterns)  # (batch_size, 1)
		Adv = self.a_head(basic_patterns)  # (batch_size, num_action)
		Adv_mean = Adv.mean(dim=-1, keepdim=True)  # (batch_size, 1)
		Q = V + Adv - Adv_mean
		return Q


def valid_dueling_dqn():
	SEED = 0

	random.seed(SEED)
	np.random.seed(SEED)
	th.manual_seed(SEED)

	env = gym.make('CartPole-v0')
	va_net = VAnet(
		state_dim=env.observation_space.shape[0],
		num_action=env.action_space.n,
		shared_net_params = [128],
		device='cuda' if th.cuda.is_available() else 'cpu',
	)
	double_dqn = DQN(
		va_net,
		learning_rate=2e-3,
		epsilon=0.01,
		gamma=0.98,
		update_period=10,
	)
	replay_buffer = ReplayBuffer(
		min_size=500,
		capacity=10000,
	)

	run_plot_dqn(
		env,
		replay_buffer,
		double_dqn,
		batch_size=64,
		num_episode=500,
		name='dueling_dqn'
	)


if __name__ == '__main__':
	# valid_double_dqn()
	valid_dueling_dqn()
