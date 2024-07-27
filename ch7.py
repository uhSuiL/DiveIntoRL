import random
import numpy as np
import torch
import torch as th
import gymnasium as gym
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from collections import deque


class ReplayBuffer(deque):
	def __init__(self, min_size: int, capacity: int):
		super(ReplayBuffer, self).__init__(maxlen=capacity)
		self.min_size = min_size  # 要当buffer缓冲到min_size时, 才开始开始sample

	def sample(self, batch_size: int):
		assert batch_size <= self.min_size

		interaction_batch = random.sample(self, batch_size)
		s_batch, a_batch, r_batch, s_next_batch, done_batch = zip(*interaction_batch)
		return np.array(s_batch), a_batch, r_batch, np.array(s_next_batch), done_batch


class Qnet:
	def __init__(self, num_state: int, hidden_dims: list, action_dim: int, device: str):
		self.num_state = num_state
		self.hidden_dims = hidden_dims
		self.num_action = action_dim
		self.device = device

	def __call__(self, *args, **kwargs) -> nn.Module:
		layers = [nn.Linear(self.num_state, self.hidden_dims[0]), nn.ReLU()]
		for i in range(len(self.hidden_dims) - 1):
			layers += [nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]), nn.ReLU()]
		layers.append(nn.Linear(self.hidden_dims[-1], self.num_action))

		return nn.Sequential(*layers).to(self.device)


class DQN:
	def __init__(self, qnet: Qnet, *, learning_rate: float, epsilon: float, gamma: float, update_period: int):
		self.num_action = qnet.num_action
		self.device = qnet.device

		self.qnet = qnet()
		self.target_qnet = qnet()
		self.optimizer = th.optim.Adam(self.qnet.parameters(), lr=learning_rate)
		self.loss_fn = nn.MSELoss()

		self.eps = epsilon
		self.gam = gamma
		self.update_period = update_period  # 每隔多少步, 更新一次target_q_net

		self.count_update = 0

		assert self.qnet is not self.target_qnet

	@torch.no_grad()
	def policy(self, state: np.ndarray) -> int:
		"""Epsilon Greedy | return action id"""
		if np.random.uniform() < self.eps:
			return np.random.randint(self.num_action)
		else:
			state = th.tensor(state, dtype=th.float).unsqueeze(dim=0).to(self.device)  # shape: (1, state_dim)
			action_id = self.qnet(state).argmax().item()
			assert type(action_id) is int
			return action_id

	def q_values(self, s, a):
		values = self.qnet(s)
		values = values.gather(dim=-1, index=a)
		return values

	def q_targets(self, r, s_, done):
		next_values = self.target_qnet(s_) * (1 - done)  # done了, 动作价值就为0; (batch_size, action_dim)
		max_next_values, _ = next_values.max(dim=-1, keepdim=True)
		targets = r + self.gam * max_next_values
		return targets

	def update_qnet(self, batch_s, batch_a, batch_r, batch_s_, batch_done):
		batch_s = th.tensor(batch_s, dtype=th.float).to(self.device)  # shape: (batch_size, state_dim)
		batch_a = th.tensor(batch_a, dtype=th.int64).unsqueeze(dim=-1).to(self.device)
		batch_r = th.tensor(batch_r, dtype=th.float).unsqueeze(dim=-1).to(self.device)
		batch_s_ = th.tensor(batch_s_, dtype=th.float).to(self.device)  # shape: (batch_size, state_dim)
		batch_done = th.tensor(batch_done, dtype=th.float).unsqueeze(dim=-1).to(self.device)

		loss = self.loss_fn(
			self.q_values(batch_s, batch_a),
			self.q_targets(batch_r, batch_s_, batch_done),
		)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# 每隔一定步数，更新一次target_q_net的参数
		if self.count_update % self.update_period == 0:
			qnet_params = self.qnet.state_dict()
			self.target_qnet.load_state_dict(qnet_params)

		self.count_update += 1

	def train_episode(self, env: gym.Env, replay_buffer: ReplayBuffer, batch_size: int) -> float | int:
		episode_return = 0

		s, info = env.reset()
		done = False
		while not done:
			a = self.policy(s)
			s_, r, terminate, trunc, _ = env.step(a)
			done = terminate or trunc

			replay_buffer.append((s, a, r, s_, done))
			if len(replay_buffer) > replay_buffer.min_size:
				b_s, b_a, b_r, b_s_, b_done = replay_buffer.sample(batch_size)
				self.update_qnet(b_s, b_a, b_r, b_s_, b_done)

			episode_return += r
			s = s_

		return episode_return


def run_plot_dqn(env, replay_buffer, model, *, batch_size, num_episode, num_iter=10, name=None):
	returns = []

	for i in range(num_iter):
		num_e = int(num_episode / num_iter)
		with tqdm(total=num_e, desc=f'Iteration {i}: ') as pbar:
			for e in range(num_e):
				e_return = model.train_episode(env, replay_buffer, batch_size)
				returns.append(e_return)

				if (e + 1) % 10 == 0:
					pbar.set_postfix({'episode': e + 1 + i * num_e, 'return': np.mean(returns[-10:])})
				pbar.update(1)

	plt.plot(range(len(returns)), returns)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.savefig(f'result_{name}.png')
	plt.show()


# def test_dqn():
if __name__ == '__main__':
	SEED = 0

	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)

	env = gym.make('CartPole-v0')
	qnet = Qnet(
		num_state=env.observation_space.shape[0],
		hidden_dims=[128],
		action_dim=env.action_space.n,
		device='cuda' if th.cuda.is_available() else 'cpu',
	)
	dqn = DQN(
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
		dqn,
		batch_size=64,
		num_episode=500,
		name='dqn'
	)
