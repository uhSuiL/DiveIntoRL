import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class CliffWalkingEnv:
	def __init__(self, ncol, nrow, start_pos=None, cliff_pos=None, end_pos=None):
		self.ncol = ncol
		self.nrow = nrow
		self.pos = (0, 0) if start_pos is None else start_pos  # 默认起点设置在左下角原点处
		self.cliff_pos = [(x, 0) for x in range(1, ncol - 1)] if cliff_pos is None else cliff_pos  # 默认悬崖设在下底边
		self.end_pos = (ncol - 1, 0) if end_pos is None else end_pos  # 默认终点在右下角

		self.action_space = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
		self.state = self.pos
		self.state_space = [(x, y) for x in range(ncol) for y in range(nrow)]
		self.done = False

	def step(self, action) -> (tuple, float):
		"""change the state(pos) with respect to given action, return [pos, reward]"""
		assert not self.done, "Episode have already been done"
		move = self.action_space[action]

		new_pos = tuple(np.add(self.pos, move))
		if not (0 <= new_pos[0] < self.ncol and 0 <= new_pos[1] < self.nrow):
			# print(f"Warning: pos {new_pos} out of space {self.ncol -1, self.nrow - 1}")
			return self.pos, -1  # 如果动作导致越界, 则位置无变化

		self.pos = new_pos
		if self.pos in self.cliff_pos:
			self.done = True  # 如果走入悬崖, 则游戏结束并得到奖励-100
			return self.pos, -100

		if self.pos == self.end_pos:
			self.done = True  # 如果达到终点, 则游戏结束并得到奖励-1
			return self.pos, -1

		return self.pos, -1  # 没有走入悬崖或终点, 则游戏继续并得到奖励-1

	def reset(self, start_pos=None):
		self.pos = (0, 0) if start_pos is None else start_pos  # 默认起点设置在左下角原点处
		self.done = False
		return self


def test_cliff_env_opt_path(nrow=4, ncol=12):
	cliff_env = CliffWalkingEnv(ncol, nrow)
	print(cliff_env.cliff_pos)
	print(cliff_env.end_pos)

	print(cliff_env.step('up'), cliff_env.done)
	for i in range(ncol - 1):
		print(cliff_env.step('right'), cliff_env.done)
	print(cliff_env.step('down'), cliff_env.done)


def test_cliff_env_out_space(nrow=4, ncol=12):
	cliff_env = CliffWalkingEnv(ncol, nrow)
	for i in range(nrow):
		print(cliff_env.step('up'))


def test_cliff_env_cliff_space(nrow=4, ncol=12):
	cliff_env = CliffWalkingEnv(ncol, nrow)
	print(cliff_env.step('up'), cliff_env.done)
	print(cliff_env.step('left'), cliff_env.done)
	print(cliff_env.step('right'), cliff_env.done)
	print(cliff_env.step('right'), cliff_env.done)
	print(cliff_env.step('down'), cliff_env.done)


class TemporalDifference:
	def update_q(self, s, a, r, s_next, a_next):
		raise NotImplementedError

	def __init__(self, env, alpha, gamma, epsilon):
		self.env = env
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		self.Q_table = pd.DataFrame(
			np.zeros((len(env.action_space), len(env.state_space))),
			index=env.action_space.keys(),
			columns=env.state_space,
		)

	def policy(self, state):
		"""Epsilon Greedy"""
		p = np.random.uniform()
		return (
			self.Q_table.index[np.random.randint(len(self.env.action_space))] if p < self.epsilon
			else self.Q_table.index[np.argmax(self.Q_table[state])]
		)

	def train_episode(self) -> float | int:
		"""Play an episode to train the policy, update Q table"""
		episode_return = 0

		s = self.env.reset().state
		a = self.policy(s)
		while not self.env.done:
			s_next, r = self.env.step(a)
			a_next = self.policy(s_next)
			self.update_q(s, a, r, s_next, a_next)

			s = s_next
			a = a_next

			episode_return += r

		return episode_return


def run_plot_td(model: TemporalDifference, num_episode=500, num_iter=10):
	returns = []

	for i in range(num_iter):
		num_e = int(num_episode / num_iter)
		with tqdm(total=num_e, desc=f'Iteration {i}: ') as pbar:
			for e in range(num_e):
				e_return = model.train_episode()
				returns.append(e_return)

				if (e + 1) % 10 == 0:
					pbar.set_postfix({'episode': e + 1 + i * num_e, 'return': np.mean(returns[-10:])})
				pbar.update(1)

	plt.plot(range(len(returns)), returns)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.show()


class Sarsa(TemporalDifference):
	def update_q(self, s, a, r, s_next, a_next):
		td_error = r + self.gamma * self.Q_table[s_next][a_next] - self.Q_table[s][a]
		self.Q_table[s][a] += self.alpha * td_error


def test_sarsa():
	cliff_env = CliffWalkingEnv(ncol=12, nrow=4)
	sarsa_model = Sarsa(env=cliff_env, alpha=0.1, gamma=0.9, epsilon=0.1)
	run_plot_td(sarsa_model)


class SarsaNStep(Sarsa):
	def __init__(self, n_step, env, alpha, gamma, epsilon):
		super().__init__(env, alpha, gamma, epsilon)
		self.n_step = n_step

		self.states = []
		self.actions = []
		self.rewards = []

	def update_q(self, s, a, r, s_next, a_next):
		self.states.append(s)
		self.actions.append(a)
		self.rewards.append(r)

		if len(self.states) == self.n_step:
			G = sum((self.gamma ** t) * self.rewards[t] for t in range(self.n_step)) + self.Q_table[s_next][a_next]
			self.Q_table[self.states[0]][self.actions[0]] += self.alpha * (G - self.Q_table[self.states[0]][self.actions[0]])

			if self.env.done:  # 如果交互结束，则将暂存的所有s, a也更新，然后清空暂存列表
				for k in range(1, self.n_step):
					G = sum((self.gamma ** (t - k)) * self.rewards[t] for t in range(k, self.n_step)) + self.Q_table[s_next][a_next]
					self.Q_table[self.states[k]][self.actions[k]] += self.alpha * (G - self.Q_table[self.states[k]][self.actions[k]])

				self.states = []
				self.actions = []
				self.rewards = []
				return None

			self.states.pop(0)
			self.actions.pop(0)
			self.rewards.pop(0)


def test_sarsa_n_step():
	cliff_env = CliffWalkingEnv(ncol=12, nrow=4)
	sarsa_model = SarsaNStep(n_step=5, env=cliff_env, alpha=0.1, gamma=0.9, epsilon=0.1)
	run_plot_td(model=sarsa_model)


class QLearning(TemporalDifference):
	def update_q(self, s, a, r, s_next, a_next):
		self.Q_table[s][a] += self.alpha * (r + self.gamma * self.Q_table[s_next].max() - self.Q_table[s][a])


def test_q_learning():
	cliff_env = CliffWalkingEnv(ncol=12, nrow=4)
	q_learning = QLearning(cliff_env, alpha=0.1, gamma=0.9, epsilon=0.1)
	run_plot_td(q_learning)

