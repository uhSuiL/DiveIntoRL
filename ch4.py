import copy
import numpy as np
from functools import cache


class CliffWalkingEnv:
	"""
	在一个(nrow, ncol)的方阵里:
		从一个格出发: 默认为左下角
		到一个格结束: 默认为右下角，到达后结束游戏
		每一步: 上下左右1格，奖励-1; 踩中悬崖，奖励-10，结束游戏; 撞墙，位置不变
		目标: 最大化奖励
	"""

	def __init__(self, nrow: int, ncol: int, cliff_pos: list[tuple], end_pos: tuple):
		self.action_space = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
		self.state_space = [(x, y) for x in range(ncol) for y in range(nrow)]
		self.nrow = nrow
		self.ncol = ncol
		self.cliffs = cliff_pos
		self.end_pos = end_pos

	@cache
	def __call__(self, state, action):
		"""return [(prob, next_state, reward, done)]"""
		move = self.action_space[action]
		# 如果原来位置在悬崖或终点，则无法继续交互，reward=0, done=True
		if state in self.cliffs or state == self.end_pos:
			return [(1, state, 0, True)]

		# 如果在边界且继续朝边界移动（移动后发生越界），则原地不动
		new_state = tuple(np.add(state, move))
		if not (0 <= new_state[0] < self.ncol and 0 <= new_state[1] < self.nrow):
			return [(1, state, -1, False)]

		return [(
			1,
			new_state,
			-1 if new_state not in self.cliffs else -100,
			new_state in self.cliffs or new_state == self.end_pos
		)]

	@cache
	def reward(self, state, action):
		return self(state, action)[0][2]

	@cache
	def P(self, next_state, state, action):
		actual_next_state = self(state, action)[0][1]
		return actual_next_state == next_state


def test_cliff_env():
	ncol = 12
	nrow = 4
	cliff_env = CliffWalkingEnv(
		ncol=ncol, nrow=nrow,
		cliff_pos=[(x, 0) for x in range(1, ncol)],
		end_pos=(ncol, 0)
	)
	print(cliff_env((0, 0), 'down'))


class PolicyIteration:
	# TODO: Seems to have bug since illegal results
	def __init__(self, env, diff_threshold, gamma):
		self.diff_threshold = diff_threshold
		self.gamma = gamma
		self.env = env

		# 初始化V^pi(s)和pi(s), for s in S
		self.V_pi, self.pi = dict(), dict()
		for s in env.state_space:
			self.V_pi[s] = 0
			self.pi[s] = {a: 1 / len(env.action_space) for a in env.action_space}

	def evaluate_policy(self):
		cnt = 0
		while True:
			diff = 0
			for s in self.env.state_space:
				new_V_pi_s = sum(
					self.pi[s][a] * (
						self.env.reward(s, a)
						+ self.gamma * sum(self.V_pi[s_] * self.env.P(s_, s, a) for s_ in self.env.state_space)
					)
					for a in self.env.action_space
				)
				diff = max(diff, abs(new_V_pi_s - self.V_pi[s]))
				self.V_pi[s] = new_V_pi_s
			cnt += 1
			if diff < self.diff_threshold:
				break
		print(f"complete evaluate_policy in {cnt} updates")
		return copy.deepcopy(self.pi)

	def improve_policy(self):
		Q = dict()
		for s in self.env.state_space:
			Q[s] = {
				a: self.env.reward(s, a) + self.gamma * sum(self.V_pi[s_] * self.env.P(s_, s, a) for s_ in self.env.state_space)
				for a in self.env.action_space
			}
			max_q_s = max(Q[s].values())
			num_max_q_s = sum(Q[s][a] == max_q_s for a in self.env.action_space)

			self.pi[s] = {
				a: 1 / num_max_q_s if Q[s][a] == max_q_s else 0
				for a in self.env.action_space
			}
		print(f"complete improve policy")

	def iterate_policy(self):
		print("start iterate")
		while True:
			old_pi = self.evaluate_policy()
			self.improve_policy()
			if old_pi == self.pi:
				break
		print("complete iterate")


def test_policy_iteration(ncol = 12, nrow = 4, theta = 0.001, gamma = 0.9):
	cliff_env = CliffWalkingEnv(
		ncol=ncol, nrow=nrow,
		cliff_pos=[(x, 0) for x in range(1, ncol)],
		end_pos=(ncol, 0)
	)
	policy = PolicyIteration(cliff_env, diff_threshold=theta, gamma=gamma)
	policy.iterate_policy()


class ValueIteration:
	# TODO: To be tested
	def __init__(self, env, gamma, diff_threshold):
		self.env = env
		self.gamma = gamma
		self.diff_threshold = diff_threshold

		# 初始化V(s) for s in state_space
		self.V = {s: 0 for s in env.state_space}
		self.pi = dict()

	def __call__(self):
		print("start value iteration")
		cnt = 0
		while True:
			diff = self.diff_threshold
			V_old = copy.deepcopy(self.V)
			for s in self.env.state_space:
				self.V[s] = max(
					self.env.reward(s, a) + self.gamma * sum(self.env.P(s_, s, a) * self.V[s_] for s_ in self.env.state_space)
					for a in self.env.action_space
				)
				diff = max(diff, abs(V_old[s] - self.V[s]))

			cnt += 1
			if not cnt % 20:
				print(f"{cnt} iteration, diff: {diff: .5f}")
			if diff <= self.diff_threshold:
				print(f"complete value iteration in {cnt} updates")
				break

		return self

	def get_policy(self) -> dict:
		Q = dict()
		for s in self.env.state_space:
			Q[s] = {
				a: self.env.reward(s, a) + self.gamma * sum(self.env.P(s_, s, a) * self.V[s_] for s_ in self.env.state_space)
				for a in self.env.action_space
			}
			max_q_sa = max(Q[s].values())
			num_max_q_sa = sum(Q[s][a] == max_q_sa for a in self.env.action_space)

			self.pi[s] = {
				a: 1 / num_max_q_sa if Q[s][a] == max_q_sa else 0
				for a in self.env.action_space
			}
		return self.pi


def test_value_iteration(ncol = 12, nrow = 4, theta = 0.001, gamma = 0.9):
	cliff_env = CliffWalkingEnv(
		ncol=ncol, nrow=nrow,
		cliff_pos=[(x, 0) for x in range(1, ncol)],
		end_pos=(ncol, 0)
	)
	value_iter = ValueIteration(cliff_env, diff_threshold=theta, gamma=gamma)
	value_iter().get_policy()
