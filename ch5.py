import numpy as np


class CliffWalkingEnv:
	def __init__(self, ncol, nrow, start_pos = None, cliff_pos = None, end_pos = None):
		self.ncol = ncol
		self.nrow = nrow
		self.pos = (0, 0) if start_pos is None else start_pos  # 默认起点设置在左下角原点处
		self.cliff_pos = [(x, 0) for x in range(1, ncol - 1)] if cliff_pos is None else cliff_pos  # 默认悬崖设在下底边
		self.end_pos = (ncol - 1, 0) if end_pos is None else end_pos  # 默认终点在右下角

		self.action_space = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
		self.done = False

	def step(self, action) -> (tuple, float):
		"""change the state(pos) with respect to given action, return [pos, reward, done]"""
		assert not self.done, "Episode have already been done"
		move = self.action_space[action]

		new_pos = tuple(np.add(self.pos, move))
		if not (0 <= new_pos[0] < self.ncol and 0 <= new_pos[1] < self.nrow):
			print(f"Warning: pos {new_pos} out of space {self.ncol -1, self.nrow - 1}")
			return self.pos, -1  # 如果动作导致越界, 则位置无变化

		self.pos = new_pos
		if self.pos in self.cliff_pos:
			self.done = True  # 如果走入悬崖, 则游戏结束并得到奖励-100
			return self.pos, -100

		if self.pos == self.end_pos:
			self.done = True  # 如果达到终点, 则游戏结束并得到奖励-1
			return self.pos, -1

		return self.pos, -1  # 没有走入悬崖或终点, 则游戏继续并得到奖励-1

	def reset(self, start_pos = None):
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


