import random
import numpy as np
import torch as th
from torch import nn
from collections import deque


class ReplayBuffer(deque):
	def sample(self, batch_size: int):
		interaction_list = random.sample(self, batch_size)
		s_list, a_list, r_list, s_n_list, done_list = zip(*interaction_list)
		return np.array(s_list), a_list, r_list, np.array(s_n_list), done_list

	@property
	def size(self):
		return len(self)


class Qnet:
	def __init__(self, num_state: int, hidden: list, num_action: int, device: str):
		self.num_state = num_state
		self.hidden = hidden
		self.num_action = num_action
		self.device = device

	def __call__(self, *args, **kwargs) -> nn.Module:
		layers = [nn.Linear(self.num_state, self.hidden[0]), nn.ReLU()]
		for i in range(len(self.hidden) - 1):
			layers += [nn.Linear(self.hidden[i], self.hidden[i + 1]), nn.ReLU()]
		layers.append(nn.Linear(self.hidden[-1], self.num_action))

		return nn.Sequential(*layers).to(self.device)
