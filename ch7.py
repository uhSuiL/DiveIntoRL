import random
import numpy as np
import torch as th
from collections import deque


class ReplayBuffer(deque):
	def sample(self, batch_size: int):
		interaction_list = random.sample(self, batch_size)
		s_list, a_list, r_list, s_n_list, done_list = zip(*interaction_list)
		return np.array(s_list), a_list, r_list, np.array(s_n_list), done_list

	@property
	def size(self):
		return len(self)
