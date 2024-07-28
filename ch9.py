import util
import random
import torch as th
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm


class Trajectory:
	def __init__(self):
		self.s, self.a, self.r, self.s_, self.done = [], [], [], [], []

	def __len__(self):
		assert len(self.s) == len(self.a) == len(self.r) == len(self.s_) == len(self.done)
		return len(self.s)

	@property
	def s_list(self):
		return np.array(self.s)

	def append(self, s, a, r, s_, done):
		self.s.append(s)
		self.a.append(a)
		self.r.append(r)
		self.s_.append(s_)
		self.done.append(done)


class PolicyNet(nn.Module):
	def __init__(self, state_dim: int, num_action: int, device: str, *, hidden_dims: list = None):
		super().__init__()
		hidden_dims = [] if hidden_dims is None else hidden_dims
		self.fc = util.MLP([state_dim, *hidden_dims, num_action])
		self.softmax = nn.Softmax(dim=-1)

		self.to(device)
		self.device = device

	def forward(self, state: th.tensor):
		return self.softmax(self.fc(state))


# @Agent
class REINFORCE:
	def __init__(self, policy_net: nn.Module, *, learning_rate, gamma, ):
		self.policy_net = policy_net
		self.optimizer = th.optim.Adam(policy_net.parameters(), lr=learning_rate)

		self.gam = gamma

	@th.no_grad()
	def policy(self, state):
		state = th.tensor(state).unsqueeze(dim=0).to(self.policy_net.device)
		action_probs = self.policy_net(state)
		action_dist = th.distributions.Categorical(action_probs)
		action = action_dist.sample().item()
		return action

	def update(self, tau: Trajectory):
		s_list = th.tensor(tau.s_list).to(self.policy_net.device)
		a_list = th.tensor(tau.a, dtype=th.long).to(self.policy_net.device)
		r_list = th.tensor(tau.r).to(self.policy_net.device)

		self.optimizer.zero_grad()
		G = 0
		objective = th.tensor([0.]).to(self.policy_net.device)
		for i in range(len(tau))[::-1]:
			s, a, r = s_list[i], a_list[i].unsqueeze(dim=-1), r_list[i]
			G = self.gam * G + r
			objective += G * th.log(self.policy_net(s).gather(dim=-1, index=a))
		loss = -objective
		loss.backward()
		self.optimizer.step()


def run_episode(env: gym.Env, agent) -> int:
	episode_return = 0
	traj = Trajectory()

	s, info = env.reset()
	done = False
	while not done:
		a = agent.policy(s)
		s_, r, terminate, trunc, _ = env.step(a)
		done = terminate or trunc

		episode_return += r
		traj.append(s, a, r, s_, done)

		s = s_

	agent.update(traj)
	return episode_return


def run_plot_pg(env, agent, *, num_episode, num_iter=10, name=None):
	returns = []

	for i in range(num_iter):
		num_e = int(num_episode / num_iter)
		with tqdm(total=num_e, desc=f'Iteration {i}: ') as pbar:
			for e in range(num_e):
				e_return = run_episode(env, agent)
				returns.append(e_return)

				if (e + 1) % 10 == 0:
					pbar.set_postfix({'episode': e + 1 + i * num_e, 'return': np.mean(returns[-10:])})
				pbar.update(1)

	plt.plot(range(len(returns)), returns)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.savefig(f'result_{name}.png')
	plt.show()


if __name__ == '__main__':
	SEED = 0
	random.seed(SEED)
	np.random.seed(SEED)
	th.manual_seed(SEED)

	env = gym.make("CartPole-v0")
	agent = REINFORCE(
		policy_net=PolicyNet(
			state_dim=env.observation_space.shape[0],
			num_action=env.action_space.n,
			hidden_dims=[128],
			device='cuda' if th.cuda.is_available() else 'cpu'
		),
		learning_rate=0.001,
		gamma=0.98
	)
	run_plot_pg(
		env, agent,
		num_episode=1000,
		num_iter=10,
		name='REINFORCE'
	)
