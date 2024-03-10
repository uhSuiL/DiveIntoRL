import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
	"""the prob of giving reward is unknown -> to be estimated while choosing action in specific steps"""
	def __init__(self, num_arm: int):
		self.num_arm = num_arm
		self.probs = np.random.uniform(0, 1, num_arm)

		self.best_bandit = np.argmax(self.probs)
		self.best_prob = self.probs[self.best_bandit]

		self.best_expected_reward = self.best_prob

	def pull_arm(self, k: int):
		"""get reward either 1 or 0"""
		p = np.random.rand()
		return 1 if p < self.probs[k] else 0


def test_bernoulli_bandit():
	np.random.seed(10)
	arm10_bandit = BernoulliBandit(10)
	print(f"\n\nBest bandit of arm10_bandit: {arm10_bandit.best_bandit}[p={arm10_bandit.best_prob}]")
	assert 0 < arm10_bandit.best_prob < 1


class Solver:
	"""Given a bandit, execute a good strategy for maximizing the cumulative reward in specific steps"""
	def __init__(self, bandit, init_estimated_q: float = 1):
		self.name = self.__class__.__name__
		self.bandit = bandit

		self.action_counter = np.zeros(bandit.num_arm)
		self.estimated_q_reward = np.ones(bandit.num_arm) * init_estimated_q
		self.cumulative_regret = 0

		self.actions = []
		self.cumulative_regrets = []

	def step(self, arm_id):
		reward_t = self.bandit.pull_arm(arm_id)

		self.action_counter[arm_id] += 1
		self.estimated_q_reward[arm_id] += (reward_t - self.estimated_q_reward[arm_id]) / self.action_counter[arm_id]
		self.cumulative_regret += self.bandit.best_expected_reward - reward_t

		self.actions.append(arm_id)
		self.cumulative_regrets.append(self.cumulative_regret)

	def policy(self) -> int:
		raise NotImplementedError

	def execute(self, num_step):
		for i in range(num_step):
			arm_id = self.policy()
			self.step(arm_id)
		return self


class EpsilonGreedy(Solver):
	def __init__(self, bandit, epsilon: float):
		super().__init__(bandit)
		assert 0 < epsilon < 1
		self.e = epsilon
		self.name = f'{self.__class__.__name__}[e={self.e}]'

	def policy(self) -> int:
		p = np.random.rand()
		return np.argmax(self.estimated_q_reward) if p > self.e else np.random.choice(self.bandit.num_arm)


def plot_result(solvers: list[Solver]):
	for solver in solvers:
		plt.plot(solver.cumulative_regrets, label=solver.name)
	plt.legend()
	plt.xlabel("Time step")
	plt.ylabel("Cumulative regrets")
	plt.show()
	plt.savefig(f'./plot_result.png')


def test_ep_g1(num_arm = 10):
	bandit = BernoulliBandit(num_arm)
	e_greedy = EpsilonGreedy(bandit, epsilon=0.4) \
		.execute(num_step=5000)
	plot_result(solvers=[e_greedy])


def test_ep_g2(num_arm = 10):
	bandit = BernoulliBandit(num_arm)
	plot_result(solvers=[
		EpsilonGreedy(bandit, epsilon=e).execute(num_step=5000)
		for e in [0.1, 0.3, 0.5, 0.6, 0.9]
	])


class DecayEpsilonGreedy(Solver):
	def policy(self) -> int:
		epsilon = 1 / (len(self.actions) + 1)
		p = np.random.rand()
		return np.argmax(self.estimated_q_reward) if p > epsilon else np.random.choice(self.bandit.num_arm)


def test_dgp_g(num_arm = 10):
	np.random.seed(1)
	bandit = BernoulliBandit(num_arm)
	plot_result(solvers=[
		DecayEpsilonGreedy(bandit).execute(num_step=5000)
	])


class UCB(Solver):
	def __init__(self, bandit, c):
		super().__init__(bandit)
		self.c = c
		self.name = f'{self.__class__.__name__}[c={c}]'

	def policy(self) -> int:
		print("Run policy")
		p = 1 / self.action_counter
		u = np.sqrt(
			np.log(p) / (-2 * self.action_counter)
		)
		q_upper_bounds = self.estimated_q_reward + self.c * u
		return np.argmax(q_upper_bounds)


def test_ucb(num_arm=10):
	bandit = BernoulliBandit(num_arm)
	plot_result(solvers=[
		UCB(bandit, c=c).execute(num_step=5000)
		for c in [.1, .3, .5, .7, .9]
	])


class ThompsonSampling(Solver):
	def __init__(self, bandit):
		super().__init__(bandit)
		self._a = np.ones(bandit.num_arm)
		self._b = np.ones(bandit.num_arm)

	def policy(self) -> int:
		estimated_expected_rewards = np.random.beta(self._a, self._b)
		return np.argmax(estimated_expected_rewards)

	def step(self, arm_id):
		reward_t = self.bandit.pull_arm(arm_id)

		self._a[arm_id] += reward_t
		self._b[arm_id] += 1 - reward_t

		self.action_counter[arm_id] += 1
		self.estimated_q_reward[arm_id] += (reward_t - self.estimated_q_reward[arm_id]) / self.action_counter[arm_id]
		self.cumulative_regret += self.bandit.best_expected_reward - reward_t

		self.actions.append(arm_id)
		self.cumulative_regrets.append(self.cumulative_regret)


def test_th_s(num_arm = 10):
	bandit = BernoulliBandit(num_arm)
	plot_result(solvers=[
		ThompsonSampling(bandit).execute(num_step=5000)
	])