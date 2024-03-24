import random
from ch5 import CliffWalkingEnv, TemporalDifference, run_plot_td


class DynaQ(TemporalDifference):
	def __init__(self, n_planning, env, alpha, gamma, epsilon):
		super().__init__(env, alpha, gamma, epsilon)
		self.n_planning = n_planning

		self.model = dict()  # model[(s, a)] = r, s_next

	def learn_q(self, s, a, r, s_):
		self.Q_table[s][a] += self.alpha * (r + self.gamma * self.Q_table[s_].max() - self.Q_table[s][a])

	def plan_q(self):
		for i in range(self.n_planning):
			s, a = random.choice(list(self.model.keys()))
			r, s_next = self.model[(s, a)]
			self.learn_q(s, a, r, s_next)

	def update_q(self, s, a, r, s_next, a_next):
		self.learn_q(s, a, r, s_next)

		self.model[(s, a)] = r, s_next
		self.plan_q()


def test_dyna_q():
	cliff_env = CliffWalkingEnv(ncol=12, nrow=4)
	for n_planning in [0, 2, 20]:
		print(f'n_planning: {n_planning}')
		dyna_q = DynaQ(n_planning, cliff_env, alpha=0.1, gamma=0.9, epsilon=0.01)
		run_plot_td(dyna_q)
