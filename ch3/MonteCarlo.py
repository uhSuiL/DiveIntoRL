import numpy as np


def sample(
        MDP: tuple,
        policy: dict,
        terminate_state_id: int,
        num_episode: int,
        max_time_step: int = 20
) -> list[list[tuple]]:
    S, A, P, R, gamma = MDP
    episodes = []
    for i in range(num_episode):
        # 不允许从终止状态开始采样
        start_state_id = np.random.randint(len(S))
        while start_state_id == terminate_state_id:
            start_state_id = np.random.randint(len(S))
        # 开始采样
        state = S[start_state_id]
        terminate_state = S[terminate_state_id]
        t, episode = 0, []
        while state != terminate_state and t < max_time_step:
            t += 1

            # get action
            prob = np.random.rand()
            action, reward = None, None
            for a in A:
                prob -= policy.get((state, a), 0)
                if prob <= 0:
                    action = a
                    break

            reward = R[(state, action)]

            episode.append((state, action, reward))

            # get next state
            prob = np.random.rand()
            for s in S:
                prob -= P.get((s, action), 0)
                if prob <= 0:
                    state = s
                    break
        episodes.append(episode)
    return episodes


def monte_carlo(episodes: list[list[tuple]], gamma: float, S: list):
    V = {s: 0 for s in S}
    N = {s: 0 for s in S}

    for episode in episodes:
        G = 0
        for state, action, reward in episode:
            N[state] += 1
            V[state] += (G - V[state]) / N[state]
            G += gamma * reward
    return V


if __name__ == "__main__":
    from MDP import S, A, P, R, GAMMA, policy_1
    NUM_EPISODE = 1000
    episodes = sample((S, A, P, R, GAMMA), policy_1, 4, NUM_EPISODE)
    print("episodes\n", episodes)
    values = monte_carlo(episodes, GAMMA, S)
    print("values\n", values)
