# Markov Reward Process (S, P, R, gamma)

import numpy as np

np.random.seed(0)


def get_return(start_id, state_sq, gamma, rewards):
    """
    给定状态序列state_sq, 折扣因子gamma, 根据奖励函数rewards,
    返回第start_id的state的回报g
    """
    g = 0
    for state_id in reversed(range(start_id, len(state_sq))):
        g += gamma * rewards[state_id]
    return g


def get_values(P: np.ndarray, r: np.ndarray, gamma):
    """
    给定状态转移矩阵P, 奖励函数r, 折扣因子gamma
    返回所有状态的价值v

    求解Bellman Equation
        v = (I - gamma * P)**(-1) * r
    """
    r = r.reshape((-1, 1))  # 转化为列向量

    num_state = r.shape[0]
    I = np.eye(num_state)

    return np.linalg.inv(
        I - gamma * P
    ) @ r


# 状态转移矩阵 P
P = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])

# 奖励函数 r(s)
rewards = np.array([-1, -2, -2, 10, 1, 0])

# 折扣因子 gamma
GAMMA = 0.5

# 求解状态价值
V = get_values(P, rewards, GAMMA)

print("Values: \n", V)
