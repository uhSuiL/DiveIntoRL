# Markov Decision Process (S, A, P, R, gamma)

import numpy as np

np.random.seed(0)

# ========================= Model Definition =========================

# 状态集 s5为终止状态
S = ["s1", "s2", "s3", "s4", "s5"]
# 动作集
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]
# 状态转移函数
P = {
    ("s1", "保持s1", "s1"): 1.0,
    ("s1", "前往s2", "s2"): 1.0,
    ("s2", "前往s1", "s1"): 1.0,
    ("s2", "前往s3", "s3"): 1.0,
    ("s3", "前往s4", "s4"): 1.0,
    ("s3", "前往s5", "s5"): 1.0,
    ("s4", "前往s5", "s5"): 1.0,
    ("s4", "概率前往", "s2"): 0.2,
    ("s4", "概率前往", "s3"): 0.4,
    ("s4", "概率前往", "s4"): 0.4,
}  # (state, action): next_state
# 奖励函数
R = {
    ("s1", "保持s1"): -1,
    ("s1", "前往s2"): 0,
    ("s2", "前往s1"): -1,
    ("s2", "前往s3"): -2,
    ("s3", "前往s4"): -2,
    ("s3", "前往s5"): 0,
    ("s4", "前往s5"): 10,
    ("s4", "概率前往"): 1,
}  # (state, action): reward

GAMMA = 0.5
MDP = (S, A, P, R, GAMMA)

policy_1 = {
    ("s1", "保持s1"): 0.5,
    ("s1", "前往s2"): 0.5,
    ("s2", "前往s1"): 0.5,
    ("s2", "前往s3"): 0.5,
    ("s3", "前往s4"): 0.5,
    ("s3", "前往s5"): 0.5,
    ("s4", "前往s5"): 0.5,
    ("s4", "概率前往"): 0.5,
}  # (state, action): probability

policy_2 = {
    ("s1", "保持s1"): 0.6,
    ("s1", "前往s2"): 0.4,
    ("s2", "前往s1"): 0.3,
    ("s2", "前往s3"): 0.7,
    ("s3", "前往s4"): 0.5,
    ("s3", "前往s5"): 0.5,
    ("s4", "前往s5"): 0.1,
    ("s4", "概率前往"): 0.9,
}  # (state, action): probability


# =============================== Util ===============================


def mdp2mrp(MDP: tuple, policy: dict) -> tuple:
    """将mdp问题转化为mrp问题--Marginalize
        对状态转移函数 P(s'|s, a) 关于a积分
        对奖励函数 R(s, a) 关于a积分
    """
    S, A, P, R, gamma = MDP

    R_new, P_new = [], []
    for s in S:
        r_s = 0
        for a in A:
            if (s, a) in R.keys() and (s, a) in policy.keys():
                r_s += R[(s, a)] * policy[(s, a)]
        R_new.append(r_s)

        p_s = []
        for s_next in S:
            p_s_s_next = 0
            for a in A:
                if (s, a, s_next) in P.keys() and (s, a) in policy.keys():
                    p_s_s_next += P[(s, a, s_next)] * policy[(s, a)]
            p_s.append(p_s_s_next)
        P_new.append(p_s)

    return S, np.array(P_new), np.array(R_new), gamma


if __name__ == "__main__":
    MRP = mdp2mrp(MDP, policy_1)
    print(MRP[1])
    print(MRP[2])

    # ================================ Solve ================================
    from MRP import get_values

    V = get_values(*MRP[1:])
    print("Values:\n", V)
