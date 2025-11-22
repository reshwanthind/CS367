
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from collections import defaultdict
import os

# --------------------------
# Problem constants
# --------------------------
MAX_BIKES = 20
MAX_MOVE = 5
RENT_REWARD = 10
MOVE_COST = 2
GAMMA = 0.9

# No free shuttle
FREE_FIRST_A_TO_B = 0    

# No overflow fee
PARKING_OVERFLOW_COST = 0  

# Poisson parameters
RENT_A = 3
RENT_B = 4
RET_A = 3
RET_B = 2

POISSON_EPS = 1e-3
THETA = 1e-3
MAX_EVAL = 200
MAX_ITER = 50

OUT_DIR = "gbike_problem3_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Truncated Poisson
# ---------------------------------------------------------
def truncated_poisson(lmbda):
    dist = {}
    cum = 0
    n = 0
    while True:
        p = poisson.pmf(n, lmbda)
        dist[n] = p
        cum += p
        n += 1
        if 1 - cum < POISSON_EPS:
            break
        if n > 100:
            break
    s = sum(dist.values())
    for k in dist:
        dist[k] /= s
    return dist

pmf_rA = truncated_poisson(RENT_A)
pmf_rB = truncated_poisson(RENT_B)
pmf_retA = truncated_poisson(RET_A)
pmf_retB = truncated_poisson(RET_B)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def apply_action(state, action):
    a, b = state
    a2 = min(max(a - action, 0), MAX_BIKES)
    b2 = min(max(b + action, 0), MAX_BIKES)
    return (a2, b2)

def move_cost(action):
    return -MOVE_COST * abs(action)

# ---------------------------------------------------------
# Precompute transitions
# ---------------------------------------------------------
transitions = {}

for a in range(MAX_BIKES + 1):
    for b in range(MAX_BIKES + 1):
        s = (a, b)
        transitions[s] = {}

        max_A_to_B = min(a, MAX_MOVE)
        max_B_to_A = min(b, MAX_MOVE)

        for act in range(-max_B_to_A, max_A_to_B + 1):
            a2, b2 = apply_action(s, act)
            base_cost = move_cost(act)

            next_states = defaultdict(lambda: [0.0, 0.0])

            for rA, pA in pmf_rA.items():
                for rB, pB in pmf_rB.items():
                    for retA, pRA in pmf_retA.items():
                        for retB, pRB in pmf_retB.items():
                            prob = pA * pB * pRA * pRB

                            rentA = min(a2, rA)
                            rentB = min(b2, rB)

                            reward_rental = (rentA + rentB) * RENT_REWARD

                            a_next = min(MAX_BIKES, a2 - rentA + retA)
                            b_next = min(MAX_BIKES, b2 - rentB + retB)

                            next_states[(a_next, b_next)][0] += prob
                            next_states[(a_next, b_next)][1] += prob * reward_rental

            transitions[s][act] = {
                "move_parking_cost": base_cost,
                "next": next_states
            }

# ---------------------------------------------------------
# Policy iteration
# ---------------------------------------------------------
V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)

def policy_evaluation(pi):
    global V
    for _ in range(MAX_EVAL):
        delta = 0
        newV = V.copy()
        for a in range(MAX_BIKES + 1):
            for b in range(MAX_BIKES + 1):

                act = pi[a, b]
                data = transitions[(a, b)][act]

                val = data["move_parking_cost"]

                for s2, (prob, rent_reward) in data["next"].items():
                    val += prob * (rent_reward + GAMMA * V[s2])

                newV[a, b] = val
                delta = max(delta, abs(V[a, b] - val))

        V = newV
        if delta < THETA:
            break

def policy_improvement():
    stable = True
    for a in range(MAX_BIKES + 1):
        for b in range(MAX_BIKES + 1):

            best_act = 0
            best_val = -1e18

            max_A_to_B = min(a, MAX_MOVE)
            max_B_to_A = min(b, MAX_MOVE)

            for act in range(-max_B_to_A, max_A_to_B + 1):
                data = transitions[(a, b)][act]
                val = data["move_parking_cost"]

                for s2, (prob, rent_reward) in data["next"].items():
                    val += prob * (rent_reward + GAMMA * V[s2])

                if val > best_val:
                    best_val = val
                    best_act = act

            if policy[a, b] != best_act:
                stable = False
                policy[a, b] = best_act

    return stable

# Main loop
for it in range(MAX_ITER):
    print("Policy iteration", it + 1)
    policy_evaluation(policy)
    if policy_improvement():
        print("Policy stable — converged.")
        break

    sns.heatmap(V, cmap="viridis")
    plt.savefig(f"{OUT_DIR}/value_{it+1}.png")
    plt.close()

    sns.heatmap(policy, cmap="coolwarm", center=0)
    plt.savefig(f"{OUT_DIR}/policy_{it+1}.png")
    plt.close()

print("Finished Problem 3 Gbike Base Policy Iteration.")
