
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import time
import os
from collections import defaultdict

# --------------------------
# Problem constants
# --------------------------
MAX_BIKES = 20
MAX_MOVE = 5
RENT_REWARD = 10
MOVE_COST = 2
PARKING_OVERFLOW_COST = 4
GAMMA = 0.9

FREE_FIRST_A_TO_B = True  # employee shuttle: first bike from A->B is free

# Poisson parameters
RENT_A = 3
RENT_B = 4
RET_A = 3
RET_B = 2

POISSON_EPS = 1e-3

# Convergence
THETA = 1e-3
MAX_EVAL = 200
MAX_ITER = 50

OUT_DIR = "gbike_problem2_outputs"
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
        if 1.0 - cum < POISSON_EPS:
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
    """apply overnight move, clip to [0,20]"""
    a, b = state
    a2 = a - action
    b2 = b + action
    a2 = min(max(a2, 0), MAX_BIKES)
    b2 = min(max(b2, 0), MAX_BIKES)
    return (a2, b2)

def move_cost(action):
    if action > 0:
        # A -> B
        if FREE_FIRST_A_TO_B:
            extra = max(0, action - 1)
            return -MOVE_COST * extra
        else:
            return -MOVE_COST * action
    else:
        # B -> A
        return -MOVE_COST * abs(action)

# ---------------------------------------------------------
# Precompute transitions
# ---------------------------------------------------------
print("Precomputing transitions...")
transitions = {}

start = time.time()

for a in range(MAX_BIKES + 1):
    for b in range(MAX_BIKES + 1):
        s = (a, b)
        transitions[s] = {}

        max_A_to_B = min(a, MAX_MOVE)
        max_B_to_A = min(b, MAX_MOVE)
        for action in range(-max_B_to_A, max_A_to_B + 1):
            a2, b2 = apply_action(s, action)

            # compute movement + parking cost
            cost = move_cost(action)
            if a2 > 10:
                cost -= PARKING_OVERFLOW_COST
            if b2 > 10:
                cost -= PARKING_OVERFLOW_COST

            next_states = defaultdict(lambda: [0.0, 0.0])  # prob, expected rental reward

            for rA, p_rA in pmf_rA.items():
                for rB, p_rB in pmf_rB.items():
                    for retA, p_retA in pmf_retA.items():
                        for retB, p_retB in pmf_retB.items():
                            prob = p_rA * p_rB * p_retA * p_retB

                            rentedA = min(a2, rA)
                            rentedB = min(b2, rB)

                            reward_rental = (rentedA + rentedB) * RENT_REWARD

                            a_next = min(max(a2 - rentedA + retA, 0), MAX_BIKES)
                            b_next = min(max(b2 - rentedB + retB, 0), MAX_BIKES)

                            next_states[(a_next, b_next)][0] += prob
                            next_states[(a_next, b_next)][1] += prob * reward_rental

            # Store full transition map
            transitions[s][action] = {
                "move_parking_cost": cost,
                "next": next_states
            }

end = time.time()
print(f"Done precomputing. Time: {end - start:.2f} sec")

# ---------------------------------------------------------
# Policy iteration
# ---------------------------------------------------------
V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)

def policy_evaluation(pi):
    global V
    for _ in range(MAX_EVAL):
        delta = 0
        new_V = V.copy()
        for a in range(MAX_BIKES + 1):
            for b in range(MAX_BIKES + 1):

                s = (a, b)
                act = pi[a, b]
                data = transitions[s][act]

                val = data["move_parking_cost"]
                for s2, (prob, rent_reward) in data["next"].items():
                    val += prob * (rent_reward + GAMMA * V[s2])

                new_V[a, b] = val
                delta = max(delta, abs(val - V[a, b]))

        V = new_V
        if delta < THETA:
            break

def policy_improvement():
    stable = True
    for a in range(MAX_BIKES + 1):
        for b in range(MAX_BIKES + 1):
            s = (a, b)

            max_A_to_B = min(a, MAX_MOVE)
            max_B_to_A = min(b, MAX_MOVE)
            best_act = None
            best_val = -1e18

            for act in range(-max_B_to_A, max_A_to_B + 1):

                data = transitions[s][act]
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
    print(f"Policy Iteration {it+1}")
    policy_evaluation(policy)
    stable = policy_improvement()

    # Save intermediate results
    sns.heatmap(V, cmap="viridis")
    plt.title(f"Value Function Iter {it+1}")
    plt.savefig(f"{OUT_DIR}/value_{it+1}.png")
    plt.close()

    sns.heatmap(policy, cmap="coolwarm", annot=False)
    plt.title(f"Policy Iter {it+1}")
    plt.savefig(f"{OUT_DIR}/policy_{it+1}.png")
    plt.close()

    if stable:
        print("Policy stable. Converged.")
        break

print("Finished Problem 2 Gbike Policy Iteration.")
