
from copy import deepcopy

# --- Environment definition  ---
ROWS = 3
COLS = 4

wall = (1, 1)                    # forbidden cell
terminal_states = {(1, 3), (2, 3)}  # (i,j) set
terminal_rewards = {(2, 3): 1.0, (1, 3): -1.0}

# Actions
ACTIONS = ['L', 'R', 'U', 'D']
ACTION_TO_DELTA = {
    'L': (0, -1),
    'R': (0, +1),
    'U': (+1, 0),
    'D': (-1, 0)
}

# When action fails, it goes to perpendicular directions:
LEFT_OF = {'L': 'D', 'R': 'U', 'U': 'L', 'D': 'R'}
RIGHT_OF = {'L': 'U', 'R': 'D', 'U': 'R', 'D': 'L'}

# Transition probabilities
P_INTENDED = 0.8
P_LEFT = 0.1
P_RIGHT = 0.1

# Helper functions
def in_grid(i, j):
    return 0 <= i < ROWS and 0 <= j < COLS

def is_valid(i, j):
    return in_grid(i, j) and (i, j) != wall

def move_from(i, j, action):
    """Return next state (i2,j2) if we attempt action from (i,j).
       If the move would hit wall/outside/forbidden, agent stays in (i,j)."""
    di, dj = ACTION_TO_DELTA[action]
    ni, nj = i + di, j + dj
    if not in_grid(ni, nj) or (ni, nj) == wall:
        return (i, j)
    return (ni, nj)

def get_transitions(i, j, action):
    """Return dict mapping s' -> probability for executing 'action' in state (i,j)."""
    transitions = {}
    # intended
    s_int = move_from(i, j, action)
    transitions[s_int] = transitions.get(s_int, 0.0) + P_INTENDED
    # left (perpendicular)
    s_left = move_from(i, j, LEFT_OF[action])
    transitions[s_left] = transitions.get(s_left, 0.0) + P_LEFT
    # right (perpendicular)
    s_right = move_from(i, j, RIGHT_OF[action])
    transitions[s_right] = transitions.get(s_right, 0.0) + P_RIGHT
    return transitions

def pretty_print_values(V):
    """Print grid from top row to bottom row (like the AIMA convention)."""
    for i in range(ROWS - 1, -1, -1):
        line = ""
        for j in range(COLS):
            if (i, j) == wall:
                line += "  WALL   "
            elif (i, j) in terminal_states:
                line += f"{V[i][j]:7.2f} "
            else:
                line += f"{V[i][j]:7.2f} "
        print(line)
    print()

def pretty_print_policy(policy):
    """Print policy arrows for non-terminal, non-wall states."""
    arrow = {'L': '<', 'R': '>', 'U': '^', 'D': 'v', None: ' '}
    for i in range(ROWS - 1, -1, -1):
        line = ""
        for j in range(COLS):
            if (i, j) == wall:
                line += "  WALL   "
            elif (i, j) in terminal_states:
                line += "  TERM   "
            else:
                a = policy[i][j]
                line += f"   {arrow[a]}    "
        print(line)
    print()

# Value iteration implementation
def value_iteration(reward_per_step, gamma=1.0, epsilon=1e-8, max_iters=10000):
    """Run value iteration and return V, policy, iterations"""
    # reward_per_step: immediate reward for moving to any non-terminal state
    # Build reward matrix (reward when *entering* state s')
    R = [[reward_per_step for _ in range(COLS)] for _ in range(ROWS)]
    for (ti, tj), tr in terminal_rewards.items():
        R[ti][tj] = tr

    # Initialize V (terminal states fixed to their rewards)
    V = [[0.0 for _ in range(COLS)] for _ in range(ROWS)]
    for (ti, tj), tr in terminal_rewards.items():
        V[ti][tj] = tr

    iters = 0
    while True:
        delta = 0.0
        iters += 1
        V_new = deepcopy(V)
        for i in range(ROWS):
            for j in range(COLS):
                if (i, j) == wall or (i, j) in terminal_states:
                    continue
                best_a_value = float('-inf')
                # evaluate each action
                for a in ACTIONS:
                    transitions = get_transitions(i, j, a)
                    exp_value = 0.0
                    for (si, sj), prob in transitions.items():
                        # reward for moving *into* (si,sj)
                        r = R[si][sj]
                        exp_value += prob * (r + gamma * V[si][sj])
                    if exp_value > best_a_value:
                        best_a_value = exp_value
                V_new[i][j] = best_a_value
                delta = max(delta, abs(V_new[i][j] - V[i][j]))
        V = V_new
        if delta < epsilon or iters >= max_iters:
            break

    # extract greedy policy
    policy = [[None for _ in range(COLS)] for _ in range(ROWS)]
    for i in range(ROWS):
        for j in range(COLS):
            if (i, j) == wall or (i, j) in terminal_states:
                policy[i][j] = None
                continue
            best_a = None
            best_val = float('-inf')
            for a in ACTIONS:
                transitions = get_transitions(i, j, a)
                exp_value = 0.0
                for (si, sj), prob in transitions.items():
                    r = R[si][sj]
                    exp_value += prob * (r + gamma * V[si][sj])
                if exp_value > best_val:
                    best_val = exp_value
                    best_a = a
            policy[i][j] = best_a

    return V, policy, iters


if __name__ == "__main__":
    rewards = [-0.04, -2.0, 0.1, 0.02, 1.0]
    gamma = 1.0          
    epsilon = 1e-8

    for r in rewards:
        print(f"--- For step reward r(s) = {r}  (gamma={gamma}) ---\n")
        V, policy, iters = value_iteration(reward_per_step=r, gamma=gamma, epsilon=epsilon)
        print(f"Converged in {iters} iterations.\nValue function (rows top->bottom):")
        pretty_print_values(V)
        print("Greedy policy ( ^ = up, v = down, < = left, > = right )")
        pretty_print_policy(policy)
        print("\n\n")
