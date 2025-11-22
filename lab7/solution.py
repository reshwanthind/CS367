import numpy as np
import matplotlib.pyplot as plt
import random
import os


class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.winner = None

    def reset(self):
        self.board = [' '] * 9
        self.winner = None
        return self.get_state()

    def get_state(self):
        return "".join(self.board)

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, position, player):
        if self.board[position] == ' ' and self.winner is None:
            self.board[position] = player
            self.check_winner(player)
            return True
        return False

    def check_winner(self, player):
        wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] == player:
                self.winner = player
                return
        if ' ' not in self.board:
            self.winner = 'Draw'

class MenacePlayer:
    def __init__(self, player_symbol='X'):
        self.symbol = player_symbol
        self.matchboxes = {}
        self.history = [] 

    def get_beads_for_stage(self, empty_squares):
        moves_played = 9 - empty_squares
        
        if moves_played <= 0: return 4
        if moves_played <= 2: return 3
        if moves_played <= 4: return 2
        return 1

    def get_move(self, board_inst):
        state = board_inst.get_state()
        available = board_inst.available_moves()
        
        if state not in self.matchboxes:
            beads_per_move = self.get_beads_for_stage(len(available))
            self.matchboxes[state] = {move: beads_per_move for move in available}

        bead_box = self.matchboxes[state]
        
        valid_moves = [m for m in available if m in bead_box]
        
        total_beads = []
        for move in valid_moves:
            count = bead_box[move]
            total_beads.extend([move] * count)

        if not total_beads:
            move = random.choice(available)
        else:
            move = random.choice(total_beads)

        self.history.append((state, move))
        return move

    def train(self, result):
        
        if result == 1:
            delta = 3
        elif result == 0:
            delta = 1
        else:
            delta = -1

        for state, move in self.history:
            if state in self.matchboxes:
                current_beads = self.matchboxes[state].get(move, 0)
                new_count = max(0, current_beads + delta)
                self.matchboxes[state][move] = new_count
        
        self.history = []

def play_menace_game():
    print("\n--- Running MENACE Simulation (Part 1) ---")
    game = TicTacToe()
    menace = MenacePlayer('X')
    random_player = 'O'
    
    episodes = 2000
    wins = 0
    draws = 0
    losses = 0
    
    win_rate = []

    for i in range(episodes):
        game.reset()
        turn = 'X'
        while game.winner is None:
            if turn == 'X':
                move = menace.get_move(game)
                game.make_move(move, 'X')
                turn = 'O'
            else:
                avail = game.available_moves()
                if avail:
                    move = random.choice(avail)
                    game.make_move(move, 'O')
                turn = 'X'
        
        if game.winner == 'X':
            menace.train(1)
            wins += 1
        elif game.winner == 'Draw':
            menace.train(0)
            draws += 1
        else:
            menace.train(-1)
            losses += 1
            
        if (i+1) % 100 == 0:
            win_rate.append(wins / (i+1))
            
    print(f"Games Played: {episodes}")
    print(f"MENACE Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print("Crucial concept demonstrated: Matchbox state representation + Bead reinforcement.")
    
    if not os.path.exists('results'):
        os.makedirs('results')
        
    plt.figure(figsize=(10, 5))
    plt.plot(range(100, episodes + 1, 100), win_rate)
    plt.title("MENACE Win Rate vs Random Player")
    plt.xlabel("Games")
    plt.ylabel("Cumulative Win Rate")
    plt.grid(True)
    plt.savefig('results/menace_win_rate.png')
    print("Plot saved to results/menace_win_rate.png")
    plt.show()


class BinaryBandit:
    def __init__(self, p_success):
        self.p = p_success
    
    def pull(self):
        return 1 if random.random() < self.p else 0

def run_binary_bandit_experiment():
    print("\n--- Running Binary Bandit (Part 2) ---")
    bandit_A = BinaryBandit(p_success=0.3)
    bandit_B = BinaryBandit(p_success=0.7)
    bandits = [bandit_A, bandit_B]
    
    epsilon = 0.1
    steps = 1000
    Q = [0.0, 0.0] 
    N = [0, 0]     
    
    rewards = []
    
    for t in range(steps):
        if random.random() < epsilon:
            action = random.choice([0, 1]) 
        else:
            if Q[0] == Q[1]:
                action = random.choice([0, 1])
            else:
                action = np.argmax(Q)
        
        reward = bandits[action].pull()
        rewards.append(reward)
        
        N[action] += 1
        Q[action] = Q[action] + (1.0 / N[action]) * (reward - Q[action])
        
    print(f"Final Estimated Q-values: A={Q[0]:.2f} (True=0.3), B={Q[1]:.2f} (True=0.7)")
    print(f"Total Reward: {sum(rewards)}/{steps}")
    
    if not os.path.exists('results'):
        os.makedirs('results')

    plt.figure(figsize=(10, 4))
    plt.plot(np.cumsum(rewards) / (np.arange(steps) + 1))
    plt.title("Binary Bandit Average Reward over Time")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.savefig('results/binary_bandit.png')
    print("Plot saved to results/binary_bandit.png")
    plt.show()


class NonStationaryBandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.zeros(k)
        self.time = 0
        
    def step_evolution(self):
        self.q_true += np.random.normal(0, 0.01, self.k)
        
    def pull(self, action):
        self.step_evolution() 
        return np.random.normal(self.q_true[action], 1.0)

def bandit_nonstat_solver(steps=10000):
    print("\n--- Running Non-Stationary Bandit (Part 3 & 4) ---")
    
    env = NonStationaryBandit()
    
    epsilon = 0.1
    alpha = 0.1 
    
    Q_avg = np.zeros(10)
    N_avg = np.zeros(10)
    rewards_avg = []
    
    Q_fixed = np.zeros(10)
    rewards_fixed = []
    
    
    env1 = NonStationaryBandit()
    env2 = NonStationaryBandit() 
    
    for t in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(10)
        else:
            action = np.argmax(Q_avg)
            
        reward = env1.pull(action)
        rewards_avg.append(reward)
        
        N_avg[action] += 1
        Q_avg[action] += (1.0 / N_avg[action]) * (reward - Q_avg[action])

    for t in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(10)
        else:
            action = np.argmax(Q_fixed)
            
        reward = env2.pull(action)
        rewards_fixed.append(reward)
        
        Q_fixed[action] += alpha * (reward - Q_fixed[action])

    def moving_average(a, n=100):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    ma_avg = moving_average(rewards_avg)
    ma_fixed = moving_average(rewards_fixed)
    
    print(f"Agent 1 (Sample Avg) Avg Reward (last 1000): {np.mean(rewards_avg[-1000:]):.3f}")
    print(f"Agent 2 (Constant Alpha) Avg Reward (last 1000): {np.mean(rewards_fixed[-1000:]):.3f}")
    
    if not os.path.exists('results'):
        os.makedirs('results')

    plt.figure(figsize=(12, 6))
    plt.plot(ma_avg, label='Sample Average (1/N)', alpha=0.6)
    plt.plot(ma_fixed, label=f'Constant Step (alpha={alpha})', alpha=0.8)
    plt.title("Non-Stationary Bandit: Sample Average vs Constant Alpha")
    plt.xlabel("Steps")
    plt.ylabel("Reward (Moving Avg)")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/non_stationary_bandit.png')
    print("Plot saved to results/non_stationary_bandit.png")
    plt.show()

if __name__ == "__main__":
    play_menace_game()
    run_binary_bandit_experiment()
    bandit_nonstat_solver()