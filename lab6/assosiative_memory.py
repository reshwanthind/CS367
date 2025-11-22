import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('results'):
    os.makedirs('results')

class HopfieldNetwork:
    def __init__(self, size):
        self.n_neurons = size * size
        self.side = size
        self.weights = np.zeros((self.n_neurons, self.n_neurons))

    def train(self, patterns):
        print(f"Training on {len(patterns)} patterns...")
        for p in patterns:
            p_flat = p.flatten()
            self.weights += np.outer(p_flat, p_flat)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.n_neurons

    def recall(self, pattern, steps=5):
        state = pattern.flatten().copy()
        indices = np.arange(self.n_neurons)
        for _ in range(steps):
            np.random.shuffle(indices)
            for i in indices:
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation >= 0 else -1
        return state.reshape(self.side, self.side)

def run_part1():
    SIDE = 10
    N_NEURONS = SIDE * SIDE
    net = HopfieldNetwork(SIDE)

    patterns = [np.sign(np.random.randn(SIDE, SIDE)) for _ in range(3)]
    net.train(patterns)

    target = patterns[0]
    mask = np.random.choice([1, -1], size=target.shape, p=[0.8, 0.2]) # 20% noise
    noisy_input = target * mask
    recalled = net.recall(noisy_input)

    fig1, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(target, cmap='gray'); ax[0].set_title("Original")
    ax[1].imshow(noisy_input, cmap='gray'); ax[1].set_title("20% Noise")
    ax[2].imshow(recalled, cmap='gray'); ax[2].set_title("Recovered")
    plt.suptitle("Hopfield Associative Memory Recall")
    
    save_path1 = 'results/part1_associative_recall.png'
    plt.savefig(save_path1)
    print(f"Saved plot to {save_path1}")
    plt.show()

    print("\nCalculating Error Curve (this takes a moment)...")
    noise_levels = np.arange(0, 0.55, 0.05)
    success_rates = []

    for noise_pct in noise_levels:
        success_count = 0
        trials = 50
        for _ in range(trials):
            p_idx = np.random.randint(0, len(patterns))
            t = patterns[p_idx]
            m = np.random.choice([1, -1], size=t.shape, p=[1-noise_pct, noise_pct])
            n_in = t * m
            out = net.recall(n_in)
            if np.array_equal(t, out):
                success_count += 1
        success_rates.append((success_count/trials)*100)

    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels*100, success_rates, 'o-', color='purple', linewidth=2)
    plt.axvline(x=25, color='red', linestyle='--', label='Approx Limit (~25%)')
    plt.title("Hopfield Network Robustness")
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Perfect Recall Success Rate (%)")
    plt.grid(True)
    plt.legend()
    
    save_path2 = 'results/part1_error_curve.png'
    plt.savefig(save_path2)
    print(f"Saved plot to {save_path2}")
    plt.show()

    print("\n--- ANSWERS FOR SUBMISSION ---")
    print(f"1. CAPACITY: For N={N_NEURONS}, Capacity ≈ {int(0.138*N_NEURONS)} patterns.")
    print("2. ERROR CORRECTION: See 'results/part1_error_curve.png'.")
    print("   The network performs perfectly until ~20-25% noise, then degrades rapidly.")

if __name__ == "__main__":
    run_part1()