import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

if not os.path.exists('results'):
    os.makedirs('results')

def plot_chessboard(solution_matrix):
    """Custom function to plot a chessboard with rooks"""
    N = solution_matrix.shape[0]
    
    chessboard = np.zeros((N, N))
    chessboard[1::2, 0::2] = 1
    chessboard[0::2, 1::2] = 1
    
    cmap_board = mcolors.ListedColormap(['#f0f0f0', '#c0c0c0'])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(chessboard, cmap=cmap_board, origin='upper')
    
    rook_rows, rook_cols = np.where(solution_matrix == 1)
    ax.scatter(rook_cols, rook_rows, s=500, c='crimson', marker='o', edgecolors='black', linewidth=2, label='Rook')
    
    for r, c in zip(rook_rows, rook_cols):
        ax.text(c, r, 'R', color='white', ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_title(f"8-Rook Solution (Count: {np.sum(solution_matrix)})", fontsize=14, pad=15)
    ax.set_xticks(np.arange(N)); ax.set_yticks(np.arange(N))
    ax.set_xticklabels(range(1, N+1)); ax.set_yticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0) 
    
    save_path = 'results/part2_eight_rooks_solution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()

def solve_eight_rooks():
    print("Initializing 8-Rook Solver...")
    N = 8
    
    print("\n--- SUBMISSION ANSWERS ---")
    print("1. ENERGY FUNCTION: E = A/2*(Row) + B/2*(Col) + C/2*(Global)")
    print("2. REASON FOR WEIGHTS: Inhibitory (-ve) for Row/Col to prevent conflicts.")
    print("   Global term ensures exactly 8 rooks.")

    A, B, C = 5.0, 5.0, 2.0
    u = np.random.uniform(-0.1, 0.1, (N, N))
    lr = 0.01
    
    print("Optimizing... (This takes a few seconds)")
    for _ in range(2000):
        V = 0.5 * (1 + np.tanh(u))
        total_active = np.sum(V)
        grad = np.zeros((N, N))
        
        for x in range(N):
            for y in range(N):
                row_sum = np.sum(V[x, :]) - V[x, y]
                col_sum = np.sum(V[:, y]) - V[x, y]
                global_diff = total_active - N
                grad[x, y] = - (A*row_sum + B*col_sum + C*global_diff)
        u += lr * grad

    final_V = 0.5 * (1 + np.tanh(u))
    board = (final_V > 0.8).astype(int)
    
    print(f"Final Board Configuration:\n{board}")
    plot_chessboard(board)

if __name__ == "__main__":
    solve_eight_rooks()