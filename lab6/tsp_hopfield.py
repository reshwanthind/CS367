import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('results'):
    os.makedirs('results')

def solve_tsp():
    num_cities = 10
    print(f"Setting up TSP for {num_cities} cities...")
    
    print("\n--- SUBMISSION ANSWERS ---")
    print(f"1. NETWORK SIZE: {num_cities}x{num_cities} grid.")
    print(f"2. WEIGHTS: Total Weights = (Neurons)^2 = {num_cities**2}^2 = 10,000 weights.")

    coords = np.random.rand(num_cities, 2)
    d = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            d[i][j] = np.linalg.norm(coords[i] - coords[j])
            
    # Optimization
    A, B, C, D = 500, 500, 250, 50 
    u = np.random.uniform(-0.02, 0.02, (num_cities, num_cities))
    lr = 0.0001
    
    print("Optimizing Network...")
    for _ in range(5000):
        V = 0.5 * (1 + np.tanh(u/0.5))
        total_sum = np.sum(V)
        grad = np.zeros_like(u)
        
        for x in range(num_cities):
            for i in range(num_cities):
                row_sum = np.sum(V[x, :]) - V[x, i]
                col_sum = np.sum(V[:, i]) - V[x, i]
                global_diff = total_sum - num_cities
                
                prev_i = (i - 1) % num_cities
                next_i = (i + 1) % num_cities
                dist_term = 0
                for y in range(num_cities):
                    dist_term += d[x][y] * (V[y, prev_i] + V[y, next_i])
                
                grad[x, i] = -A*row_sum - B*col_sum - C*global_diff - D*dist_term
        u += lr * grad

    # Decode
    final_V = 0.5 * (1 + np.tanh(u/0.5))
    tour = []
    available = set(range(num_cities))
    for i in range(num_cities):
        scores = final_V[:, i]
        best = -1
        best_val = -np.inf
        for city in available:
            if scores[city] > best_val:
                best_val = scores[city]
                best = city
        if best != -1:
            tour.append(best); available.remove(best)
        else:
            tour.append(list(available)[0]); available.pop()

    print("Tour found:", tour)
    
    # Plotting
    tour_coords = coords[tour]
    tour_coords = np.vstack([tour_coords, tour_coords[0]])
    
    plt.figure(figsize=(7, 7))
    plt.plot(tour_coords[:,0], tour_coords[:,1], color='royalblue', linewidth=2.5, zorder=1)
    plt.scatter(tour_coords[0,0], tour_coords[0,1], s=200, c='green', marker='s', edgecolors='black', zorder=2, label='Start')
    plt.scatter(tour_coords[1:-1,0], tour_coords[1:-1,1], s=150, c='crimson', edgecolors='black', zorder=2)
    
    for i, (x, y) in enumerate(coords):
        plt.text(x, y+0.02, f'C{i}', ha='center', fontsize=9, fontweight='bold')

    plt.title("TSP Hopfield Solution Map", fontsize=14, pad=15)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    
    # SAVE PLOT
    save_path = 'results/part3_tsp_map.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    solve_tsp()