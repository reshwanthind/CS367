import numpy as np
import random
from utils import calculate_total_distance

def solve_tsp_sa(distance_matrix, initial_temp=10000, cooling_rate=0.999, min_temp=1e-3, iter_per_temp=None):
    """
    Solves the Traveling Salesman Problem using Simulated Annealing.
    
    Args:
        distance_matrix (np.array): A square matrix of distances between cities.
        initial_temp (float): The starting temperature.
        cooling_rate (float): The rate at which temperature decreases (e.g., 0.995).
        min_temp (float): The temperature at which to stop the algorithm.
        iter_per_temp (int): Number of iterations at each temperature step. 
                             Defaults to the number of cities.

    Returns:
        tuple: A tuple containing the best tour found (list of indices) and its total distance.
    """
    num_cities = distance_matrix.shape[0]
    
    if iter_per_temp is None:
        iter_per_temp = num_cities * 2

    # 1. Initialization
    temp = initial_temp
    current_tour = list(range(num_cities))
    random.shuffle(current_tour)
    current_cost = calculate_total_distance(current_tour, distance_matrix)
    
    best_tour = current_tour[:]
    best_cost = current_cost
    
    print(f"Initial random tour distance: {current_cost:.2f}")

    # 2. Main SA Loop
    while temp > min_temp:
        for _ in range(iter_per_temp):
            # 3. Generate a neighbor solution (2-opt swap)
            new_tour = current_tour[:]
            i, j = random.sample(range(num_cities), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            
            # 4. Calculate cost of the new tour
            new_cost = calculate_total_distance(new_tour, distance_matrix)
            
            # 5. Acceptance criteria
            cost_diff = new_cost - current_cost
            
            if cost_diff < 0 or random.uniform(0, 1) < np.exp(-cost_diff / temp):
                current_tour = new_tour
                current_cost = new_cost
                
                # Update best solution found so far
                if current_cost < best_cost:
                    best_tour = current_tour
                    best_cost = current_cost
        
        # 6. Cool down
        temp *= cooling_rate

    print(f"Finished. Best distance found: {best_cost:.2f}")
    return best_tour, best_cost