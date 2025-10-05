import random
import math
import copy
from collections import deque
from utils import get_best_matching_patch, calculate_grid_cost

def greedy_bfs_fill(patches, start_patch_id):
    """Creates a greedy best-fit solution starting with a given patch."""
    grid_size = int(math.sqrt(len(patches)))
    grid = [[-1 for _ in range(grid_size)] for _ in range(grid_size)]
    
    available_patch_ids = set(patches.keys())
    
    grid[0][0] = start_patch_id
    available_patch_ids.remove(start_patch_id)
    
    queue = deque([(0, 0)])
    visited = set([(0, 0)])

    while queue:
        r, c = queue.popleft()
        parent_patch = patches[grid[r][c]]
        
        # Check neighbors (Down, Right, Up, Left)
        for dr, dc, direction, opposite_direction in [(1,0,'down','up'), (0,1,'right','left'), (-1,0,'up','down'), (0,-1,'left','right')]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in visited:
                available_patches_dict = {pid: patches[pid] for pid in available_patch_ids}
                
                best_patch_id = get_best_matching_patch(parent_patch, available_patches_dict, direction)
                
                if best_patch_id != -1:
                    grid[nr][nc] = best_patch_id
                    available_patch_ids.remove(best_patch_id)
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return grid

def simulated_annealing(initial_grid, patches, initial_temp=1000, final_temp=1, alpha=0.995):
    """Refines a grid solution using the Simulated Annealing algorithm."""
    current_grid = copy.deepcopy(initial_grid)
    best_grid = copy.deepcopy(initial_grid)
    
    current_cost = calculate_grid_cost(current_grid, patches)
    best_cost = current_cost
    
    temp = initial_temp
    
    while temp > final_temp:
        # Select two random cells to swap
        r1, c1 = random.randint(0, 3), random.randint(0, 3)
        r2, c2 = random.randint(0, 3), random.randint(0, 3)

        # Swap the patches
        current_grid[r1][c1], current_grid[r2][c2] = current_grid[r2][c2], current_grid[r1][c1]
        
        new_cost = calculate_grid_cost(current_grid, patches)
        
        # Decide whether to accept the new solution
        cost_delta = new_cost - current_cost
        if cost_delta < 0 or random.uniform(0, 1) < math.exp(-cost_delta / temp):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_grid = copy.deepcopy(current_grid)
        else:
            # Revert the swap if not accepted
            current_grid[r1][c1], current_grid[r2][c2] = current_grid[r2][c2], current_grid[r1][c1]

        temp *= alpha

    return best_grid, best_cost