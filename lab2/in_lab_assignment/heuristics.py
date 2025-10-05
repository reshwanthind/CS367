import numpy as np

def heuristic_misplaced(curr_state, goal_state):
    """
    Heuristic 1: Counts the number of misplaced tiles.
    An admissible heuristic since every misplaced tile must be moved at least once.
    """
    # The sum of a boolean array where True is 1 and False is 0
    return np.sum(curr_state != goal_state)


def heuristic_manhattan(curr_state, goal_state):
    """
    Heuristic 2: Calculates the Manhattan distance for each tile.
    For each tile, it sums the vertical and horizontal distance to its goal position.
    This is also admissible and generally more informed than the misplaced tiles heuristic.
    """
    dist = 0
    for i in range(3):
        for j in range(3):
            tile = curr_state[i, j]
            if tile != '_':
                # Find the coordinates of the tile in the goal state
                goal_pos = np.where(goal_state == tile)
                goal_i, goal_j = goal_pos[0][0], goal_pos[1][0]
                
                # Add the Manhattan distance
                dist += abs(i - goal_i) + abs(j - goal_j)
    return dist

def heuristic_zero(curr_state, goal_state):
    """
    Heuristic 0: A trivial heuristic that always returns 0.
    Using this heuristic turns the A* search into Uniform Cost Search (or Dijkstra's Algorithm).
    """
    return 0