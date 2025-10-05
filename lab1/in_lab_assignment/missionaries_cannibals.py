from collections import deque

def is_valid_state(state):
    """Checks if a state is valid according to the rules."""
    m_left, c_left, boat_pos = state
    
    # Check for invalid numbers of people (less than 0 or more than 3)
    if not (0 <= m_left <= 3 and 0 <= c_left <= 3):
        return False
        
    m_right = 3 - m_left
    c_right = 3 - c_left
    
    # Check if cannibals outnumber missionaries on the left bank
    if m_left > 0 and m_left < c_left:
        return False
        
    # Check if cannibals outnumber missionaries on the right bank
    if m_right > 0 and m_right < c_right:
        return False
        
    return True

def get_successors(current_state):
    """Generates all valid successor states from the current state."""
    successors = []
    m_left, c_left, boat_pos = current_state
    
    # Possible moves are (missionaries_to_move, cannibals_to_move)
    possible_moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
    
    for m_move, c_move in possible_moves:
        if boat_pos == 1:  # Boat is on the left, moves to the right
            next_state = (m_left - m_move, c_left - c_move, 0)
        else:  # Boat is on the right, moves to the left
            next_state = (m_left + m_move, c_left + c_move, 1)
            
        if is_valid_state(next_state):
            successors.append(next_state)
            
    return successors

def breadth_first_search(initial_state, goal_state):
    """Performs a breadth-first search to find the shortest path."""
    # The frontier holds states to be explored. For BFS, it's a queue.
    frontier = deque([(initial_state, [])]) 
    explored = set()
    
    while frontier:
        current_state, path = frontier.popleft()
        
        if current_state in explored:
            continue
            
        explored.add(current_state)
        new_path = path + [current_state]
        
        if current_state == goal_state:
            return new_path  # Solution found
            
        for successor_state in get_successors(current_state):
            if successor_state not in explored:
                frontier.append((successor_state, new_path))
                
    return None # No solution found

# --- Main Execution ---
initial_state = (3, 3, 1)
goal_state = (0, 0, 0)

solution_path = breadth_first_search(initial_state, goal_state)

if solution_path:
    print("Solution found:")
    for i, state_in_path in enumerate(solution_path):
        print(f"Step {i}: {state_in_path}")
else:
    print("No solution found.")