from collections import deque

def is_valid(current_state):
    empty_index = list(current_state).index(-1)
    if(current_state==(1,1,1,-1,0,0,0)):
        return True
    if(empty_index == 0 and current_state[empty_index+1]==0  and current_state[empty_index+2] == 0):
        return False
    if(empty_index == 6 and current_state[empty_index-1]==1 and current_state[empty_index-2] == 1):
        return False
    if(empty_index == 1 and current_state[empty_index-1]==1 and current_state[empty_index+1]==0 and current_state[empty_index+2]==0):
        return False
    if(empty_index == 5 and current_state[empty_index+1]==0 and current_state[empty_index-1]==1 and current_state[empty_index-2]==1):
        return False
    if(current_state[empty_index-1]==1 and current_state[empty_index-2]==1 and current_state[empty_index+1]==0 and current_state[empty_index+2]==0):
        return False
    return True

def swap_pieces(current_state, index_a, index_b):
    new_state_list = list(current_state)
    temp_piece = new_state_list[index_a]
    new_state_list[index_a] = new_state_list[index_b]
    new_state_list[index_b] = temp_piece
    return tuple(new_state_list)

def get_successors(current_state):
    successor_states = []
    empty_index = list(current_state).index(-1)
    move_offsets = [-2,-1,1,2]
    for offset in move_offsets:
        if(empty_index + offset >= 0 and empty_index + offset < 7):
            if(offset > 0 and list(current_state)[offset+empty_index]==1):
                new_state = swap_pieces(current_state, empty_index, empty_index + offset)
                if is_valid(new_state):
                    successor_states.append(new_state)
            if(offset < 0 and list(current_state)[offset+empty_index]==0):
                new_state = swap_pieces(current_state, empty_index, empty_index + offset)
                if(current_state==(1,1,1,0,-1,0,0)):
                    print(new_state)
                if is_valid(new_state):
                    successor_states.append(new_state)
    return successor_states

def breadth_first_search(initial_state, goal_state):
    frontier = deque([(initial_state, [])])
    explored_set = set()
    nodes_visited = 0
    max_frontier_size = 0
    while frontier:
        current_frontier_size = len(frontier)
        max_frontier_size = max(current_frontier_size, max_frontier_size)
        (current_state, current_path) = frontier.popleft()
        if current_state in explored_set:
            continue
        explored_set.add(current_state)
        new_path = current_path + [current_state]
        nodes_visited += 1
        if current_state == goal_state:
            print(f"Total Number Of Nodes Visited: {nodes_visited}")
            print(f"Max Size Of queue at a point was: {max_frontier_size}")
            return new_path
        for successor_state in get_successors(current_state):
            frontier.append((successor_state, new_path))
    return None

initial_state = (0,0,0,-1,1,1,1)
goal_state = (1,1,1,-1,0,0,0)

solution_path = breadth_first_search(initial_state, goal_state)
if solution_path:
    print("Solution found:")
    print(f"Number Of nodes in solution: {len(solution_path)}")
    for state_in_path in solution_path:
        print(state_in_path)
else:
    print("No solution found.")