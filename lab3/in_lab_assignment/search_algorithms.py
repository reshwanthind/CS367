import heapq
import time
from board_utils import SolitaireNode, get_possible_moves, apply_move, is_goal

def solve(initial_state, heuristic_func, algorithm_type):
    """A general solver function for both Best-First and A*."""
    if algorithm_type not in ['best_first', 'a_star']:
        raise ValueError("Algorithm type must be 'best_first' or 'a_star'")

    start_time = time.time()
    start_node = SolitaireNode(initial_state, g=0, h=heuristic_func(initial_state))
    
    open_list = []
    priority = start_node.h if algorithm_type == 'best_first' else start_node.f
    # id(start_node) is used as a tie-breaker to handle nodes with the same priority
    heapq.heappush(open_list, (priority, id(start_node), start_node))
    
    visited = {tuple(map(tuple, start_node.state)): 0}
    
    nodes_expanded = 0
    max_queue_size = 0
    max_expansions = 5000000 

    while open_list:
        max_queue_size = max(max_queue_size, len(open_list))
        
        _, _, current_node = heapq.heappop(open_list)
        nodes_expanded += 1

        if nodes_expanded > max_expansions:
            print(f"Search limit of {max_expansions} expansions reached. No solution found.")
            return None

        if is_goal(current_node.state):
            path = []
            temp = current_node
            while temp:
                path.append(temp.state)
                temp = temp.parent
            
            end_time = time.time()
            print(f"--- {algorithm_type.replace('_', ' ').title()} Search Results ---")
            print(f"Solution Found!")
            print(f"Path Length: {len(path) - 1} moves")
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Max Queue Size: {max_queue_size}")
            print(f"Time Taken: {end_time - start_time:.4f} seconds")
            return path[::-1]

        for move in get_possible_moves(current_node.state):
            new_state = apply_move(current_node.state, move)
            new_state_tuple = tuple(map(tuple, new_state))
            
            g_cost = current_node.g + 1

            if new_state_tuple in visited and visited[new_state_tuple] <= g_cost:
                continue
            
            visited[new_state_tuple] = g_cost
            
            h_cost = heuristic_func(new_state)
            new_node = SolitaireNode(new_state, current_node, g=g_cost, h=h_cost)

            priority = new_node.h if algorithm_type == 'best_first' else new_node.f
            heapq.heappush(open_list, (priority, id(new_node), new_node))

    end_time = time.time()
    print(f"--- {algorithm_type.replace('_', ' ').title()} Search Results ---")
    print("No solution found.")
    print(f"Nodes Expanded: {nodes_expanded}")
    print(f"Max Queue Size: {max_queue_size}")
    print(f"Time Taken: {end_time - start_time:.4f} seconds")
    return None