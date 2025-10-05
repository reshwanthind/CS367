import heapq
from typing import Tuple, List, Optional, Any
from cost_functions import get_transition_cost, heuristic_cost

class Node:
    """A node in the search graph for the A* algorithm."""
    def __init__(self, state: Tuple[int, int, int], parent: Optional['Node'] = None, g: int = 0, h: int = 0):
        self.state = state      # (index_doc1, index_doc2, move_type)
        self.parent = parent
        self.g = g              # Cost from start to current node
        self.h = h              # Heuristic cost from current to goal
        self.f = g + h          # Total estimated cost

    def __lt__(self, other: 'Node') -> bool:
        """Comparator for the priority queue (heap)."""
        return self.f < other.f
        
    def __eq__(self, other: object) -> bool:
        """Equality check based on state."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.state == other.state

def get_successors(node: Node, goal_state: Tuple[int, int, int]) -> List[Node]:
    """
    Generates successor nodes from the current node.

    Moves (c):
        0: Alignment (move one step in both documents)
        1: Insertion (move one step in document 2)
        2: Deletion (move one step in document 1)
    """
    successors = []
    curr_idx1, curr_idx2, _ = node.state
    goal_idx1, goal_idx2, _ = goal_state
    
    # Possible moves: (d_idx1, d_idx2, move_type)
    moves = [(1, 1, 0), (0, 1, 1), (1, 0, 2)]

    for d_idx1, d_idx2, move_type in moves:
        new_idx1 = curr_idx1 + d_idx1
        new_idx2 = curr_idx2 + d_idx2

        # Ensure the new state does not exceed the goal state indices
        if new_idx1 <= goal_idx1 and new_idx2 <= goal_idx2:
            new_state = (new_idx1, new_idx2, move_type)
            successors.append(Node(new_state, parent=node))
            
    return successors

def a_star(start_state: Tuple[int, int, int], goal_state: Tuple[int, int, int], doc1: List[str], doc2: List[str]) -> Optional[List[Tuple[int, int, int]]]:
    """
    Performs A* search to find the optimal alignment path between two documents.
    """
    start_node = Node(start_state, g=0, h=heuristic_cost(start_state, goal_state))
    
    open_list = [start_node]  # Use a list as a priority queue with heapq
    visited = set()
    nodes_explored = 0

    while open_list:
        # Get the node with the lowest f-score
        node = heapq.heappop(open_list)
        nodes_explored += 1

        if node.state in visited:
            continue
        visited.add(node.state)
        
        # Goal reached when current indices match goal indices
        if node.state[0] == goal_state[0] and node.state[1] == goal_state[1]:
            print(f"Goal reached! Total nodes explored: {nodes_explored}")
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1] # Return reversed path

        for successor in get_successors(node, goal_state):
            if successor.state in visited:
                continue
                
            # Calculate costs for the successor
            g_cost = node.g + get_transition_cost(successor.state, doc1, doc2)
            h_cost = heuristic_cost(successor.state, goal_state)
            
            successor.g = g_cost
            successor.h = h_cost
            successor.f = g_cost + h_cost
            
            heapq.heappush(open_list, successor)
            
    print(f"No path found. Total nodes explored: {nodes_explored}")
    return None