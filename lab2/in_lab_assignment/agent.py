import numpy as np
import heapq 
import sys

class Node:
    """
    A node in the search tree. Contains the state, parent node, and costs.
    """
    def __init__(self, state, parent, pcost, hcost):
        self.state = state
        self.parent = parent
        self.pcost = pcost  # g(n): Path cost from start to current node
        self.hcost = hcost  # h(n): Heuristic cost from current node to goal
        self.cost = pcost + hcost # f(n) = g(n) + h(n)

    # The __lt__ method is needed for heapq to compare nodes
    def __lt__(self, other):
        return self.cost < other.cost

    # Using state.tobytes() for hashing is more reliable for numpy arrays
    def __hash__(self):
        return hash(self.state.tobytes())

    def __eq__(self, other):
        return hash(self) == hash(other)

class PriorityQueue:
    """
    A more efficient Priority Queue implementation using Python's heapq library.
    The pop operation is O(log N) instead of O(N).
    """
    def __init__(self):
        self.queue = []

    def push(self, node):
        heapq.heappush(self.queue, node)

    def pop(self):
        return heapq.heappop(self.queue)

    def is_empty(self):
        return len(self.queue) == 0

    def __len__(self):
        return len(self.queue)

class Agent:
    """
    The A* search agent that solves the 8-puzzle.
    """
    def __init__(self, env, heuristic_func):
        self.env = env
        self.heuristic = heuristic_func
        self.frontier = PriorityQueue()
        self.explored = set()
        self.goal_node = None
        self.nodes_expanded = 0

    def run(self):
        """
        Executes the A* search algorithm.
        Returns a tuple of (nodes_expanded, solution_depth).
        """
        start_state = self.env.get_start_state()
        goal_state = self.env.goal_state
        
        # Create the initial node
        h_cost = self.heuristic(start_state, goal_state)
        init_node = Node(state=start_state, parent=None, pcost=0, hcost=h_cost)
        self.frontier.push(init_node)

        while not self.frontier.is_empty():
            curr_node = self.frontier.pop()

            if hash(curr_node) in self.explored:
                continue
            
            self.explored.add(hash(curr_node))
            self.nodes_expanded += 1

            if self.env.reached_goal(curr_node.state):
                self.goal_node = curr_node
                return self.nodes_expanded, self.get_solution_depth()

            for next_state in self.env.get_possible_moves(curr_node.state):
                node_hash = hash(next_state.tobytes())
                if node_hash not in self.explored:
                    h_cost = self.heuristic(next_state, goal_state)
                    new_node = Node(
                        state=next_state,
                        parent=curr_node,
                        pcost=curr_node.pcost + 1, # Uniform cost of 1 per move
                        hcost=h_cost
                    )
                    self.frontier.push(new_node)
        
        return None # Return None if no solution is found

    def get_solution_depth(self):
        """Calculates the depth of the solution path."""
        if not self.goal_node:
            return 0
        return self.goal_node.pcost

    def get_solution_path(self):
        """
        Backtracks from the goal node to the start node to produce the solution path.
        """
        if not self.goal_node:
            return []
            
        path = []
        node = self.goal_node
        while node is not None:
            path.append(node.state)
            node = node.parent
        
        # The path is from goal to start, so we reverse it
        return path[::-1]

    def get_memory_usage(self):
        """
        Estimates memory usage by the frontier and explored sets.
        This is a rough estimate; actual memory usage is more complex.
        """
        frontier_mem = len(self.frontier) * sys.getsizeof(self.frontier.queue[0]) if not self.frontier.is_empty() else 0
        # For a set, we estimate based on the size of a sample item
        explored_mem = len(self.explored) * sys.getsizeof(next(iter(self.explored))) if self.explored else 0
        return frontier_mem + explored_mem