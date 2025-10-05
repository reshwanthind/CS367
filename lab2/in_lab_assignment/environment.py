import numpy as np

class Environment:
    """
    Represents the 8-Puzzle environment.
    - Generates a solvable start state by making 'depth' random moves from the goal state.
    - Provides valid next states from a given state.
    - Checks if the goal state has been reached.
    """
    def __init__(self, depth, goal_state):
        self.goal_state = goal_state
        self.start_state = self._generate_start_state(depth)

    def _generate_start_state(self, depth):
        """
        Generates a start state by moving backward from the goal.
        This ensures that the generated puzzle is always solvable.
        """
        current_state = np.copy(self.goal_state)
        for _ in range(depth):
            possible_next_states = self._get_next_states(current_state)
            # Choose a random move
            current_state = possible_next_states[np.random.randint(0, len(possible_next_states))]
        return current_state
    
    def get_start_state(self):
        return self.start_state

    def _get_blank_space_pos(self, state):
        """Helper function to find the (row, col) of the blank space '_'."""
        pos = np.where(state == '_')
        return pos[0][0], pos[1][0]

    def _get_next_states(self, state):
        """Returns a list of all possible states reachable from the current state."""
        possible_states = []
        row, col = self._get_blank_space_pos(state)

        # Possible moves: up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

        for dr, dc in moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = np.copy(state)
                # Swap the blank tile with the adjacent tile
                new_state[row, col], new_state[new_row, new_col] = new_state[new_row, new_col], new_state[row, col]
                possible_states.append(new_state)
        
        return possible_states

    def get_possible_moves(self, state):
        return self._get_next_states(state)

    def reached_goal(self, state):
        """Checks if the given state is the goal state."""
        return np.array_equal(state, self.goal_state)