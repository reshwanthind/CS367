class SolitaireNode:
    """Node class to store state, parent, and costs for search algorithms."""
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g  # Cost from start to current node (path length)
        self.h = h  # Heuristic estimate to goal
        self.f = g + h  # Total cost for A*

    def __lt__(self, other):
        """Comparator for the priority queue."""
        return self.f < other.f

def is_valid_position(r, c, state):
    """Check if a position is within the 7x7 grid and on the playable board."""
    if not (0 <= r < 7 and 0 <= c < 7):
        return False
    # The '-' character denotes an invalid, off-board position
    return state[r][c] != '-'

def get_possible_moves(state):
    """Get all possible valid moves from the current state."""
    moves = []
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]  # Up, Down, Left, Right
    jump_over = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(7):
        for c in range(7):
            if state[r][c] == 'O':
                for i in range(len(directions)):
                    dr, dc = directions[i]
                    jr, jc = jump_over[i]

                    end_r, end_c = r + dr, c + dc
                    jump_r, jump_c = r + jr, c + jc

                    if is_valid_position(end_r, end_c, state) and is_valid_position(jump_r, jump_c, state):
                        if state[end_r][end_c] == '0' and state[jump_r][jump_c] == 'O':
                            moves.append((r, c, end_r, end_c))
    return moves

def apply_move(state, move):
    """Return a new state after applying a move."""
    new_state = [row[:] for row in state]
    start_r, start_c, end_r, end_c = move
    jump_r = (start_r + end_r) // 2
    jump_c = (start_c + end_c) // 2

    new_state[start_r][start_c] = '0'
    new_state[jump_r][jump_c] = '0'
    new_state[end_r][end_c] = 'O'
    return new_state

def is_goal(state):
    """Check if the state is the goal state (one marble in the center)."""
    marble_count = sum(row.count('O') for row in state)
    return marble_count == 1 and state[3][3] == 'O'

def print_board(state):
    """Prints the board in a readable format."""
    for row in state:
        print(" ".join(row))
    print("-" * 20)