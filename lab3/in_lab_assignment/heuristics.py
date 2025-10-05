def heuristic_marble_count(state):
    """h1: Counts marbles. Goal is 1, so moves left = count - 1."""
    return sum(row.count('O') for row in state) - 1

def heuristic_manhattan_distance(state):
    """h2: Sum of Manhattan distances of all marbles from the center (3,3)."""
    center_r, center_c = 3, 3
    total_distance = 0
    for r in range(7):
        for c in range(7):
            if state[r][c] == 'O':
                total_distance += abs(r - center_r) + abs(c - center_c)
    return total_distance