def check_literal(literal, state):
    """Checks if a single literal is true in the given state."""
    if literal > 0:
        return state[abs(literal) - 1] == 1
    else:
        return state[abs(literal) - 1] == 0

def heuristic_1_satisfied_clauses(state, clauses):
    """H1: Counts the number of clauses satisfied by the state."""
    count = 0
    for clause in clauses:
        for literal in clause:
            if check_literal(literal, state):
                count += 1
                break
    return count

def heuristic_2_satisfaction_degree(state, clauses):
    """H2: Sum of true literals across all clauses."""
    total_degree = 0
    for clause in clauses:
        degree = 0
        for literal in clause:
            if check_literal(literal, state):
                degree += 1
        total_degree += degree
    return total_degree