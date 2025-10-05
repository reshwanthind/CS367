from typing import Tuple, List

def char_level_edit_distance(s1: str, s2: str) -> int:
    """
    Calculates the Levenshtein (edit) distance between two strings.
    This represents the cost of aligning, inserting, or deleting characters.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # Deletion
                           dp[i][j - 1] + 1,        # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution

    return dp[m][n]


def get_transition_cost(state: Tuple[int, int, int], doc1: List[str], doc2: List[str]) -> int:
    """
    Calculates the cost g(n) for a single transition (move).
    
    State Legend:
    - move 0: Align sentences
    - move 1: Insert sentence from doc2 (skip in doc1)
    - move 2: Delete sentence from doc1 (skip in doc2)
    """
    idx1, idx2, move = state
    cost = 0

    if move == 0:  # Alignment
        sentence1 = doc1[idx1 - 1]
        sentence2 = doc2[idx2 - 1]
        cost = char_level_edit_distance(sentence1, sentence2)
    elif move == 1:  # Insertion
        sentence2 = doc2[idx2 - 1]
        cost = len(sentence2) # Cost of inserting all characters
    elif move == 2:  # Deletion
        sentence1 = doc1[idx1 - 1]
        cost = len(sentence1) # Cost of deleting all characters

    return cost


def heuristic_cost(current_state: Tuple[int, int, int], goal_state: Tuple[int, int, int]) -> int:
    """
    Estimates the future cost h(n) from the current state to the goal.
    An admissible heuristic: the cost of insertions/deletions for the
    remaining sentences. We assume a cost of 0 for alignment for the heuristic.
    """
    current_idx1, current_idx2, _ = current_state
    goal_idx1, goal_idx2, _ = goal_state

    remaining_doc1 = goal_idx1 - current_idx1
    remaining_doc2 = goal_idx2 - current_idx2
    
    # The heuristic is the difference in the number of remaining sentences,
    # as these must be inserted or deleted.
    return abs(remaining_doc1 - remaining_doc2)