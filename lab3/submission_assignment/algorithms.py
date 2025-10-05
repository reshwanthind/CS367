import random
import itertools

def hill_climbing(clauses, n, heuristic_func, max_restarts=10):
    """Performs steepest-ascent hill-climbing with random restarts."""
    best_state_overall = []
    best_score_overall = -1
    
    total_nodes_expanded = 0
    total_path_length = 0

    for _ in range(max_restarts):
        current_state = [random.randint(0, 1) for _ in range(n)]
        current_score = heuristic_func(current_state, clauses)
        path_length = 0
        
        while True:
            neighbors = []
            for i in range(n):
                neighbor = current_state[:]
                neighbor[i] = 1 - neighbor[i]
                neighbors.append(neighbor)
            
            total_nodes_expanded += len(neighbors)
            
            neighbor_scores = [heuristic_func(nb, clauses) for nb in neighbors]
            best_neighbor_score = max(neighbor_scores)
            
            if best_neighbor_score <= current_score:
                break
            
            best_neighbor = neighbors[neighbor_scores.index(best_neighbor_score)]
            current_state = best_neighbor
            current_score = best_neighbor_score
            path_length += 1
        
        total_path_length += path_length
        if current_score > best_score_overall:
            best_score_overall = current_score
            best_state_overall = current_state
            if best_score_overall == len(clauses):
                break
                
    return best_state_overall, best_score_overall, total_path_length, total_nodes_expanded



def beam_search(clauses, n, heuristic_func, beam_width, max_iter=100):
    """Performs beam search for SAT."""
    current_states = [[random.randint(0, 1) for _ in range(n)] for _ in range(beam_width)]
    nodes_expanded = 0
    path_length = 0

    for i in range(max_iter):
        all_neighbors = []
        for state in current_states:
            for i in range(n):
                neighbor = state[:]
                neighbor[i] = 1 - neighbor[i]
                all_neighbors.append(neighbor)
        
        nodes_expanded += len(all_neighbors)
        all_candidates = current_states + all_neighbors
        
        scored_candidates = [(heuristic_func(s, clauses), s) for s in all_candidates]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        next_states = []
        seen_states = set()
        for _, state in scored_candidates:
            state_tuple = tuple(state)
            if state_tuple not in seen_states:
                next_states.append(state)
                seen_states.add(state_tuple)
            if len(next_states) == beam_width:
                break
        
        current_states = next_states
        path_length += 1
        
        if heuristic_func(current_states[0], clauses) == len(clauses):
            break

    best_state = current_states[0]
    best_score = heuristic_func(best_state, clauses)
    
    return best_state, best_score, path_length, nodes_expanded




def get_neighbors_k_flip(state, k):
    """Generator for neighbors by flipping k bits."""
    n = len(state)
    for indices in itertools.combinations(range(n), k):
        neighbor = state[:]
        for i in indices:
            neighbor[i] = 1 - neighbor[i]
        yield neighbor

def variable_neighborhood_descent(clauses, n, heuristic_func, max_k=3):
    """Performs Variable Neighborhood Descent for SAT."""
    current_state = [random.randint(0, 1) for _ in range(n)]
    current_score = heuristic_func(current_state, clauses)
    
    nodes_expanded = 0
    path_length = 0
    k = 1
    
    while k <= max_k:
        best_neighbor_in_k, best_score_in_k = None, -1
        
        for neighbor in get_neighbors_k_flip(current_state, k):
            nodes_expanded += 1
            score = heuristic_func(neighbor, clauses)
            if score > best_score_in_k:
                best_score_in_k = score
                best_neighbor_in_k = neighbor

        if best_score_in_k > current_score:
            current_state = best_neighbor_in_k
            current_score = best_score_in_k
            path_length += 1
            k = 1
            if current_score == len(clauses):
                break
        else:
            k += 1

    return current_state, current_score, path_length, nodes_expanded

