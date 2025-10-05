import time

# Import functions from our other files
from problem_generator import generate_k_sat_problem
from heuristics import heuristic_1_satisfied_clauses, heuristic_2_satisfaction_degree
from algorithms import hill_climbing, beam_search, variable_neighborhood_descent


def main():
    """Main function to run the SAT solver comparison."""
    
    # --- Problem Parameters ---
    K = 3
    N = 20
    M = 85
    
    print(f"--- Generating {K}-SAT Problem ---")
    print(f"Variables (n): {N}")
    print(f"Clauses (m): {M}")
    print("-" * 30)

    clauses = generate_k_sat_problem(K, M, N)

    # --- Solvers and Heuristics to Compare ---
    heuristics = {
        "H1 (Satisfied Clauses)": heuristic_1_satisfied_clauses,
        "H2 (Satisfaction Degree)": heuristic_2_satisfaction_degree
    }

    solvers = {
        "Hill-Climbing": lambda h_func: hill_climbing(clauses, N, h_func),
        "Beam Search (w=3)": lambda h_func: beam_search(clauses, N, h_func, beam_width=3),
        "Beam Search (w=4)": lambda h_func: beam_search(clauses, N, h_func, beam_width=4),
        "VND (k_max=3)": lambda h_func: variable_neighborhood_descent(clauses, N, h_func, max_k=3)
    }


    # --- Run Experiments ---
    print(f"\n{'Algorithm':<22} | {'Heuristic Used':<25} | {'Clauses Satisfied':<20} | {'Literals Satisfied':<20} | {'Time (s)':>10} | {'Penetrance':>15}")
    print("-" * 130)

    for h_name, h_func in heuristics.items():
        for s_name, s_func in solvers.items():
            start_time = time.time()
            
            # Run the solver to get the best state found
            # We no longer need the 'best_score' it returns, as we will recalculate
            best_state, _, path_len, nodes_exp = s_func(h_func)
            
            end_time = time.time()
            duration = end_time - start_time

            penetrance = (path_len / nodes_exp) if nodes_exp > 0 else 0
            
            # Evaluate the final state using BOTH metrics 
            clauses_satisfied = 0
            literals_satisfied = 0
            # Ensure a valid state was returned before scoring
            if best_state:
                clauses_satisfied = heuristic_1_satisfied_clauses(best_state, clauses)
                literals_satisfied = heuristic_2_satisfaction_degree(best_state, clauses)
            
            # Format the output strings 
            clauses_str = f"{clauses_satisfied}/{M}"
            literals_str = f"{literals_satisfied}"

            # Print the result row with the new format 
            print(f"{s_name:<22} | {h_name:<25} | {clauses_str:<20} | {literals_str:<20} | {duration:>10.4f} | {penetrance:>15.5e}")

if __name__ == "__main__":
    main()