import numpy as np
from time import time
from environment import Environment
from agent import Agent
from heuristics import heuristic_manhattan, heuristic_misplaced, heuristic_zero

def run_experiment():
    """
    Runs the 8-puzzle solver for various depths and prints the performance metrics.
    """
    # Define the goal state for the 8-puzzle
    goal_state = np.array([
        ['1', '2', '3'], 
        ['8', '_', '4'], 
        ['7', '6', '5']
    ])

    # Depths to test. Start with smaller depths first.
    depths = [2, 5, 8, 10, 12, 15, 18, 20] 
    num_runs_per_depth = 10 # Number of random puzzles to average over for each depth

    # Choose the heuristic to use for the experiment
    chosen_heuristic = heuristic_manhattan
    print(f"Running experiment with heuristic: {chosen_heuristic.__name__}\n")

    results = {}

    print(f"{'Depth':<10}{'Avg Time (s)':<15}{'Avg Memory (KB)':<20}{'Avg Nodes Expanded':<20}")
    print("-" * 70)

    for depth in depths:
        total_time = 0
        total_mem = 0
        total_nodes = 0

        for i in range(num_runs_per_depth):
            print(f"Running depth {depth}, instance {i+1}/{num_runs_per_depth}...", end='\r')
            env = Environment(depth=depth, goal_state=goal_state)
            agent = Agent(env=env, heuristic_func=chosen_heuristic)

            start_time = time()
            result = agent.run()
            end_time = time()

            if result:
                nodes_expanded, soln_depth = result
                total_time += end_time - start_time
                total_mem += agent.get_memory_usage() / 1024 # Convert to KB
                total_nodes += nodes_expanded
            else:
                print(f"No solution found for a puzzle of depth {depth}")


        avg_time = total_time / num_runs_per_depth
        avg_mem = total_mem / num_runs_per_depth
        avg_nodes = total_nodes / num_runs_per_depth
        
        results[depth] = (avg_time, avg_mem, avg_nodes)
        
        print(f"{depth:<10}{avg_time:<15.4f}{avg_mem:<20.2f}{avg_nodes:<20.1f}")

if __name__ == "__main__":
    run_experiment()