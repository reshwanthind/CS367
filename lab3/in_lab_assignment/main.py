# main.py

from board_utils import print_board
from heuristics import heuristic_manhattan_distance
from search_algorithms import solve

def main():
    """Main function to run the solitaire solver."""
    start_state = [
        ['-', '-', 'O', 'O', 'O', '-', '-'],
        ['-', '-', 'O', 'O', 'O', '-', '-'],
        ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', '0', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['-', '-', 'O', 'O', 'O', '-', '-'],
        ['-', '-', 'O', 'O', 'O', '-', '-']
    ]

    print("Initial Board:")
    print_board(start_state)
    
    # Using the more informative Manhattan Distance heuristic for both searches
    
    # Run Best-First Search
    solution_bfs = solve(start_state, heuristic_manhattan_distance, 'best_first')
    
    print("\n" + "="*40 + "\n")

    # Run A* Search
    solution_a_star = solve(start_state, heuristic_manhattan_distance, 'a_star')


if __name__ == "__main__":
    main()