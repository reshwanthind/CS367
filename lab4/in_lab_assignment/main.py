# filename: main.py
import os
import requests
from locations import RAJA_LOCATIONS
from utils import create_distance_matrix, calculate_haversine_distance, calculate_euclidean_distance
from tsp_parser import parse_tsp_file
from plotter import plot_tour
from simulated_annealing import solve_tsp_sa

# --- Configuration ---
# SA parameters can be tuned for better results or faster execution
SA_PARAMS = {
    "initial_temp": 10000,
    "cooling_rate": 0.9995,
    "min_temp": 1e-4,
}

VLSI_PROBLEMS = {
    "xqf131.tsp": 564,
    "xqg237.tsp": 1019,
    "pbl395.tsp": 1198,
    "pbm436.tsp": 1443,
    "xql662.tsp": 2513
}
VLSI_DATA_URL = "http://www.math.uwaterloo.ca/tsp/vlsi/"

def download_vlsi_data():
    """Downloads the necessary .tsp files if they don't exist."""
    if not os.path.exists("vlsi_data"):
        print("Creating directory 'vlsi_data'...")
        os.makedirs("vlsi_data")
        
    for filename in VLSI_PROBLEMS.keys():
        filepath = os.path.join("vlsi_data", filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                response = requests.get(VLSI_DATA_URL + filename)
                response.raise_for_status()
                with open(filepath, 'w') as f:
                    f.write(response.text)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {filename}: {e}")
                
def run_rajasthan_tour():
    """Plans and displays the tour for Rajasthan."""
    print("\n" + "="*50)
    print(" Planning Tour for Rajasthan")
    print("="*50)
    
    # Create distance matrix using Haversine distance
    dist_matrix = create_distance_matrix(RAJA_LOCATIONS, calculate_haversine_distance)
    
    # Run Simulated Annealing
    best_tour_indices, best_dist = solve_tsp_sa(dist_matrix, **SA_PARAMS)
    
    # Map indices back to city names
    city_names = [loc[0] for loc in RAJA_LOCATIONS]
    best_tour_cities = [city_names[i] for i in best_tour_indices]
    
    print("\n--- Optimized Rajasthan Tour ---")
    print(f"Total Distance: {best_dist:.2f} km")
    print("Tour Order:")
    print(" -> ".join(best_tour_cities))
    
    # Plot the result
    plot_tour(
        best_tour_indices,
        RAJA_LOCATIONS,
        f"Rajasthan Tour Plan (Distance: {best_dist:.2f} km)",
        is_map=True
    )

def run_vlsi_benchmark():
    """Runs the SA solver on the VLSI dataset and reports results."""
    print("\n" + "="*50)
    print("Benchmarking with VLSI Dataset")
    print("="*50)
    
    download_vlsi_data()
    
    print("\n--- Benchmark Results ---")
    print(f"{'Instance':<15} {'Optimal':<10} {'SA Result':<12} {'Difference (%)':<15}")
    print("-"*55)

    for filename, optimal_dist in VLSI_PROBLEMS.items():
        filepath = os.path.join("vlsi_data", filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filename}, file not found.")
            continue
            
        locations = parse_tsp_file(filepath)
        dist_matrix = create_distance_matrix(locations, calculate_euclidean_distance)
        
        print(f"\nSolving {filename} ({len(locations)} cities)...")
        best_tour, best_dist = solve_tsp_sa(dist_matrix, iter_per_temp=len(locations)*2, **SA_PARAMS)
        
        diff = ((best_dist - optimal_dist) / optimal_dist) * 100
        print(f"{filename:<15} {optimal_dist:<10} {best_dist:<12.2f} {f'+{diff:.2f}%':<15}")

        # Uncomment the line below to plot each VLSI result
        # plot_tour(best_tour, locations, f"{filename} - SA Result: {best_dist:.2f}")

if __name__ == "__main__":
    run_rajasthan_tour()
    run_vlsi_benchmark()