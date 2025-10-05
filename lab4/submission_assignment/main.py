import os
import time
import psutil
import copy
from tqdm import tqdm
import cv2

# The import for 'calculate_grid_cost' has been added here
from utils import (
    load_mat_image,
    create_patches,
    reconstruct_image,
    display_image,
    calculate_grid_cost, # <-- THIS WAS THE MISSING PART
)
from solver import greedy_bfs_fill, simulated_annealing


def main():
    """Main execution script for the jigsaw puzzle solver."""

    # --- Configuration ---
    INPUT_FILE = os.path.join( "data", "scrambled_lena.mat")
    OUTPUT_DIR = "output"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "solved_lena.png")
    NUM_PIECES = 16

    # --- Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process = psutil.Process()
    start_time = time.time()

    # --- Load and Prepare Data ---
    print("Loading and preparing image patches...")
    image_matrix = load_mat_image(INPUT_FILE)
    if image_matrix is None:
        return

    patches = create_patches(image_matrix)
    print(f"Created {len(patches)} patches.")

    # --- Step 1: Find the Best Initial Greedy Solution ---
    print("\nStep 1: Finding the best initial greedy solution by trying all starting pieces...")
    best_greedy_grid = None
    min_greedy_cost = float("inf")

    for i in tqdm(range(NUM_PIECES), desc="Trying start pieces"):
        grid = greedy_bfs_fill(patches, start_patch_id=i)
        cost = calculate_grid_cost(grid, patches)
        if cost < min_greedy_cost:
            min_greedy_cost = cost
            best_greedy_grid = grid

    print(f"Found best initial grid with cost: {min_greedy_cost:.2f}")
    initial_solution_img = reconstruct_image(patches, best_greedy_grid)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "initial_greedy_solution.png"), initial_solution_img)

    # --- Step 2: Refine the Best Greedy Solution with Simulated Annealing ---
    print("\nStep 2: Refining the solution using Simulated Annealing...")
    final_grid, final_cost = simulated_annealing(
        best_greedy_grid, patches, initial_temp=50000, alpha=0.998
    )

    # --- Finalize and Report ---
    total_time = time.time() - start_time
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB

    print("\n--- Solver Finished ---")
    print(f"Final solution cost: {final_cost:.2f}")
    print(f"Total time required: {total_time:.2f} seconds")
    print(f"Memory usage: {memory_usage:.2f} MB")

    # Save and display the final image
    final_image = reconstruct_image(patches, final_grid)
    cv2.imwrite(OUTPUT_FILE, final_image)
    print(f"Solved image saved to: {OUTPUT_FILE}")
    display_image(final_image, title="Final Solved Puzzle")


if __name__ == "__main__":
    main()