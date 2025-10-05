import numpy as np
import matplotlib.pyplot as plt

def load_mat_image(file_path):
    """Loads image data from a structured text .mat file."""
    matrix_data = []
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()[5:]  # Skip header
        for line in lines:
            line = line.strip()
            if line:
                matrix_data.append(int(line))
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'")
        return None

    matrix = np.array(matrix_data)
    if matrix.size != 512 * 512:
        print(f"Error: Expected 262144 elements, but got {matrix.size}.")
        return None
    
    # Reshape and transpose to correct orientation
    return matrix.reshape((512, 512)).T

def create_patches(image, patch_size=128):
    """Splits an image into a dictionary of patches."""
    patches = {}
    num_patches_dim = image.shape[0] // patch_size
    patch_id = 0
    for i in range(num_patches_dim):
        for j in range(num_patches_dim):
            patch = image[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            patches[patch_id] = patch
            patch_id += 1
    return patches

def reconstruct_image(patches, grid):
    """Reconstructs a full image from a grid of patch IDs."""
    patch_size = patches[0].shape[0]
    grid_size = len(grid)
    full_image = np.zeros((grid_size * patch_size, grid_size * patch_size), dtype=np.uint8)
    for i, row in enumerate(grid):
        for j, patch_id in enumerate(row):
            full_image[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ] = patches[patch_id]
    return full_image

def display_image(image, title="Image"):
    """Displays an image using matplotlib."""
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.show()

def calculate_edge_dissimilarity(patch1, patch2, direction):
    """Calculates the sum of absolute differences between the edges of two patches."""
    dissimilarity = 0
    patch_size = patch1.shape[0]
    if direction == 'right': # patch2 is to the right of patch1
        dissimilarity = np.sum(np.abs(patch1[:, -1] - patch2[:, 0]))
    elif direction == 'left': # patch2 is to the left of patch1
        dissimilarity = np.sum(np.abs(patch1[:, 0] - patch2[:, -1]))
    elif direction == 'down': # patch2 is below patch1
        dissimilarity = np.sum(np.abs(patch1[-1, :] - patch2[0, :]))
    elif direction == 'up': # patch2 is above patch1
        dissimilarity = np.sum(np.abs(patch1[0, :] - patch2[-1, :]))
    return dissimilarity

def get_best_matching_patch(parent_patch, available_patches, direction):
    """Finds the patch with the minimum edge dissimilarity from a list of available patches."""
    best_patch_id = -1
    min_dissimilarity = float('inf')

    for patch_id, patch in available_patches.items():
        dissimilarity = calculate_edge_dissimilarity(parent_patch, patch, direction)
        if dissimilarity < min_dissimilarity:
            min_dissimilarity = dissimilarity
            best_patch_id = patch_id
            
    return best_patch_id

def calculate_grid_cost(grid, patches):
    """Calculates the total dissimilarity cost for an entire grid arrangement."""
    total_cost = 0
    grid_size = len(grid)
    for r in range(grid_size):
        for c in range(grid_size):
            current_patch = patches[grid[r][c]]
            # Check right neighbor
            if c + 1 < grid_size:
                right_patch = patches[grid[r][c+1]]
                total_cost += calculate_edge_dissimilarity(current_patch, right_patch, 'right')
            # Check bottom neighbor
            if r + 1 < grid_size:
                bottom_patch = patches[grid[r+1][c]]
                total_cost += calculate_edge_dissimilarity(current_patch, bottom_patch, 'down')
    return total_cost