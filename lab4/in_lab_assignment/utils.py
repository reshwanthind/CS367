# filename: utils.py
import numpy as np

def calculate_haversine_distance(coord1, coord2):
    """Calculates the great-circle distance between two points on Earth."""
    R = 6371  # Earth radius in kilometers
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def calculate_euclidean_distance(coord1, coord2):
    """Calculates the 2D Euclidean distance."""
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def calculate_total_distance(tour, distance_matrix):
    """Calculates the total distance of a tour given a distance matrix."""
    total_dist = 0
    num_cities = len(tour)
    for i in range(num_cities):
        total_dist += distance_matrix[tour[i], tour[(i + 1) % num_cities]]
    return total_dist

def create_distance_matrix(locations, dist_func):
    """Creates a distance matrix for a list of locations."""
    num_locations = len(locations)
    matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(i, num_locations):
            dist = dist_func(locations[i][1:], locations[j][1:])
            matrix[i, j] = matrix[j, i] = dist
    return matrix