# filename: plotter.py
import matplotlib.pyplot as plt

def plot_tour(tour, locations, title, is_map=False):
    """Plots the TSP tour."""
    plt.figure(figsize=(12, 8))
    
    # Extract coordinates
    coords = [loc[1:] for loc in locations]
    tour_coords = [coords[i] for i in tour]
    
    # Add the starting point to the end to close the loop
    tour_coords.append(tour_coords[0])
    
    x, y = zip(*tour_coords)
    
    # Plot tour path
    plt.plot(y, x, 'b-') # Note: lon=y, lat=x for map-like orientation
    
    # Plot cities
    plt.plot(y, x, 'ro')
    
    # Annotate cities
    for i, city in enumerate(tour):
        plt.text(coords[city][1], coords[city][0], locations[city][0], fontsize=9)
        
    if is_map:
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
    else:
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        
    plt.title(title)
    plt.grid(True)
    plt.show()