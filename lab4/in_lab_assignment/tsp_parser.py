import re

def parse_tsp_file(filepath):
    """Parses a TSPLIB .tsp file to extract node coordinates."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the node coordinate section
    coord_section_match = re.search(r'NODE_COORD_SECTION\s*([\s\S]*?)(\s*EOF|$)', content)
    if not coord_section_match:
        raise ValueError("Could not find NODE_COORD_SECTION in file")

    coord_str = coord_section_match.group(1)
    lines = coord_str.strip().split('\n')
    
    locations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            # Format: node_id x_coord y_coord
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            locations.append((f"Node_{node_id}", x, y))
            
    return locations