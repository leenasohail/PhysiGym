import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def hexagon_polygon(cx, cy, R):
    """Return shapely Polygon of regular hexagon centered at (cx,cy) with radius R."""
    vertices = [
        (cx + R * np.cos(np.pi/3 * k), cy + R * np.sin(np.pi/3 * k))
        for k in range(6)
    ]
    return Polygon(vertices)


def generate_hex_layers(cx, cy, n_layers, cell_radius):
    """Generate concentric hexagonal lattice of 'other_tissue' cells."""
    points = [(cx, cy)]  # center
    for layer in range(1, n_layers + 1):
        q, r = layer, 0
        directions = [(0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,0)]
        for d in range(6):
            dq, dr = directions[d]
            for _ in range(layer):
                x = np.sqrt(3) * (q + r/2) * cell_radius + cx
                y = 1.5 * r * cell_radius + cy
                points.append((x, y))
                q, r = q + dq, r + dr
    return np.array(points)


def generate_hex_population(cx, cy, R, n_layers, n_tumor, n_cell1, cell_radius=10.0):
    hex_poly = hexagon_polygon(cx, cy, R)

    # other_tissue lattice
    other_points = generate_hex_layers(cx, cy, n_layers, cell_radius)
    other_df = pd.DataFrame({
        "x": other_points[:,0],
        "y": other_points[:,1],
        "z": 0.0,
        "type": "other_tissue",
    })

    # tumor inside hex
    tumor_x, tumor_y = [], []
    while len(tumor_x) < n_tumor:
        x = np.random.uniform(cx - R, cx + R)
        y = np.random.uniform(cy - R, cy + R)
        if hex_poly.contains(Point(x, y)):
            tumor_x.append(x)
            tumor_y.append(y)
    tumor_df = pd.DataFrame({
        "x": tumor_x,
        "y": tumor_y,
        "z": 0.0,
        "type": "tumor",
    })

    # cell_1 outside hex (in bounding box but not inside hex)
    cell1_x, cell1_y = [], []
    while len(cell1_x) < n_cell1:
        x = np.random.uniform(cx - 1.5*R, cx + 1.5*R)
        y = np.random.uniform(cy - 1.5*R, cy + 1.5*R)
        if not hex_poly.contains(Point(x, y)):
            cell1_x.append(x)
            cell1_y.append(y)
    cell1_df = pd.DataFrame({
        "x": cell1_x,
        "y": cell1_y,
        "z": 0.0,
        "type": "cell_1",
    })

    return pd.concat([other_df, tumor_df, cell1_df], ignore_index=True)
