import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def hexagon_polygon(cx, cy, R):
    """Return shapely Polygon of regular hexagon centered at (cx,cy) with radius R."""
    vertices = [
        (cx + R * np.cos(np.pi / 3 * k), cy + R * np.sin(np.pi / 3 * k))
        for k in range(6)
    ]
    return Polygon(vertices)


def generate_hex_layers(cx, cy, n_layers, cell_radius):
    """Generate concentric hexagonal lattice of 'other_tissue' cells centered at (cx, cy)."""
    points = []
    for q in range(-n_layers, n_layers + 1):
        for r in range(-n_layers, n_layers + 1):
            s = -q - r
            if abs(s) <= n_layers:
                x = cell_radius * np.sqrt(3) * (q + r / 2.0) + cx
                y = cell_radius * 1.5 * r + cy
                points.append((x, y))
    return np.array(points)


def generate_cell_positions(
    cell_radius=3,
    n_tumor=200,
    tumor_radius=20,
    n_layers=6,
    n_cell1=300,
    cell1_radius=90,
):
    """
    Generate positions for tumor, other_tissue (hex lattice), and cell_1 (random).
    """

    # --- Tumor (random inside circle) ---
    tumor_x, tumor_y = [], []
    while len(tumor_x) < n_tumor:
        x = np.random.uniform(-tumor_radius, tumor_radius)
        y = np.random.uniform(-tumor_radius, tumor_radius)
        if x**2 + y**2 <= tumor_radius**2:  # keep inside circle
            tumor_x.append(x)
            tumor_y.append(y)
    tumor_df = pd.DataFrame({"x": tumor_x, "y": tumor_y, "z": 0.0, "type": "tumor"})

    # --- Other tissue (dense hex lattice around tumor) ---
    hex_points = []
    dx = 2 * cell_radius
    dy = np.sqrt(3) * cell_radius
    for row in range(-n_layers, n_layers + 1):
        for col in range(-n_layers, n_layers + 1):
            x = col * dx + (row % 2) * cell_radius
            y = row * dy
            r = np.hypot(x, y)
            if r > tumor_radius:  # keep only outside tumor
                hex_points.append((x, y))
    other_df = pd.DataFrame(
        {
            "x": [p[0] for p in hex_points],
            "y": [p[1] for p in hex_points],
            "z": 0.0,
            "type": "other_tissue",
        }
    )

    # --- Cell_1 (random scattered far away) ---
    cell1_x, cell1_y = [], []
    while len(cell1_x) < n_cell1:
        x = np.random.uniform(-cell1_radius, cell1_radius)
        y = np.random.uniform(-cell1_radius, cell1_radius)
        if np.hypot(x, y) > (
            tumor_radius + 2 * cell_radius * n_layers
        ):  # outside tissue ring
            cell1_x.append(x)
            cell1_y.append(y)
    cell1_df = pd.DataFrame({"x": cell1_x, "y": cell1_y, "z": 0.0, "type": "cell_1"})

    # --- Combine ---
    return pd.concat([tumor_df, other_df, cell1_df], ignore_index=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = generate_cell_positions()

    fig, ax = plt.subplots(figsize=(6, 6))
    for t, c in zip(["tumor", "other_tissue", "cell_1"], ["green", "orange", "blue"]):
        subset = df[df["type"] == t]
        ax.scatter(subset.x, subset.y, s=20, c=c, label=t, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_title("Cell positions (2D)")
    ax.legend()
    plt.savefig("cells.png", dpi=300)
    plt.close()
