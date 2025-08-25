import numpy as np
import pandas as pd


def generate_hex_layers(cx, cy, max_radius, cell_radius):
    """Generate hexagonal lattice covering a circle of radius max_radius."""
    points = []
    dx = np.sqrt(3) * cell_radius
    dy = 1.5 * cell_radius
    qmax = int(max_radius / dx) + 2
    rmax = int(max_radius / dy) + 2

    for q in range(-qmax, qmax + 1):
        for r in range(-rmax, rmax + 1):
            x = cell_radius * np.sqrt(3) * (q + r / 2.0) + cx
            y = cell_radius * 1.5 * r + cy
            if np.hypot(x, y) <= max_radius:  # keep inside bounding circle
                points.append((x, y))
    return np.array(points)


def generate_cell_positions(
    cell_radius=10,
    n_tumor=512,
    tumor_radius=200,
    n_layers=5,
    n_cell1=256,
    cell1_radius=512,
    gap=25.0,  # <-- extra empty space between tumor and tissue
):
    """
    Generate positions for tumor, other_tissue (hex lattice in circular ring), and cell_1 (random).
    """

    # --- Tumor (random inside circle) ---
    tumor_x, tumor_y = [], []
    while len(tumor_x) < n_tumor:
        x = np.random.uniform(-tumor_radius, tumor_radius)
        y = np.random.uniform(-tumor_radius, tumor_radius)
        if x**2 + y**2 <= tumor_radius**2:
            tumor_x.append(x)
            tumor_y.append(y)
    tumor_df = pd.DataFrame({"x": tumor_x, "y": tumor_y, "z": 0.0, "type": "tumor"})

    # --- Other tissue (hexagonal lattice ring with gap) ---
    inner_radius = tumor_radius + gap
    max_radius = inner_radius + n_layers * 2 * cell_radius
    hex_points = generate_hex_layers(0, 0, max_radius, cell_radius)

    other_points = [
        (x, y) for x, y in hex_points if inner_radius < np.hypot(x, y) <= max_radius
    ]

    other_df = pd.DataFrame(
        {
            "x": [p[0] for p in other_points],
            "y": [p[1] for p in other_points],
            "z": 0.0,
            "type": "other_tissue",
            "volume": "",
            "cycle entry": "",
            "custom:GFP": "",
            "custom:sample": "",
        }
    )

    # --- Cell_1 (random scattered far away) ---
    cell1_x, cell1_y = [], []
    while len(cell1_x) < n_cell1:
        x = np.random.uniform(-cell1_radius, cell1_radius)
        y = np.random.uniform(-cell1_radius, cell1_radius)
        if np.hypot(x, y) > max_radius:  # outside tissue ring
            cell1_x.append(x)
            cell1_y.append(y)
    cell1_df = pd.DataFrame({"x": cell1_x, "y": cell1_y, "z": 0.0, "type": "cell_1"})

    # --- Combine ---
    return pd.concat([tumor_df, other_df, cell1_df], ignore_index=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = generate_cell_positions()

    fig, ax = plt.subplots(figsize=(8, 8))
    for t, c in zip(["tumor", "other_tissue", "cell_1"], ["green", "orange", "blue"]):
        subset = df[df["type"] == t]
        ax.scatter(subset.x, subset.y, s=20, c=c, label=t, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_title("Cell positions (2D)")
    ax.legend()
    plt.savefig("cells.png", dpi=300)
    plt.close()
