import numpy as np
import pandas as pd
import random
from typing import Union
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os


##################
# Helper Functions
##################


def generate_ellipse_points(
    n, r1, r2, center=(0, 0), orientation_deg=0.0, jitter=0.0, perimeter=False
):
    """
    Generate 2D points in or on an ellipse.

    Parameters
    ----------
    n : int
        Number of points
    r1, r2 : float
        Semi-axes along x and y
    center : tuple
        (x, y) center of ellipse
    orientation_deg : float
        Rotation of ellipse in degrees
    jitter : float
        Standard deviation of Gaussian noise added to points
    perimeter : bool
        If True, points lie along the perimeter; else inside the ellipse

    Returns
    -------
    xr, yr : arrays of shape (n,)
    """
    if perimeter:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = np.ones(n)
    else:
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))  # uniform in area

    x = radii * r1 * np.cos(angles)
    y = radii * r2 * np.sin(angles)

    if orientation_deg != 0.0:
        theta = np.radians(orientation_deg)
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        x, y = x_rot, y_rot

    x += center[0]
    y += center[1]

    if jitter > 0.0:
        x += np.random.normal(0, jitter, n)
        y += np.random.normal(0, jitter, n)

    return x, y


##################
# Population Generators
##################


def generate_population_circulars(
    n_tumor,
    n_cell_1,
    x_min,
    x_max,
    y_min,
    y_max,
    tumor_scale=(0.4, 0.4),
    cell1_scale=(0.8, 0.8),
    jitter_tumor=15.0,
    jitter_cell_1=10.0,
):
    """Generate tumor and surrounding cell_1 in ellipses."""
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_width = (x_max - x_min) / 2
    half_height = (y_max - y_min) / 2

    r1_tumor, r2_tumor = tumor_scale[0] * half_width, tumor_scale[1] * half_height
    r1_cell1, r2_cell1 = cell1_scale[0] * half_width, cell1_scale[1] * half_height

    tumor_x, tumor_y = generate_ellipse_points(
        n_tumor, r1_tumor, r2_tumor, center=(center_x, center_y), jitter=jitter_tumor
    )
    tumor_df = pd.DataFrame(
        {
            "x": tumor_x,
            "y": tumor_y,
            "z": 0.0,
            "type": "tumor",
            "volume": "",
            "cycle entry": "",
            "custom:GFP": "",
            "custom:sample": "",
        }
    )

    cell1_x, cell1_y = generate_ellipse_points(
        n_cell_1,
        r1_cell1,
        r2_cell1,
        center=(center_x, center_y),
        jitter=jitter_cell_1,
        perimeter=True,
    )
    cell1_df = pd.DataFrame(
        {
            "x": cell1_x,
            "y": cell1_y,
            "z": 0.0,
            "type": "cell_1",
            "volume": "",
            "cycle entry": "",
            "custom:GFP": "",
            "custom:sample": "",
        }
    )

    return pd.concat([tumor_df, cell1_df], ignore_index=True)


def generate_oriented_ellipse(n, r1, r2, center, angle=0.0, jitter=5.0):
    """
    Generate n points in an oriented ellipse with radii r1, r2, centered at `center`,
    rotated by `angle` radians, with optional jitter.
    """
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n))
    x = radii * r1 * np.cos(angles)
    y = radii * r2 * np.sin(angles)

    # Rotation
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)

    # Translate to center and add jitter
    x_final = center[0] + x_rot + np.random.normal(0, jitter, n)
    y_final = center[1] + y_rot + np.random.normal(0, jitter, n)

    return x_final, y_final


def latin_hypercube_centers_2d(bounds, n_centers, seed=None):
    """
    Generate n_centers 2D points using Latin Hypercube Sampling.
    """
    if seed is not None:
        np.random.seed(seed)

    (x_min, x_max), (y_min, y_max) = bounds

    x_intervals = (
        np.linspace(0, 1, n_centers, endpoint=False)
        + np.random.rand(n_centers) / n_centers
    )
    y_intervals = (
        np.linspace(0, 1, n_centers, endpoint=False)
        + np.random.rand(n_centers) / n_centers
    )
    np.random.shuffle(y_intervals)

    x_coords = x_min + x_intervals * (x_max - x_min)
    y_coords = y_min + y_intervals * (y_max - y_min)

    centers = np.stack([x_coords, y_coords], axis=1)
    return centers


def generate_asymmetric_population(
    n_tumor_total,
    n_cell_1,
    x_min,
    x_max,
    y_min,
    y_max,
    n_tumors=3,
    tumor_radii_range=((25, 50), (15, 40)),
    jitter_tumor=5.0,
    cell1_radius_scale=2.0,
    cell1_jitter=10.0,
):
    """
    Generate multiple asymmetric (oriented) tumors using oriented ellipses,
    and surrounding cell_1 population.
    """

    # Tumor centers using LHS
    centers = latin_hypercube_centers_2d(
        bounds=((x_min, x_max), (y_min, y_max)), n_centers=n_tumors
    )

    tumor_dfs = []
    n_per_tumor = n_tumor_total // n_tumors

    for cx, cy in centers:
        # Sample ellipse radii
        r1 = np.random.uniform(*tumor_radii_range[0])
        r2 = np.random.uniform(*tumor_radii_range[1])
        angle = np.random.uniform(0, 2 * np.pi)

        # ---------------------------
        # FIX 3: shift center inward
        # ---------------------------
        x_low = cx - r1
        x_high = cx + r1
        y_low = cy - r2
        y_high = cy + r2

        if x_low < x_min:
            cx += x_min - x_low
        if x_high > x_max:
            cx -= x_high - x_max
        if y_low < y_min:
            cy += y_min - y_low
        if y_high > y_max:
            cy -= y_high - y_max

        # Now generate ellipse points
        x, y = generate_oriented_ellipse(
            n_per_tumor, r1, r2, (cx, cy), angle, jitter_tumor
        )

        # Store
        df = pd.DataFrame({"x": x, "y": y, "z": 0.0, "type": "tumor"})
        tumor_dfs.append(df)

    tumor_df = pd.concat(tumor_dfs, ignore_index=True)

    # Surrounding cell_1 (hex/circular/random) - here simple circular ring
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_width = (x_max - x_min) / 2
    half_height = (y_max - y_min) / 2
    r_outer = max(half_width, half_height) * cell1_radius_scale

    cell1_x = np.random.uniform(x_min, x_max, n_cell_1)
    cell1_y = np.random.uniform(y_min, y_max, n_cell_1)
    cell1_df = pd.DataFrame({"x": cell1_x, "y": cell1_y, "z": 0.0, "type": "cell_1"})

    return pd.concat([tumor_df, cell1_df], ignore_index=True)


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
            if np.hypot(x, y) <= max_radius:
                points.append((x, y))
    return np.array(points)


def generate_cell_positions(
    cell_radius=10,
    n_tumor=512,
    tumor_radius=200,
    n_layers=5,
    n_cell1=256,
    cell1_radius=512,
    gap=25.0,
):
    """
    Generate positions for tumor, other_tissue (hex lattice in circular ring), and cell_1 (random).
    """
    # Tumor
    tumor_x, tumor_y = [], []
    while len(tumor_x) < n_tumor:
        x, y = np.random.uniform(-tumor_radius, tumor_radius, 2)
        if x**2 + y**2 <= tumor_radius**2:
            tumor_x.append(x)
            tumor_y.append(y)
    tumor_df = pd.DataFrame({"x": tumor_x, "y": tumor_y, "z": 0.0, "type": "tumor"})

    # Other tissue
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
    drop_frac = np.random.uniform(0.45, 0.5)
    other_df = other_df.sample(frac=1 - drop_frac).reset_index(drop=True)

    # Cell_1
    cell1_x, cell1_y = [], []
    while len(cell1_x) < n_cell1:
        x, y = np.random.uniform(-cell1_radius, cell1_radius, 2)
        if np.hypot(x, y) > max_radius:
            cell1_x.append(x)
            cell1_y.append(y)
    cell1_df = pd.DataFrame({"x": cell1_x, "y": cell1_y, "z": 0.0, "type": "cell_1"})

    return pd.concat([tumor_df, other_df, cell1_df], ignore_index=True)


def generate_connected_tumor(
    n_cells_total,
    n_seeds=4,
    x_min=-256,
    x_max=256,
    y_min=-256,
    y_max=256,
    jitter=5.0,
    n_neighors=4,
):
    # Step 1: random seeds
    seeds = np.random.uniform(x_min, x_max, size=(n_seeds, 2))

    # Step 2: connect seeds using KNN edges (each seed to its nearest neighbor)
    nbrs = NearestNeighbors(n_neighbors=n_neighors).fit(seeds)
    distances, indices = nbrs.kneighbors(seeds)

    edges = []
    for i, neighbors in enumerate(indices):
        # neighbor[0] is self, neighbor[1] is nearest neighbor
        edges.append((i, neighbors[1]))

    # Step 3: populate cells along edges
    cells_per_edge = int(0.9 * n_cells_total) // len(edges)
    tumor_cells = []

    for i, j in edges:
        start = seeds[i]
        end = seeds[j]
        xs = np.linspace(start[0], end[0], cells_per_edge)
        ys = np.linspace(start[1], end[1], cells_per_edge)
        line_points = np.stack([xs, ys], axis=1)
        line_points += np.random.normal(0, jitter, size=line_points.shape)
        tumor_cells.append(line_points)

    tumor_cells = np.vstack(tumor_cells)
    n_seed_cells = (n_cells_total - int(0.9 * n_cells_total)) // n_cells_total
    # Step 4: optional: add small cluster around each seed
    for seed in seeds:
        seed_cells = seed + np.random.normal(0, jitter, size=(n_seed_cells, 2))
        tumor_cells = np.vstack([tumor_cells, seed_cells])

    df = pd.DataFrame(
        {"x": tumor_cells[:, 0], "y": tumor_cells[:, 1], "z": 0.0, "type": "tumor"}
    )
    return df


##################
# CSV & Plotting
##################


def create_csv(
    x_min,
    x_max,
    y_min,
    y_max,
    n_tumor,
    n_cell_1,
    range_jitter_tumor,
    range_cell_1,
    range_r2_frac_tumor,
    range_frac_cell_1,
    range_r1,
    range_cell_dist,
    csv_path,
    init_mode,
    cell_2_fraction=None,
):
    if isinstance(init_mode, list):
        init_mode = random.choice(init_mode)
    if cell_2_fraction is None:
        cell_2_fraction = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cell_2_fraction = (
        np.random.choice(cell_2_fraction)
        if isinstance(cell_2_fraction, (list, np.ndarray))
        else cell_2_fraction
    )

    if init_mode == "circular_mode":
        jitter_tumor = random.randint(*range_jitter_tumor)
        jitter_cell_1 = random.randint(*range_cell_1)
        r2_frac_tumor = random.uniform(*range_r2_frac_tumor)
        r2_frac_cell_1 = random.uniform(*range_frac_cell_1)
        r1 = random.uniform(*range_r1)
        cell_dist = random.uniform(*range_cell_dist)
        r1_cell1 = r1 * random.uniform(1.5, 1 / r1 - 0.2)
        df = generate_population_circulars(
            n_tumor,
            n_cell_1,
            x_min,
            x_max,
            y_min,
            y_max,
            tumor_scale=(r1, r2_frac_tumor),
            cell1_scale=(r1_cell1, r2_frac_cell_1 * cell_dist),
            jitter_tumor=jitter_tumor,
            jitter_cell_1=jitter_cell_1,
        )

    elif init_mode == "random_mode":
        tumor_df = pd.DataFrame(
            {
                "x": np.random.uniform(x_min, x_max, n_tumor),
                "y": np.random.uniform(y_min, y_max, n_tumor),
                "z": 0.0,
                "type": "tumor",
                "volume": "",
                "cycle entry": "",
                "custom:GFP": "",
                "custom:sample": "",
            }
        )
        cell1_df = pd.DataFrame(
            {
                "x": np.random.uniform(x_min, x_max, n_cell_1),
                "y": np.random.uniform(y_min, y_max, n_cell_1),
                "z": 0.0,
                "type": "cell_1",
            }
        )
        df = pd.concat([tumor_df, cell1_df], ignore_index=True)

    elif init_mode == "connected_mst_mode":
        n_seeds = random.randint(5, 15)
        df_tumor = generate_connected_tumor(
            n_cells_total=n_tumor,
            n_seeds=n_seeds,
            x_min=int(x_min * 0.9),
            x_max=int(x_max * 0.9),
            y_min=int(y_min * 0.9),
            y_max=int(y_max * 0.9),
            n_neighors=n_seeds - 2,
        )

        # cell_1 randomly around the whole zone
        cell1_df = pd.DataFrame(
            {
                "x": np.random.uniform(x_min, x_max, n_cell_1),
                "y": np.random.uniform(y_min, y_max, n_cell_1),
                "z": 0.0,
                "type": "cell_1",
            }
        )

        df = pd.concat([df_tumor, cell1_df], ignore_index=True)

    elif init_mode == "hex_mode":
        df = generate_cell_positions()

    elif init_mode == "asymmetric_mode":
        df = generate_asymmetric_population(
            n_tumor_total=n_tumor,
            n_cell_1=n_cell_1,
            x_min=int(x_min * 0.9),
            x_max=int(x_max * 0.9),
            y_min=int(y_min * 0.9),
            y_max=int(y_max * 0.9),
            n_tumors=random.randint(3, 12),
        )

    else:
        raise ValueError(f"Invalid init_mode: {init_mode}")

    # Convert some cell_1 to cell_2
    cell1_indices = df[df["type"] == "cell_1"].index
    n_to_change = int(cell_2_fraction * len(cell1_indices))
    if n_to_change > 0:
        indices_to_change = np.random.choice(cell1_indices, n_to_change, replace=False)
        df.loc[indices_to_change, "type"] = "cell_2"

    # Drop trailing empty columns
    while df.iloc[:, -1].isna().all() or (df.iloc[:, -1] == "").all():
        df = df.iloc[:, :-1]

    df.to_csv(csv_path, index=False, float_format="%.6f")


def generate_plot(df, path_title):
    fig, ax = plt.subplots(figsize=(8, 8))
    for t, c in zip(
        ["tumor", "other_tissue", "cell_1", "cell_2"],
        ["green", "orange", "blue", "red"],
    ):
        subset = df[df["type"] == t]
        ax.scatter(subset.x, subset.y, s=20, c=c, label=t, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_title("Cell positions (2D)")
    ax.legend()
    plt.savefig(path_title, dpi=300)
    plt.close()


##################
# Main
##################

if __name__ == "__main__":
    name_folder = "config_connected_mst_mode"
    os.makedirs(f"./{name_folder}", exist_ok=True)
    size = 256
    for i in range(20):
        create_csv(
            x_min=-size,
            x_max=size,
            y_min=-size,
            y_max=size,
            n_tumor=512,
            n_cell_1=128,
            range_jitter_tumor=[5, 15],
            range_cell_1=[5, 10],
            range_r2_frac_tumor=[0.1, 0.4],
            range_frac_cell_1=[0.1, 0.4],
            range_r1=[0.1, 0.4],
            range_cell_dist=[1.5, 2.0],
            cell_2_fraction=None,
            csv_path=f"./{name_folder}/df_{i}.csv",
            init_mode=["connected_mst_mode"],
        )
        df = pd.read_csv(f"./{name_folder}/df_{i}.csv")
        generate_plot(df, f"./{name_folder}/cells_{i}")
