import numpy as np
import pandas as pd
import random
from typing import Union
import matplotlib.pyplot as plt
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


def generate_asymmetric_population(
    n_foci=3,
    n_cells_per_tumor=200,
    bounds=(-512, 512, -512, 512),
    tumor_radii_range=((50, 120), (30, 80)),
    jitter=10.0,
    min_distance=100,
):
    """Generate multiple asymmetric / elongated tumor foci."""
    x_min, x_max, y_min, y_max = bounds
    centers = []

    attempts = 0
    while len(centers) < n_foci and attempts < n_foci * 20:
        cx = np.random.uniform(x_min, x_max)
        cy = np.random.uniform(y_min, y_max)
        if all(np.hypot(cx - c[0], cy - c[1]) >= min_distance for c in centers):
            centers.append((cx, cy))
        attempts += 1

    dfs = []
    for cx, cy in centers:
        r1 = np.random.uniform(*tumor_radii_range[0])
        r2 = np.random.uniform(*tumor_radii_range[1])
        angle = np.random.uniform(0, 360)
        x, y = generate_ellipse_points(
            n_cells_per_tumor,
            r1,
            r2,
            center=(cx, cy),
            orientation_deg=angle,
            jitter=jitter,
        )
        dfs.append(
            pd.DataFrame(
                {
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "type": "tumor",
                    "volume": "",
                    "cycle entry": "",
                    "custom:GFP": "",
                    "custom:sample": "",
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


##################
# Hex & Cell Positions
##################


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
    drop_frac = np.random.uniform(0.25, 0.5)
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

    elif init_mode == "hex_mode":
        df = generate_cell_positions()

    elif init_mode == "asymmetric_mode":
        n_foci = random.randint(3, 8)
        tumor_df = generate_asymmetric_population(
            n_foci=n_foci,
            n_cells_per_tumor=n_tumor // n_foci,
            bounds=(x_min, x_max, y_min, y_max),
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
    os.makedirs("./config_2", exist_ok=True)
    for i in range(20):
        create_csv(
            x_min=-256,
            x_max=256,
            y_min=-256,
            y_max=256,
            n_tumor=512,
            n_cell_1=128,
            range_jitter_tumor=[5, 15],
            range_cell_1=[5, 10],
            range_r2_frac_tumor=[0.1, 0.4],
            range_frac_cell_1=[0.1, 0.4],
            range_r1=[0.1, 0.4],
            range_cell_dist=[1.5, 2.0],
            csv_path=f"./config_2/df_{i}.csv",
            init_mode=["asymmetric_mode", "hex_mode", "random_mode", "circular_mode"],
        )
        df = pd.read_csv(f"./config_2/df_{i}.csv")
        generate_plot(df, f"./config_2/cells_{i}")
