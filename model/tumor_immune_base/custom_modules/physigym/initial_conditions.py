import numpy as np
import pandas as pd
import random
from typing import Union

##################
# Initial States #
##################


def generate_cells_2d_ellipse(n, r1, r2, center, jitter=10.0):
    """Generate random 2D points within an ellipse centered at `center` with semi-axes r1 (x), r2 (y)."""
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, 1, n))  # uniform distribution in area
    x = center[0] + radii * r1 * np.cos(angles) + np.random.normal(0, jitter, n)
    y = center[1] + radii * r2 * np.sin(angles) + np.random.normal(0, jitter, n)
    return x, y


def generate_ellipse_ring(n, r1, r2, center, jitter=5.0):
    """Generate points along an elliptical ring with semi-axes r1 and r2."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + r1 * np.cos(angles) + np.random.normal(0, jitter, n)
    y = center[1] + r2 * np.sin(angles) + np.random.normal(0, jitter, n)
    return x, y


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
    """
    Generate tumor and cell_1 cells in ellipses within (x_min, x_max, y_min, y_max).

    `tumor_scale` and `cell1_scale` are fractional sizes (relative to width/height).
    """
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_width = (x_max - x_min) / 2
    half_height = (y_max - y_min) / 2

    # Compute ellipse radii for tumor and cell_1 regions
    r1_tumor = tumor_scale[0] * half_width
    r2_tumor = tumor_scale[1] * half_height

    r1_cell1 = cell1_scale[0] * half_width
    r2_cell1 = cell1_scale[1] * half_height

    # Tumor cells inside ellipse
    tumor_x, tumor_y = generate_cells_2d_ellipse(
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

    # Surrounding cells in elliptical ring
    cell1_x, cell1_y = generate_ellipse_ring(
        n_cell_1, r1_cell1, r2_cell1, center=(center_x, center_y), jitter=jitter_cell_1
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


def create_csv(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    n_tumor: int,
    n_cell_1: int,
    range_jitter_tumor: list,
    range_cell_1: list,
    range_r2_frac_tumor: list,
    range_frac_cell_1: list,
    range_r1: list,
    range_cell_dist: list,
    cell_2_fraction: Union[list, float],
    csv_path: str,
    init_mode: str,
    **kwargs,
):
    if init_mode not in ["robust", "circular_mode", "random_mode", "hex_mode"]:
        raise ValueError("Problem with mode")
    if init_mode == "robust":
        init_mode = random.choice(["circular_mode", "random_mode", "hex_mode"])
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
            n_tumor=n_tumor,
            n_cell_1=n_cell_1,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            tumor_scale=(r1, r2_frac_tumor),
            cell1_scale=(
                r1_cell1,
                r2_frac_cell_1 * cell_dist,
            ),
            jitter_tumor=jitter_tumor,
            jitter_cell_1=jitter_cell_1,
        )
    elif init_mode == "random_mode":
        # Tumor cells randomly in box
        tumor_x = np.random.uniform(x_min, x_max, n_tumor)
        tumor_y = np.random.uniform(y_min, y_max, n_tumor)
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

        # Cell_1 cells randomly in box
        cell1_x = np.random.uniform(x_min, x_max, n_cell_1)
        cell1_y = np.random.uniform(y_min, y_max, n_cell_1)
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

        df = pd.concat([tumor_df, cell1_df], ignore_index=True)
    elif init_mode == "hex_mode":
        df = generate_cell_positions()
    else:
        raise ValueError("Problem with mode")

    mask = df["type"] == "cell_1"
    # Get indices of those rows
    cell1_indices = df[mask].index

    # Randomly select 50% of them
    n_to_change = int(cell_2_fraction * len(cell1_indices))
    indices_to_change = np.random.choice(cell1_indices, n_to_change, replace=False)

    # Change type to "cell_2"
    df.loc[indices_to_change, "type"] = "cell_2"
    # Drop trailing all-empty columns
    while df.iloc[:, -1].isna().all() or (df.iloc[:, -1] == "").all():
        df = df.iloc[:, :-1]
    # fname = f"ellipse_r1_{r1:.2f}_r2_frac_cell_1_{r2_frac_cell_1:.2f}_r2_frac_tumor_{r2_frac_tumor:.2f}_cell_dist_{cell_dist:.2f}_jitter_tumor_{jitter_tumor:.2f}_jitter_cell_1_{jitter_cell_1:.2f}"
    # csv_path = f"./config/{fname}.csv"
    # Save without trailing empty fields
    df.to_csv(csv_path, index=False, float_format="%.6f")


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


def generate_plot(df, path_title):
    fig, ax = plt.subplots(figsize=(8, 8))
    for t, c in zip(
        ["tumor", "other_tissue", "cell_1", "cell_2"],
        ["green", "orange", "blue", "red"],
    ):
        subset = df[df["type"] == t]
        print(t)
        ax.scatter(subset.x, subset.y, s=20, c=c, label=t, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_title("Cell positions (2D)")
    ax.legend()
    plt.savefig(path_title, dpi=300)
    plt.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = generate_cell_positions()

    import os

    os.makedirs("./config_2", exist_ok=True)
    for i in range(10):
        create_csv(
            x_min=-512,
            x_max=512,
            y_min=-512,
            y_max=512,
            n_tumor=1024,
            n_cell_1=128,
            range_jitter_tumor=[5, 15],
            range_cell_1=[5, 10],
            range_r2_frac_tumor=[0.1, 0.4],
            range_frac_cell_1=[0.1, 0.4],
            range_r1=[0.1, 0.4],
            range_cell_dist=[1.5, 2.0],
            csv_path=f"./config_2/df_{i}.csv",
            cell_2_fraction=0.3,
            init_mode="robust",
        )
        df = pd.read_csv(f"./config_2/df_{i}.csv")
        generate_plot(df, f"./config_2/cells_{i}")
