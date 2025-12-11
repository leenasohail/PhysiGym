import numpy as np
import pandas as pd
from tysserand import tysserand as ty
from collections import defaultdict
import os
import shutil

from math import sqrt
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")


def generate_correlated_field(
    shape,
    correlation_length,
):
    """
    Generate a 2D Gaussian-correlated random field.

    Parameters
    ----------
    shape : tuple
        Field dimensions (height, width).
    correlation_length : float
        Spatial correlation length (Gaussian kernel std).

    Returns
    -------
    field : ndarray
        2D correlated scalar field.
    """

    noise = np.random.randn(*shape)
    sigma = correlation_length / sqrt(2)
    field = gaussian_filter(noise, sigma=sigma, mode="reflect")
    return field


def generate_balanced_fields(
    shape,
    dict_correlation_length,
    amplitude=1.0,
    local_noise_level=0.3,
):
    """
    Génère un champ global spatialement corrélé + un bruit local filtré pour chaque type.

    Parameters:
    -----------
    shape : tuple
        Dimensions du champ (height, width)
    cell_types : list
        Liste des types cellulaires
    correlation_length : float
        Longueur de corrélation spatiale
    amplitude : float
        Amplitude globale du champ
    local_noise_level : float
        Intensité du bruit local relatif

    Returns:
    --------
    fields : dict
        Dictionnaire {type_cellulaire: champ 2D}
    """

    local_noise_level = local_noise_level / 100 + 0.05

    fields = {}

    for key in dict_correlation_length.keys():
        correlation_length = dict_correlation_length[key]
        # === Génère un bruit local filtré ===
        local_noise = np.random.randn(*shape)
        filtered_noise = gaussian_filter(local_noise, sigma=correlation_length / 3)

        # === Champ final : base + bruit doux ===
        final_field = amplitude * (
            generate_correlated_field(shape, correlation_length)
            + local_noise_level * filtered_noise
        )
        min_final_field = np.min(final_field)
        max_final_field = np.max(final_field)
        fields[key] = (max_final_field - final_field) / (
            max_final_field - min_final_field
        )

    return fields


def weighted_pick(arr, threshold, n=1):
    # mask only the values above threshold
    mask = arr > threshold

    # get coordinates of valid pixels
    coords = np.argwhere(mask)

    # get the corresponding probabilities
    probs = arr[mask].astype(float)

    # normalize probabilities to sum to 1
    probs /= probs.sum()

    # weighted choice
    idx = np.random.choice(len(coords), size=n, p=probs, replace=False)

    return coords[idx]


def generate_synthetic_network_field(
    dict_correlation_length,
    domain_size,
    cell_types,
    number_cells,
    name_folder,
    dict_threshold,
    amplitude=1,
):
    # === Generate Fields ===
    fields = generate_balanced_fields(
        shape=domain_size,
        dict_correlation_length=dict_correlation_length,
        amplitude=amplitude,
    )
    xs_final = []
    ys_final = []
    phenotypes_final = []

    n_types = len(cell_types)
    fig, axes = plt.subplots(n_types, 2, figsize=(10, 5 * n_types))

    # Handle case of single row
    if n_types == 1:
        axes = np.array([axes])

    for row_idx, ct in enumerate(cell_types):
        field = fields[ct]
        coords = weighted_pick(field, threshold=dict_threshold[ct], n=number_cells[ct])
        xs = coords[:, 0]
        ys = coords[:, 1]
        # ========== LEFT: FIELD ==========
        ax_field = axes[row_idx, 0]
        im = ax_field.imshow(field, cmap="viridis")
        ax_field.set_title(f"Field: {ct}")
        ax_field.axis("off")
        fig.colorbar(im, ax=ax_field, fraction=0.046, pad=0.04)

        # ========== RIGHT: SCATTER CELLS ==========
        ax_scatter = axes[row_idx, 1]
        ax_scatter.scatter(ys, domain_size[1] - xs, s=10, alpha=0.8)
        ax_scatter.set_title(f"Cells: {ct}")
        ax_scatter.set_xlabel("X")
        ax_scatter.set_ylabel("Y")
        ax_scatter.set_aspect("equal")

        xs_final.extend(xs)  # extend, not append
        ys_final.extend(ys)
        phenotypes_final.extend([ct] * len(coords))  # repeat ct for each cell

    df_cells = pd.DataFrame(
        data={
            "X_position": xs_final,
            "Y_position": ys_final,
            "Phenotypes": phenotypes_final,
        }
    )
    plt.tight_layout()
    plt.savefig(f"{name_folder}_all.png", dpi=300)
    plt.close(fig)

    return df_cells


def network_field(
    number_cells,
    cell_types,
    name_folder,
    i,
    dict_correlation_length,
    dict_threshold,
    amplitude,
):
    np.random.seed(np.random.randint(0, 1000) + i)
    df_cells = generate_synthetic_network_field(
        dict_correlation_length=dict_correlation_length,
        number_cells=number_cells,
        domain_size=domain_size,
        cell_types=cell_types,
        amplitude=amplitude,
        name_folder=f"./{name_folder}/field_{i}",
        dict_threshold=dict_threshold,
    )

    df_cells = df_cells.rename(columns={"Phenotypes": "type"})
    df_cells.to_csv(f"./{name_folder}/df_{i}.csv", index=False)
    df_cells["PhenotypeID"] = df_cells["type"].astype("category").cat.codes

    plt.figure(figsize=(8, 8))

    scatter = plt.scatter(
        df_cells["X_position"],
        df_cells["Y_position"],
        c=df_cells["PhenotypeID"],
        cmap="tab10",  # Pick a nice categorical colormap
        alpha=0.8,
        s=20,
    )

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Cell Positions by Phenotype")

    # Add legend mapped to phenotype categories
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.8)
    plt.legend(handles, labels, title="Phenotypes")

    plt.gca().set_aspect("equal")
    plt.savefig(f"./{name_folder}/df_{i}.png")


if __name__ == "__main__":
    # === Global parameter ===

    # === Network parameter ===
    n_cells = 512 + 128
    domain_size = (512, 512)
    number_cells = {"tumor": 512, "cell_1": 128}
    cell_types = list(number_cells.keys())
    name_folder = "config_network_field"
    dict_correlation_length = {"tumor": 100, "cell_1": 50}
    dict_threshold = {"tumor": 0.55, "cell_1": 0.65}
    amplitude = 1
    os.makedirs(f"./{name_folder}", exist_ok=True)

    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    def wrapper(i):
        return network_field(
            number_cells,
            cell_types,
            name_folder,
            i,
            dict_correlation_length,
            dict_threshold,
            amplitude,
        )

    N = 10
    num_workers = cpu_count()  # or set manually

    with Pool(num_workers) as p:
        list(tqdm(p.imap_unordered(wrapper, range(N)), total=N))
