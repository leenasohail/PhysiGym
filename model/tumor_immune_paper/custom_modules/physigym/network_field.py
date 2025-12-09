import numpy as np
import pandas as pd
from tysserand import tysserand as ty
from collections import defaultdict
import os
import shutil

from math import sqrt
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.stats import zscore
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    cell_types,
    correlation_length,
    amplitude=1.0,
    local_noise_level=0.0,
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

    # === Génère un champ global corrélé ===
    base_field = generate_correlated_field(shape, correlation_length)

    for ct in cell_types:
        # === Génère un bruit local filtré ===
        local_noise = np.random.randn(*shape)
        filtered_noise = gaussian_filter(local_noise, sigma=correlation_length / 3)

        # === Champ final : base + bruit doux ===
        final_field = amplitude * (base_field + local_noise_level * filtered_noise)
        fields[ct] = final_field

    return fields


def generate_synthetic_network_potts_field(
    correlation_length,
    n_cells,
    domain_size,
    target_proportions,
    randomness_rate=0.2,
    oversample_factor=10,
):
    """
    Generate a synthetic spatial network of cells using correlated fields.

    Parameters
    ----------
    nodes_initial : DataFrame
        Input nodes from real data, with coordinates and phenotype info.
    X_col : str
        Name of the column containing X coordinates in nodes_initial.
    Y_col : str
        Name of the column containing Y coordinates in nodes_initial.
    label_col : str
        Name of the column containing phenotypes / cell types in nodes_initial.
    n_cells : int
        Desired number of cells in the synthetic network.
    domain_size : tuple
        Size of the spatial domain (width, height).
    cell_types : list
        List of all possible cell types.
    randomness_rate : float
        Fraction of cells chosen according to the correlated fields.
        0.0 -> choix entièrement aléatoire.
        1.0 -> choix entièrement guidé par les champs.
    oversample_factor : int
        Multiplicative factor used to generate candidate positions.
    output_csv_path : str
        Path where the CSV file will be saved. If None, no file is written.

    Returns
    -------
    df_positions : DataFrame
        DataFrame with columns ['X_position', 'Y_position', 'Phenotypes'].
    """

    # === Generate Fields ===
    fields = generate_balanced_fields(domain_size, cell_types, correlation_length)

    # === Generate random points and compute scores from fields ===
    n_points = n_cells * oversample_factor
    xs = np.random.randint(0, domain_size[0], size=n_points)
    ys = np.random.randint(0, domain_size[1], size=n_points)

    # scores[i, j] = valeur du champ du type j au point i
    scores = np.vstack([fields[ct][ys, xs] for ct in cell_types]).T

    assigned_types = np.full(n_points, fill_value=None, dtype=object)
    target_counts = {ct: int(p * n_cells) for ct, p in target_proportions.items()}
    remaining_indices = set(range(n_points))

    # === Assignation des types cellule par cellule ===
    for i, ct in enumerate(tqdm(cell_types, desc="[PROCESS] Cell Assignation")):
        count = target_counts.get(ct, 0)
        if not remaining_indices or count == 0:
            continue

        subset = np.array(list(remaining_indices))
        raw_scores = scores[subset, i]

        # Mise à l'échelle pour obtenir des probabilités positives
        min_score = raw_scores.min()
        scaled_scores = raw_scores - min_score

        if scaled_scores.sum() == 0:
            probabilities = np.ones_like(scaled_scores) / len(scaled_scores)
        else:
            probabilities = scaled_scores / scaled_scores.sum()

        # Choix guidé par le champ
        num_main = int(count * randomness_rate)
        num_random = count - num_main

        num_main = min(num_main, len(subset))
        chosen_main = np.random.choice(
            subset,
            size=num_main,
            replace=False,
            p=probabilities if num_main > 0 else None,
        )

        # Choix aléatoire de bruit
        remaining_for_noise = list(set(subset) - set(chosen_main))
        if len(remaining_for_noise) < num_random:
            num_random = len(remaining_for_noise)

        if num_random > 0:
            chosen_noise = np.random.choice(
                remaining_for_noise, size=num_random, replace=False
            )
            chosen_indices = np.concatenate([chosen_main, chosen_noise])
        else:
            chosen_indices = chosen_main

        assigned_types[chosen_indices] = ct
        remaining_indices -= set(chosen_indices)

    assigned_types = np.array(assigned_types)
    keep_indices = np.where(assigned_types != None)[0]

    if len(keep_indices) > n_cells:
        keep_indices = keep_indices[:n_cells]

    xs_final = xs[keep_indices]
    ys_final = ys[keep_indices]
    phenotypes_final = assigned_types[keep_indices]

    df_cells = pd.DataFrame(
        {
            "X_position": xs_final,
            "Y_position": ys_final,
            "Phenotypes": phenotypes_final,
        }
    )

    return df_cells


def network_field(n_cells, target_proportions, cell_types, path, i):
    np.random.seed(np.random.randint(0, 1000))
    correlation_length = 100
    randomness_rate = np.random.uniform(0.001, 0.99)
    beta = np.random.uniform(0.01, 5)
    J = np.random.uniform(0.01, 5)
    n_iter = np.random.randint(1, 20)

    df_cells = generate_synthetic_network_potts_field(
        correlation_length=correlation_length,
        n_cells=n_cells,
        domain_size=domain_size,
        target_proportions=target_proportions,
        randomness_rate=0.00,
        oversample_factor=10,
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
    import random

    SEED = random.randint(0, 100)

    # === Network parameter ===
    n_cells = 512 + 128
    domain_size = (512, 512)
    target_proportions = {"cell_1": 512 / (n_cells), "tumor": 128 / (n_cells)}
    cell_types = list(target_proportions.keys())
    name_folder = "config_network_field"
    os.makedirs(f"./{name_folder}", exist_ok=True)

    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    def wrapper(i):
        return network_field(
            n_cells,
            target_proportions,
            cell_types,
            name_folder,
            i,
        )

    N = 100
    num_workers = cpu_count()  # or set manually

    with Pool(num_workers) as p:
        list(tqdm(p.imap_unordered(wrapper, range(N)), total=N))
