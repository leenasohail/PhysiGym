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


######################################################### HELPER FUNCTION #################################################################


def coords_to_pairs(coords):
    """
    Generate edge pairs from coordinates using Delaunay triangulation
    and links isolated nodes using additional criteria.

    Parameters
    ----------
    coords : ndarray
        Nx2 array of spatial coordinates.

    Returns
    -------
    pairs : list of tuple
        List of node index pairs representing edges.
    """

    pairs = ty.build_delaunay(coords)
    pairs = ty.link_solitaries(
        coords, pairs, method="delaunay", min_neighbors=15, verbose=0
    )
    return pairs


######################################################### USING FUNCTION #################################################################


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
    field = gaussian_filter(noise, sigma=correlation_length / sqrt(2), mode="reflect")
    return field


def build_lattice_indices(xs, ys, shape):
    """
    Create a lattice index matrix mapping spatial coordinates to cell indices.

    Parameters
    ----------
    xs : array-like
        X coordinates of cells.
    ys : array-like
        Y coordinates of cells.
    shape : tuple
        Shape of the output lattice (height, width).

    Returns
    -------
    lattice : ndarray of int
        2D array where each position holds the index of the corresponding cell, or -1 if empty.
    """

    lattice = -np.ones(shape, dtype=int)
    lattice[ys, xs] = np.arange(len(xs))
    return lattice


######################################################### FUNCTION CORE #################################################################


def estimate_correlation_length_fit(nodes, nb_bins=100, sample_size=80000):
    """
    Estimate the spatial correlation length by fitting exponential decay
    to phenotype similarity vs. distance.

    Parameters
    ----------
    nodes : DataFrame
        Node table with positions and phenotypes.
    nb_bins : int
        Number of bins for distance grouping.
    sample_size : int
        Number of random pairs sampled.

    Returns
    -------
    ξ_estimated : float
        Estimated correlation length (decay parameter).
    max_dist : float
        Maximum distance considered.
    """

    rng = np.random.default_rng(SEED)
    coords = nodes[["X_position", "Y_position"]].values
    labels = pd.get_dummies(nodes["Phenotypes"]).values
    n = len(coords)

    dists, sims, seen_pairs = [], [], set()

    while len(dists) < sample_size:
        i, j = rng.integers(0, n), rng.integers(0, n)
        if i == j or (i, j) in seen_pairs or (j, i) in seen_pairs:
            continue
        seen_pairs.add((i, j))
        d = np.linalg.norm(coords[i] - coords[j])
        s = np.dot(labels[i], labels[j])
        dists.append(d)
        sims.append(s)

    dists, sims = np.array(dists), np.array(sims)
    bins = np.linspace(0, dists.max(), nb_bins)
    bin_means, bin_edges, _ = binned_statistic(dists, sims, statistic="mean", bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = np.nan_to_num(bin_means, nan=0.0)

    def exp_decay(x, a, ξ):
        return a * np.exp(-x / ξ)

    popt, _ = curve_fit(exp_decay, bin_centers, bin_means, p0=(1.0, 50.0))

    ξ_estimated = popt[1]
    return ξ_estimated, dists.max()


def generate_balanced_fields(
    shape,
    cell_types,
    correlation_length,
    amplitude=1.0,
    local_noise_level=0.5,  ### between 0 and 1
    save_dir="FIELDS",
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
    save_dir : str
        Dossier où sauvegarder les champs générés

    Returns:
    --------
    fields : dict
        Dictionnaire {type_cellulaire: champ 2D}
    """
    local_noise_level = local_noise_level / 100 + 0.05

    os.makedirs(save_dir, exist_ok=True)
    fields = {}

    # === Génère un champ global corrélé ===
    base_field = generate_correlated_field(shape, correlation_length)

    for ct in tqdm(cell_types, desc="[PROCESS] Correlated Field per Type"):
        field_path = (
            f"{save_dir}/{ct}_field_corrl-{int(correlation_length)}_domain-{shape}.npy"
        )

        if os.path.isfile(field_path):
            fields[ct] = np.load(field_path)
        else:
            # === Génère un bruit local filtré ===
            local_noise = np.random.randn(*shape)
            filtered_noise = gaussian_filter(local_noise, sigma=correlation_length / 3)

            # === Champ final : base + bruit doux ===
            final_field = amplitude * (base_field + local_noise_level * filtered_noise)
            fields[ct] = final_field

            np.save(field_path, final_field)

    return fields


def gibbs_sampling_potts_field(
    labels_init,
    fields,
    lattice_indices,
    edges,
    beta,
    J,
    n_iter,
    cell_types,
    mixmat,
    verbose=False,
    apply_gibbs=True,
):
    """
    Perform Gibbs sampling using a Potts model with spatial field and cell-cell interaction matrix.

    Parameters
    ----------
    labels_init : array-like
        Initial cell type labels (strings).
    fields : dict
        Dictionary of {cell_type: 2D field array}.
    lattice_indices : ndarray
        Matrix mapping spatial positions to cell indices.
    edges : DataFrame
        DataFrame of graph edges with columns ['source', 'target'].
    beta : float
        Weight of the spatial field in energy calculation.
    J : float
        Global interaction strength scaling factor.
    n_iter : int
        Number of Gibbs iterations.
    cell_types : list
        List of cell types (labels).
    mixmat : DataFrame
        Normalized cell-cell interaction matrix.
    verbose : bool
        If True, print progress.
    apply_gibbs : bool
        If False, skip sampling and return initial labels.
    Returns
    -------
    final_labels : ndarray
        Final cell type labels after sampling.
    """

    n_types = len(cell_types)
    inv_cell_types = {ct: i for i, ct in enumerate(cell_types)}
    dict_cell_types = {i: ct for i, ct in enumerate(cell_types)}
    labels_int = np.array([inv_cell_types[label] for label in labels_init])

    xs, ys = np.where(lattice_indices != -1)
    n_cells = len(xs)
    field_stack = np.stack([fields[ct] for ct in cell_types], axis=-1)

    if not apply_gibbs:
        if verbose:
            print("[INFO] Gibbs sampling skipped — returning initial labels.")
        return np.array(labels_init)

    # === Carte des voisins basée sur les arêtes du graphe ===
    neighbor_map = defaultdict(list)
    for _, row in edges.iterrows():
        neighbor_map[row["source"]].append(row["target"])
        neighbor_map[row["target"]].append(row["source"])

    mixmat.apply(zscore)
    changes = []
    for it in tqdm(
        range(n_iter), desc="[PROCESS] Gibbs Sampling to balance phenotypes"
    ):
        old_labels = labels_int.copy()

        for idx in np.random.permutation(n_cells):
            neighbors = neighbor_map[idx]
            neighbor_labels = labels_int[neighbors] if neighbors else []

            log_probs = np.zeros(n_types)
            for k in range(n_types):
                # === Interaction Cells-Cells Energy ===
                interaction_energy = 0.0
                for label_j in neighbor_labels:
                    interaction_energy += (
                        J * mixmat.loc[dict_cell_types[k], dict_cell_types[label_j]]
                    )

                # === Fields Influence Energy ===
                x, y = xs[idx], ys[idx]
                field_energy = beta * field_stack[x, y, k]

                log_probs[k] = interaction_energy + field_energy

            # === Coversion in Probalities ===

            probs = np.exp(log_probs)
            probs /= np.sum(probs)

            # === Switch Phenotype ===

            labels_int[idx] = np.random.choice(n_types, p=probs)

        changes.append(np.sum(old_labels != labels_int))

        if verbose:
            changes = np.sum(old_labels != labels_int)
            tqdm.write(f"[Gibbs iteration {it + 1}] Changes: {changes[it]}")

    final_labels = np.array([cell_types[i] for i in labels_int])
    return final_labels


def generate_synthetic_network_potts_field(
    correlation_length,
    n_cells,
    domain_size,
    target_proportions,
    cell_types,
    mixmat,
    randomness_rate=0.8,
    beta=1.0,
    J=1.0,
    oversample_factor=10,
    n_iter=5,
    verbose=True,
    gibbs_sampling=True,
):
    """
    Generate a synthetic spatial network of cells using Potts field dynamics and correlated attraction.

    Parameters
    ----------
    correlation_length : float
        Correlation length
    n_cells : int
        Desired number of cells in the synthetic network.
    domain_size : tuple
        Size of the spatial domain (width, height).
    target_proportions : dict
        Desired cell type proportions.
    cell_types : list
        List of all possible cell types.
    max_dist_domain : float
        Max possible distance in the domain (for normalization).
    mixmat : DataFrame
        Interaction matrix for cell-cell affinity.
    randomness_rate : float
        rate to select random positions with correlated fields or by noise
        0.0 = random pick
        1.0 = pick only by correlated fields.
    beta : float
        Weight of the spatial field.
    J : float
        Scaling factor for cell-cell interaction energy.
    oversample_factor : int
        Number of candidates generated before selecting final n_cells.
    n_iter : int
        Number of Gibbs sampling iterations.
    gibbs_sampling : bool
        Whether to apply Gibbs sampling or not.
    Returns
    -------
    nodes : DataFrame
        Final node table with coordinates and phenotypes.
    edges : DataFrame
        Final edge list based on spatial proximity.
    fields : dict
        Correlated fields used for sampling.
    """
    # === Compute the correlation length ===

    tqdm.write(f"Correlation Lentgh Estimated = {correlation_length}")

    # === Generate Fields ===

    fields = generate_balanced_fields(domain_size, cell_types, correlation_length)

    # === Generate random point and compute score by using fields ===

    n_points = n_cells * oversample_factor
    xs = np.random.randint(0, domain_size[0], size=n_points)
    ys = np.random.randint(0, domain_size[1], size=n_points)

    scores = np.vstack([fields[ct][ys, xs] for ct in cell_types]).T
    assigned_types = np.full(n_points, fill_value=None, dtype=object)
    target_counts = {ct: int(p * n_cells) for ct, p in target_proportions.items()}
    remaining_indices = set(range(n_points))

    for i, ct in enumerate(tqdm(cell_types, desc="[PROCESS] Cell Assignation")):
        count = target_counts.get(ct, 0)
        if not remaining_indices or count == 0:
            continue
        subset = np.array(list(remaining_indices))
        raw_scores = scores[subset, i]

        # Rescale les scores pour avoir des probabilités positives
        min_score = raw_scores.min()
        scaled_scores = raw_scores - min_score

        if scaled_scores.sum() == 0:
            probabilities = np.ones_like(scaled_scores) / len(scaled_scores)
        else:
            probabilities = scaled_scores / scaled_scores.sum()

        # Choix principal par scores pondérés
        num_main = int(count * randomness_rate)
        num_random = count - num_main

        chosen_main = np.random.choice(
            subset, size=min(num_main, len(subset)), replace=False, p=probabilities
        )

        # Choix aléatoire bruité
        remaining_for_noise = list(set(subset) - set(chosen_main))
        if len(remaining_for_noise) < num_random:
            num_random = len(remaining_for_noise)
        chosen_noise = np.random.choice(
            remaining_for_noise, size=num_random, replace=False
        )

        chosen_indices = np.concatenate([chosen_main, chosen_noise])
        assigned_types[chosen_indices] = ct
        remaining_indices -= set(chosen_indices)

    keep_indices = np.where(assigned_types != None)[0]
    if len(keep_indices) > n_cells:
        keep_indices = keep_indices[:n_cells]

    xs_final = xs[keep_indices]
    ys_final = ys[keep_indices]
    assigned_types = np.array(assigned_types)[keep_indices]
    """
    updated_labels = gibbs_sampling_potts_field(
        labels_init=assigned_types,
        fields=fields,
        lattice_indices=lattice_indices,
        edges=edges,
        beta=beta,
        J=J,
        n_iter=n_iter,
        cell_types=cell_types,
        mixmat=mixmat,
        verbose=verbose,
        apply_gibbs=gibbs_sampling,
    )
    """

    df_cells = pd.DataFrame(
        {
            "X_position": xs_final,
            "Y_position": ys_final,
            "Phenotypes": assigned_types,
        }
    )

    return df_cells


######################################################### MAIN #################################################################
def main():
    """
    Main function to load real data, compute fields, generate a synthetic spatial network,
    and optionally visualize results.

    This function:
    - Loads input nodes and edges
    - Computes correlation length
    - Builds spatially correlated fields
    - Performs Gibbs sampling with Potts model
    - Computes and visualizes assortativity
    - Saves plots and synthetic network
    """

    global panel
    # max_dist_domain = (domain_size[0] ** 2 + domain_size[1] ** 2) ** (1 / 2)
    np.random.seed(SEED)

    ################################### Import Data ###########################

    target_proportions = {"cell_1": 512 / (512 + 128), "tumor": 128 / (512 + 128)}

    cell_types = ["cell_1", "tumor"]
    mixmat_inital = None

    ################################### RUN ###################################
    df_cells = generate_synthetic_network_potts_field(
        correlation_length=0.8,
        n_cells=nb_cells,
        domain_size=domain_size,
        target_proportions=target_proportions,
        cell_types=cell_types,
        mixmat=mixmat_inital,
        randomness_rate=0.8,
        beta=1.0,
        J=1.0,
        oversample_factor=10,
        n_iter=5,
        verbose=True,
        gibbs_sampling=True,
    )
    df_cells["PhenotypeID"] = df_cells["Phenotypes"].astype("category").cat.codes

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
    print("Done")
    plt.show()


if __name__ == "__main__":
    # === Global parameter ===
    import random

    SEED = random.randint(0, 100)
    RUN_TEST = False
    FIELD_PLOT = False

    # === Gibbs Sampling parameters ===
    gibbs_sampling = True
    iter_Gibbs = 50
    verbose = True

    # === Network parameter ===
    nb_cells = 512 + 128
    domain_size = (512, 512)

    # === Pott parameter ===
    J = 1.0
    beta = 0.5

    # === MAIN ===
    print("\n")
    main()
