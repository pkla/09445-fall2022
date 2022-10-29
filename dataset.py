#%%
import itertools
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from math import ceil

import h5py
import matplotlib.markers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandarallel import pandarallel
from scipy.spatial.distance import cdist
from tqdm import tqdm

pandarallel.initialize(progress_bar=True)

data_path = os.getenv("DATA_PATH", "../data/raw/ANI-1ccx_clean_fullentry.h5")

ATOMIC_PAIRS = [x for x in itertools.combinations_with_replacement([1, 6, 7, 8], 2)]
ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
}

average_by_empirical_formula = False
nbins = 100
n_formulas = 500


def load_h5_dataset(path, n_formulas=None, show_progress=True):
    dframes = []
    with h5py.File(path, "r") as f:
        iterator = itertools.islice(f.items(), n_formulas)

        if show_progress:
            iterator = tqdm(iterator, total=n_formulas)

        for empirical_formula, entry in iterator:
            coordinates = list([np.asarray(c) for c in entry["coordinates"]])
            atomic_numbers = list(entry["atomic_numbers"])
            dframes.append(
                pd.DataFrame(
                    {
                        "mol": np.array(
                            [empirical_formula] * len(coordinates), dtype=str
                        ),
                        "iconf": np.array(
                            list(range(len(coordinates))), dtype=np.int32
                        ),
                        "atomic_numbers": [atomic_numbers] * len(coordinates),
                        "coordinates": coordinates,
                    },
                )
            )
    return pd.concat(dframes).reset_index(drop=True)


def torch_cdist(x, y=None):
    x = torch.from_numpy(x)
    if y is None:
        return torch.cdist(x, x)
    else:
        y = torch.from_numpy(y)
        return torch.cdist(x, y)


def group_distances(distance_matrix, atomic_numbers):
    distances = {pair: [] for pair in ATOMIC_PAIRS}
    for i, j in zip(*np.triu_indices_from(distance_matrix, k=1)):
        atom_atom_distance = distance_matrix[i, j]
        atomic_number_pair = tuple(sorted((atomic_numbers[i], atomic_numbers[j])))
        distances[atomic_number_pair].append(atom_atom_distance)
    return distances


def distances_from_coordinates(coordinates, atomic_numbers):
    if len(coordinates.shape) == 2:
        pairwise_distances = [cdist(coordinates, coordinates)]
    else:
        pairwise_distances = torch_cdist(coordinates).numpy()

    if isinstance(atomic_numbers, np.ndarray) and len(atomic_numbers.shape) == 1:
        atomic_numbers = np.array([atomic_numbers] * len(coordinates))
    elif isinstance(atomic_numbers, list) and not isinstance(atomic_numbers[0], list):
        atomic_numbers = [atomic_numbers]

    all_distances = []
    for pairwise_distance, _atomic_numbers in zip(pairwise_distances, atomic_numbers):
        all_distances.append(group_distances(pairwise_distance, _atomic_numbers))

    if len(all_distances) == 1:
        return all_distances[0]
    else:
        return all_distances

def calculate_relative_extrema(counts, bins, order=None):
    idx_minima = scipy.signal.argrelmin(counts, order=order)[0]
    idx_maxima = scipy.signal.argrelmax(counts, order=order)[0]

    val_minima = np.asarray(counts[idx_minima])
    val_maxima = np.asarray(counts[idx_maxima])

    arg_minima = bins[idx_minima]
    arg_maxima = bins[idx_maxima]

    return arg_minima, arg_maxima, val_minima, val_maxima


def calculate_peaks(counts, bins, min_dist=None, **kwargs):
    if min_dist is not None:
        kwargs["distance"] = max(1, int(min_dist / (bins[1] - bins[0])))
    else:
        kwargs["distance"] = None

    idx_minima = scipy.signal.find_peaks(1-counts, **kwargs)[0]
    idx_maxima = scipy.signal.find_peaks(counts, **kwargs)[0]

    val_minima = np.asarray(counts[idx_minima])
    val_maxima = np.asarray(counts[idx_maxima])

    arg_minima = np.asarray(bins[idx_minima])
    arg_maxima = np.asarray(bins[idx_maxima])

    return arg_minima, arg_maxima, val_minima, val_maxima


def plot_extrema(arg_minima, arg_maxima, val_minima, val_maxima, ax):
    for arg, val in zip(arg_minima, val_minima):
        ax.axvline(
            arg,
            color="red",
            linestyle="dashed",
            ymax=val / ax.get_ylim()[1],
            linewidth=0.5,
        )
        ax.scatter(arg, val, color="red", marker= matplotlib.markers.CARETUP, clip_on=False)

    ax.set_xticks(arg_minima)
    ax.set_xticklabels([f"{x:.2f}" for x in arg_minima], rotation=90)

    top = ax.get_ylim()[1]
    top_text_pos = top * 1.01
    for arg, val in zip(arg_maxima, val_maxima):
        ax.axvline(
            arg,
            color="green",
            linestyle="dashed",
            ymin=val / ax.get_ylim()[1],
            linewidth=0.5,
        )
        ax.scatter(arg, val, color="green", marker= matplotlib.markers.CARETDOWN, clip_on=False)
        ax.text(arg, top_text_pos, f"{arg:.2f}", rotation=90, va="bottom", ha="center")

    return ax


def plot_boundaries(min_dist, max_dist, ax):
    ax.axvline(min_dist, color="red", linestyle="solid")
    ax.text(
        min_dist,
        ax.get_ylim()[1] / 2,
        f"min = {min_dist:.3f}",
        rotation=90,
        va="top",
        ha="right",
    )
    ax.axvline(max_dist, color="green", linestyle="solid")
    ax.text(
        max_dist,
        ax.get_ylim()[1] / 2,
        f"max = {max_dist:.3f}",
        rotation=90,
        va="top",
        ha="right",
    )
    return ax


def plot_hist(counts, bins, ax):
    if len(counts) == len(bins):
        counts = counts[:-1]
    lower = bins[:-1]
    upper = bins[1:]
    ax.hist(
        lower,
        upper,
        weights=counts,
        histtype="step",
        color="black",
    )
    return ax


class CutoffType(Enum):
    SHORT = "One Peak"
    MEDIUM = "Two Peaks"
    LONG = "Three Peaks"


def histogram(x, asdict=True, *args, **kwargs):
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    counts, bins = np.histogram(x, *args, **kwargs)
    counts = np.concatenate([counts, [np.nan]])

    if asdict:
        return {"counts": counts, "bins": bins}
    else:
        return pd.DataFrame({"counts": counts, "bins": bins})


def get_cutoffs(extrema, bins, cutoff_type: CutoffType = CutoffType.SHORT):
    arg_minima, arg_maxima, val_minima, val_maxima = extrema
    min_dist = np.min(bins)
    arg_minima = arg_minima[arg_maxima[0] < arg_minima]
    if cutoff_type.value == CutoffType.SHORT.value:
        cutoff = min_dist, arg_minima[0]
    elif cutoff_type.value == CutoffType.MEDIUM.value:
        cutoff = min_dist, arg_minima[min(arg_minima.shape[0] - 1, 1)]
    elif cutoff_type.value == CutoffType.LONG.value:
        cutoff = min_dist, arg_minima[min(arg_minima.shape[0] - 1, 2)]
    else:
        raise NotImplementedError("Unknown cutoff type")

    return cutoff


def plot_annotated_histogram(counts, bins, ax, extrema=None):
    if extrema is None:
        extrema = calculate_peaks(counts, bins)

    ax = plot_hist(counts, bins, ax)
    ax = plot_extrema(*extrema, ax)
    ax = plot_boundaries(bins.min(), bins.max(), ax)
    return ax


def load_h5_dataset_compact(path, targets=None):
    mol_counter = defaultdict(defaultdict)
    n_atom_counter = defaultdict(int)

    if targets is None:
        targets = []
    elif isinstance(targets, str):
        targets = [targets]

    # Scout the molecule sizes & number of conformations for each molecule
    with h5py.File(path, "r") as f:
        for empirical_formula, entry in f.items():
            num_conformers, num_atoms, _ = entry["coordinates"].shape
            n_atom_counter[num_atoms] += num_conformers
            mol_counter[num_atoms][empirical_formula] = num_conformers

    # Sort by number of atoms, then alphabetically by formula
    n_atom_counter = dict(sorted(n_atom_counter.items(), key=lambda item: item[0]))
    mol_counter = dict(sorted(mol_counter.items(), key=lambda item: item[0]))
    for key in mol_counter:
        mol_counter[key] = dict(sorted(mol_counter[key].items(), key=lambda item: item[0], reverse=True))

    # Preallocate the coordinate and target arrays
    coordinates = {n: np.empty((count, n, 3), dtype=np.float32) for n, count in n_atom_counter.items()}

    target_vals = {n: np.empty((count, len(targets)), dtype=np.float32) for n, count in n_atom_counter.items()}

    with h5py.File(path, "r") as f:
        for n_atoms, counter in mol_counter.items():
            start = 0
            for mol, n_conformers in counter.items():
                coordinates[n_atoms][start:start+n_conformers] = f[mol]["coordinates"]
                for i, target in enumerate(targets):
                    target_vals[n_atoms][start:start+n_conformers, i] = f[mol][target]

                start += n_conformers

    return mol_counter, coordinates, target_vals


@dataclass
class Dataset:
    mol_counts: dict
    coordinates: dict
    targets: dict

    # Load from file
    def __init__(self, path, targets=None):
        self.mol_counts, self.coordinates, self.targets = load_h5_dataset_compact(path, targets=targets)


def molecule_to_numbers(molecule: str) -> np.ndarray:
    counts = re.findall(r'\d+', molecule)
    elements = re.findall(r"[A-Za-z]+", molecule)
    numbers = [ATOMIC_NUMBERS[element] for element in elements]
    numbers = [[int(number)] * int(count) for count, number in zip(counts, numbers)]
    numbers = list(itertools.chain.from_iterable(numbers))
    return np.array(numbers, dtype=np.int8)


def get_formula_range(formula, mol_counts):
    atomic_numbers = molecule_to_numbers(formula)
    num_atoms = atomic_numbers.shape[-1]        
    start = mol_counts[num_atoms][formula]
    end = start + mol_counts[num_atoms][formula]
    return start, end


# %%
if __name__ == "__main__":
    df = load_h5_dataset(data_path, n_formulas=n_formulas)

    # %%
    df[ATOMIC_PAIRS] = df.parallel_apply(
        lambda row: distances_from_coordinates(
            row["coordinates"], row["atomic_numbers"]
        ),
        axis=1,
        result_type="expand",
    )

    # %%
    if average_by_empirical_formula:
        distances = df.set_index("mol")[ATOMIC_PAIRS].applymap(np.mean)
    else:
        distances = df[ATOMIC_PAIRS].apply(np.concatenate, axis=0).apply(pd.Series).T

    # %%
    ax = distances.hist(bins=nbins, figsize=(20, 20))
    plt.show()

    # %%

    histograms = {pair: histogram(distances[pair], bins=nbins) for pair in ATOMIC_PAIRS}

    extrema = {
        pair: calculate_peaks(histograms[pair]["counts"], histograms[pair]["bins"])
        for pair in ATOMIC_PAIRS
    }

    # %%

    fig, axes = plt.subplots(ncols=2, nrows=ceil(len(histograms) / 2), figsize=(20, 20))
    for ax, (pair, hist) in zip(axes.flatten(), histograms.items()):
        ax = plot_annotated_histogram(**hist, ax=ax, extrema=extrema[pair])
        ax.set_title(f"{pair}", y=1.13, fontsize=20)

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.show()

    # %%

    for pair, (arg_minima, arg_maxima, val_minima, val_maxima) in extrema.items():
        lower, upper = get_cutoffs(arg_minima)
        print(f"{pair}: [{lower:.4f}, {upper:.4f}]")

    # %%
