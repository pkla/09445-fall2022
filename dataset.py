#%%
from enum import Enum
from math import ceil
import h5py
import pandas as pd
import os
from scipy.spatial.distance import cdist
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

data_path = os.getenv("DATA_PATH", "../ANI-1ccx_clean_fullentry.h5")

ATOMIC_PAIRS = [x for x in itertools.combinations_with_replacement([1, 6, 7, 8], 2)]
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


def distances_from_coordinates(coordinates, atomic_numbers):
    pairwise_distance = cdist(coordinates, coordinates)
    distances = {pair: [] for pair in ATOMIC_PAIRS}
    for i, j in zip(*np.triu_indices_from(pairwise_distance, k=1)):
        atom_atom_distance = pairwise_distance[i, j]
        atomic_number_pair = tuple(sorted((atomic_numbers[i], atomic_numbers[j])))
        if atomic_number_pair in distances:
            distances[atomic_number_pair].append(atom_atom_distance)

    return distances


def calculate_relative_extrema(counts, bins, order=None):
    idx_minima = scipy.signal.argrelmin(counts, order=order)[0]
    idx_maxima = scipy.signal.argrelmax(counts, order=order)[0]

    val_minima = np.asarray(counts[idx_minima])
    val_maxima = np.asarray(counts[idx_maxima])

    arg_minima = bins[idx_minima]
    arg_maxima = bins[idx_maxima]

    return arg_minima, arg_maxima, val_minima, val_maxima


def calculate_peaks(counts, bins, **kwargs):
    idx_minima = scipy.signal.find_peaks(-counts, **kwargs)[0]
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
        ax.scatter(arg, val, color="red", marker="x")

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
        ax.scatter(arg, val, color="green", marker="x")
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
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


def histogram(x, asdict=True, *args, **kwargs):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    counts, bins = np.histogram(x, *args, **kwargs)
    counts = np.concatenate([counts, [np.nan]])
    if asdict:
        return {"counts": counts, "bins": bins}
    else:
        return pd.DataFrame({"counts": counts, "bins": bins})


def get_cutoffs(arg_minima, cutoff_type: CutoffType = CutoffType.SHORT):
    if cutoff_type == CutoffType.SHORT:
        cutoff = arg_minima[0], arg_minima[1]
    elif cutoff_type == CutoffType.MEDIUM:
        cutoff = arg_minima[0], arg_minima[2]
    elif cutoff_type == CutoffType.LONG:
        cutoff = arg_minima[0], arg_minima[3]
    else:
        raise NotImplementedError("Unknown cutoff type")

    return cutoff


def plot_annotated_histogram(counts, bins, ax):
    extrema = calculate_peaks(counts, bins)
    ax = plot_hist(counts, bins, ax)
    ax = plot_extrema(*extrema, ax)
    ax = plot_boundaries(bins.min(), bins.max(), ax)
    return ax


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
        ax = plot_annotated_histogram(**hist, ax=ax)
        ax.set_title(f"{pair}", y=1.13, fontsize=20)

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.show()

    # %%

    for pair, (arg_minima, arg_maxima, val_minima, val_maxima) in extrema.items():
        lower, upper = get_cutoffs(arg_minima)
        print(f"{pair}: [{lower:.4f}, {upper:.4f}]")

    # %%
