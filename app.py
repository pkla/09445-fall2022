import streamlit as st
import os
import matplotlib.pyplot as plt
from dataset import (
    get_cutoffs,
    load_h5_dataset,
    histogram,
    plot_annotated_histogram,
    ATOMIC_PAIRS,
    distances_from_coordinates,
    calculate_peaks,
)
import pandas as pd
import numpy as np

data_path = os.getenv("DATA_PATH", "../ANI-1ccx_clean_fullentry.h5")


@st.experimental_singleton(show_spinner=False)
def load_dataset_cached(data_path, n_formulas=None, average_by_empirical_formula=False):
    df = load_h5_dataset(data_path, n_formulas=n_formulas)
    df = df.set_index("mol", drop=True)

    df_dist = df.parallel_apply(
        lambda row: distances_from_coordinates(
            row["coordinates"], row["atomic_numbers"]
        ),
        axis=1,
        result_type="expand",
    )

    if average_by_empirical_formula:
        distances = df_dist.applymap(np.mean)
    else:
        distances = (
            df_dist.apply(np.concatenate, axis=0).apply(pd.Series).T
        )

    return df, distances


def main():
    st.set_page_config(page_title="ANI-1ccx Dataset", layout="wide")
    st.title("ANI-1ccx Dataset Explorer")

    with st.sidebar:
        nbins = st.number_input(
            "Number of Bins", value=100, min_value=1, max_value=1000, step=1
        )

        subset_data = st.checkbox("Subset Data", value=True)
        if subset_data:
            n_formulas = st.number_input(
                "Max Empirical Formulas", value=200, min_value=1, max_value=1000, step=1
            )
        else:
            n_formulas = None

        atomic_pairs = st.multiselect("Atomic Pairs", ATOMIC_PAIRS, ATOMIC_PAIRS)
        average_by_empirical_formula = st.checkbox(
            "Average Distances by Empirical Formula", value=False
        )

    if True or st.button("Load Data"):
        with st.spinner("Loading Data..."):
            df, distances = load_dataset_cached(data_path, n_formulas=n_formulas, average_by_empirical_formula=average_by_empirical_formula)

        with st.spinner("Calculating Histograms..."):
            histograms = {
                pair: histogram(distances[pair], bins=nbins) for pair in atomic_pairs
            }

        with st.spinner("Calculating Peaks..."):
            extrema = {
                pair: calculate_peaks(histograms[pair]["counts"], histograms[pair]["bins"])
                for pair in atomic_pairs
            }

        # fig, axes = plt.subplots(
        #     ncols=2 if len(histograms) > 1 else 1,
        #     nrows=math.ceil(len(histograms) / 2),
        #     figsize=(20, 20),
        # )

        tabs = st.tabs([str(pair) for pair in atomic_pairs])
        tab_empties = []
        for tab, (pair, hist) in zip(tabs, histograms.items()):
            with tab:
                tab_empties.append(st.empty())
               
        with st.spinner("Plotting Histograms..."):
            for tab_empty, (pair, hist) in zip(tabs, histograms.items()):
                fig, ax = plt.subplots(figsize=(10, 3))
                ax = plot_annotated_histogram(**hist, ax=ax)
                ax.set_title(f"{pair}", y=1.13, fontsize=20)
                plt.subplots_adjust(wspace=0.1, hspace=0.5)
                tab_empty.pyplot(fig)

        cutoffs = []
        for pair, (arg_minima, arg_maxima, val_minima, val_maxima) in extrema.items():
            lower, upper = get_cutoffs(arg_minima)
            cutoffs.append(f'"{pair}": [{lower:.5f}, {upper:.5f}]')

        st.code("cutoffs = {\n    " + "\n    ".join(cutoffs) + "\n}")
        # st.code(" \n"*100)


if __name__ == "__main__":
    main()
