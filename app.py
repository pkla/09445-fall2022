import os
os.chdir(os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dataset import (
    ATOMIC_PAIRS,
    calculate_peaks,
    torch_cdist,
    distances_from_coordinates,
    group_distances,
    get_cutoffs,
    histogram,
    load_h5_dataset,
    plot_annotated_histogram,
    CutoffType,
)
import hashlib
import pickle
import itertools

os.chdir(os.path.join(os.path.dirname(__file__), "DFTBrepulsive"))
from data.dataset import AniDataset
from generator import Generator
from model import RepulsiveModel
from options import Options
from consts import BCONDS
os.chdir(os.path.dirname(__file__))

data_path = os.getenv("DATA_PATH", "../data/raw/ANI-1ccx_clean_fullentry.h5")
data_path = os.path.abspath(data_path)

@st.cache(show_spinner=False, persist=True, suppress_st_warning=True, allow_output_mutation=True)
def load_dataset_cached(data_path, n_formulas=None, average_by_empirical_formula=False, max_heavy_atoms=None):
    df = load_h5_dataset(data_path, n_formulas=n_formulas)
    df = df.set_index("mol", drop=True)
    df["num_heavy_atoms"] = df["atomic_numbers"].parallel_apply(lambda x: len(x) - np.sum(np.array(x) == 1))

    if max_heavy_atoms is not None:
        df = df.query("num_heavy_atoms <= @max_heavy_atoms")

    df_dist = df.parallel_apply(
        lambda row: group_distances(
            torch_cdist(row["coordinates"]).numpy(), row["atomic_numbers"]
        ),
        axis=1,
        result_type="expand",
    )

    if average_by_empirical_formula:
        distances = df_dist.applymap(np.mean)
    else:
        distances = df_dist.apply(np.concatenate, axis=0).apply(pd.Series).T

    return df, distances


def label_coefs(nknots, deg, bconds, atoms=("H", "C", "N", "O")):
    pairs = list(itertools.combinations_with_replacement(atoms, 2))
    vals_per_pair = (nknots + (deg - 1)) - len(bconds)

    labels = []
    for pair in pairs:
        for i in range(vals_per_pair):
            labels.append((f"{pair[0]}-{pair[1]}", i))

    labels.append(("ref", "const"))
    labels.extend([("ref", a) for a in atoms])

    return labels

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]



def main():
    st.set_page_config(page_title="ANI-1ccx Dataset", layout="wide")
    st.title("ANI-1ccx Dataset Explorer")

    with st.sidebar:
        st.subheader("Dataset")
        subset_data = False #st.checkbox("Subset Data", value=False)
        if subset_data:
            n_formulas = st.number_input(
                "Max Empirical Formulas", value=600, min_value=1, max_value=1000, step=1
            )
        else:
            n_formulas = None

        atomic_pairs = st.multiselect("Atomic Pairs", ATOMIC_PAIRS, ATOMIC_PAIRS)
        average_by_empirical_formula = False 
        # st.checkbox(
        #     "Average Distances by Empirical Formula", value=False
        # )

        max_heavy_atoms = st.number_input("Max Heavy Atoms", value=20, min_value=1, max_value=20, step=1)

        st.subheader("Histogram")
        nbins = st.number_input(
            "Histogram Bins", value=125, min_value=1, max_value=1000, step=1
        )

        st.subheader("Peak Finding")
        min_height = st.number_input("Minimum Height", value=0.1, min_value=0.0, max_value=1.0, step=0.01)
        min_dist = st.number_input("Minimum Horizontal Distance", value=0.2, min_value=0.0, max_value=1.0, step=0.001, format="%.3f",  help="Required minimal horizontal distance between neighbouring peaks in Angstrom")
        prominence = st.number_input("Prominence", value=0.005, min_value=0.0, max_value=1.0, step=0.001, format="%.3f", help="The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.")

    if True or st.button("Load Data"):
        with st.spinner("Loading Data..."):
            df, distances = load_dataset_cached(
                data_path,
                n_formulas=n_formulas,
                average_by_empirical_formula=average_by_empirical_formula,
                max_heavy_atoms=max_heavy_atoms,
            )

        number_of_formulas = len(df.index.unique())
        number_of_conformations = len(df)
        
        # Descriptive stats about the ANI-1ccx dataset
        st.write(f"Number of Empirical Formulas: {number_of_formulas}")
        st.write(f"Number of Conformations: {number_of_conformations}")

        histograms = {
            pair: histogram(distances[pair], bins=nbins, density=True) for pair in atomic_pairs
        }

        extrema = {
            pair: calculate_peaks(
                histograms[pair]["counts"], histograms[pair]["bins"], min_dist=min_dist, height=min_height, prominence=prominence
            )
            for pair in atomic_pairs
        }

        # check if there are any peaks
        no_minima = [str(x) for x in itertools.compress(atomic_pairs, [not len(x[0]) for x in extrema.values()])]
        no_maxima = [str(x) for x in list(itertools.compress(atomic_pairs, [not len(x[1]) for x in extrema.values()]))]

        if no_minima:
            st.warning(f"No minima found for {', '.join(no_minima)}")

        if no_maxima:
            st.warning(f"No maxima found for {', '.join(no_maxima)}")

        # fig, axes = plt.subplots(
        #     ncols=2 if len(histograms) > 1 else 1,
        #     nrows=math.ceil(len(histograms) / 2),
        #     figsize=(20, 20),
        # )

        if not atomic_pairs:
            st.stop()

        with st.expander("Interatomic Distances and Cutoffs", expanded=True):
            tabs = st.tabs([str(pair) for pair in atomic_pairs])
            tab_empties = []
            for tab, (pair, hist) in zip(tabs, histograms.items()):
                with tab:
                    tab_empties.append(st.empty())

            cutoff_type = st.selectbox("Cutoff Type", CutoffType, format_func=lambda x: x.name)
            cols = st.columns(2)
            lower_bound_add = cols[0].number_input("Add to Lower Bound", value=0.02, min_value=-1.0, max_value=1.0, step=0.01, format="%.3f")
            upper_bound_add = cols[1].number_input("Add to Upper Bound", value=0.00, min_value=-1.0, max_value=1.0, step=0.01, format="%.3f")
            cutoff_strs = []
            cutoffs = {}
            for pair, pair_extrema in extrema.items():
                lower, upper = get_cutoffs(pair_extrema, histograms[pair]["bins"], cutoff_type=cutoff_type)
                lower += lower_bound_add
                upper += upper_bound_add
                cutoff_strs.append(f"{pair}: ({lower:.5f}, {upper:.5f})")
                cutoffs[pair] = (lower, upper)

            st.code("cutoffs = {\n    " + ",\n    ".join(cutoff_strs) + "\n}")

            with st.spinner("Plotting Histograms..."):
                for tab_empty, (pair, hist) in zip(tabs, histograms.items()):
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax = plot_annotated_histogram(**hist, ax=ax, extrema=extrema[pair])
                    ax.set_title(f"{pair}", y=1.13, fontsize=20)
                    # plt.subplots_adjust(wspace=0.1, hspace=0.5)
                    tab_empty.pyplot(fig)
                    tab_empty.caption(f"Distribution of {pair}-Pairwise Interatomic Distances")

        with st.expander("Summary", expanded=True):
            # Bar chart of heavy atom counts
            fig = df["num_heavy_atoms"].value_counts().sort_index().plot.bar(backend="plotly", title="Heavy Atom Counts").update_layout(
                showlegend=False, xaxis_title="Heavy Atom Count", yaxis_title="Number of Conformations", 
            )
            fig.update_traces(marker_color="rgb(158,202,225)", marker_line_color="rgb(8,48,107)", marker_line_width=1.5, opacity=0.8)
            st.plotly_chart(fig)

            # Bar chart of atoms counts
            fig =  df["atomic_numbers"].apply(len).value_counts().sort_index().plot.bar(backend="plotly", title="Atom Counts").update_layout(
                showlegend=False, xaxis_title="Atom Count", yaxis_title="Number of Conformations", 
            )
            fig.update_traces(marker_color="rgb(158,202,225)", marker_line_color="rgb(8,48,107)", marker_line_width=1.5, opacity=0.8)
            st.plotly_chart(fig)

            # Bar chart of empirical formulas
            fig = df.index.value_counts().plot(kind="bar", title="Number of Conformations per Empirical Formula", backend="plotly").update_layout(showlegend=False, xaxis_title="Empirical Formula", yaxis_title="Number of Conformations")
            fig.update_traces(marker_color="rgb(158,202,225)", marker_line_color="rgb(8,48,107)", marker_line_width=0.1, opacity=1)
            st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        st.subheader("Model Options")
        nknots = st.number_input("Number of Knots", value=20, min_value=1, max_value=100, step=1)
        deg = st.number_input("Degree", value=3, min_value=1, max_value=10, step=1)
        maxder = st.number_input("Max Derivative", value=2, min_value=0, max_value=10, step=1)
        bconds = st.selectbox("Boundary Conditions", BCONDS.keys(), index=2)
        constr = tuple(st.text_input("Constraints", "-1,+2").split(","))
        ngrid = st.number_input("Number of Grid Points", value=500, min_value=1, max_value=10000, step=1)

    if st.button("Train Model"):
        opts_dict = {
            "zs": tuple(atomic_pairs),
            "model": {
                "model": "spline",
                "nknots": nknots,
                "cutoffs": cutoffs,
                "deg": deg,
                "maxder": maxder,
                "bconds": bconds,
            },
            "constr": {
                "constr": constr,
                "ngrid": ngrid,
            },
        }

        os.chdir(os.path.join(os.path.dirname(__file__), "DFTBrepulsive"))
        cache_filename = hashlib.sha256(str(opts_dict).encode("utf-8")).hexdigest() + ".pkl"

        if os.path.exists(cache_filename):
            with st.spinner("Loading cached model..."):
                with open(cache_filename, "rb") as f:
                    gamma, generator, target = pickle.load(f)
        else:
            dset = AniDataset.from_hdf(data_path)  # type: ignore
            opts = Options(opts_dict)
            model = RepulsiveModel(opts)
            generator = Generator(model.models, model.loc)

            with st.spinner("Generating gammas..."):
                gammas = generator.get_gammas(dset, num_workers=8)  # type: ignore

                total = np.hstack([moldata["ccsd(t)_cbs.energy"] for moldata in dset.values()])
                electronic = np.hstack(
                    [moldata["dftb.elec_energy"] for moldata in dset.values()]
                )
                target = total - electronic
                gamma = np.vstack([moldata["gammas"] for moldata in gammas.values()])

                with open(cache_filename, "wb") as f:
                    pickle.dump([gamma, generator, target], f)

        os.chdir(os.path.dirname(__file__))

        coefs, residuals, rank, sing_values = np.linalg.lstsq(gamma, target, rcond=1.0e-6)
        pred = np.dot(gamma, coefs)
        mae_fit = np.mean(np.abs(pred - target))
        st.write(f"MAE {mae_fit:.4f} kcal/mol")

        coefs_labels = label_coefs(
            opts_dict["model"]["nknots"],
            opts_dict["model"]["deg"],
            BCONDS[opts_dict["model"]["bconds"]],
        )
        labeled_coefs = pd.Series(coefs.flatten(), index=coefs_labels)

        # View the top 20 correlated features
        columns = ["_".join([str(x) for x in lab]) for lab in coefs_labels]
        gamma_df = pd.DataFrame(gamma, columns=columns)
        top_abs_correlations = get_top_abs_correlations(gamma_df, 20)
        # set column name to correlation 
        top_abs_correlations.columns = ["Correlation"]
        st.dataframe(top_abs_correlations)
        st.caption("Top 20 Correlated Features")

        with st.spinner("Generating feature correlation plot..."):
            import plotly.express as px
            fig = px.imshow(gamma_df.corr())
            fig.update_layout(
                title="Correlation heatmap",
                yaxis_nticks=len(gamma_df.columns),
                height=1000,
                width=1400,
            )
            st.plotly_chart(fig)


if __name__ == "__main__":
    main()
