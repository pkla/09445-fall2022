import hashlib
import importlib
import itertools
import os
import pickle
import sys

import DFTBrepulsive
import DFTBrepulsive.consts
import DFTBrepulsive.data.dataloader
import DFTBrepulsive.generator
import DFTBrepulsive.model
import DFTBrepulsive.options
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dataset import (
    ATOMIC_PAIRS,
    ATOMIC_PAIRS_LETTERS,
    CutoffType,
    calculate_peaks,
    density_sensitive_grid,
    distances_from_coordinates,
    get_cutoffs,
    group_distances,
    histogram,
    load_h5_dataset,
    pdist_mol,
    plot_annotated_histogram,
    torch_cdist,
)
from DFTBrepulsive.consts import BCONDS

# from DFTBrepulsive.data.dataloader import DataSplitter
# from DFTBrepulsive.data.dataset import AniDataset
# from DFTBrepulsive.generator import Generator
# from DFTBrepulsive.model import RepulsiveModel
# from DFTBrepulsive.options import Options

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "DFTBrepulsive", "DFTBrepulsive")
)

# This is a quick hack to import the DFTBrepulsive modules
# from the parent directory. This is not a good practice.
# TODO: Package DFTBrepulsive properly for an editable install.
# os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DFTBrepulsive"))


# os.chdir(os.path.dirname(os.path.abspath(__file__)))


data_path = os.getenv("DATA_PATH", "../data/raw/ANI-1ccx_clean_fullentry.h5")
data_path = os.path.abspath(data_path)


@st.cache(show_spinner=False, persist=False, suppress_st_warning=True, allow_output_mutation=True)
def load_dataset_cached(
    data_path, n_formulas=None, average_by_empirical_formula=False, max_heavy_atoms=None
):
    with st.spinner("Loading dataset from disk..."):
        df = load_h5_dataset(data_path, n_formulas=n_formulas, max_heavy=max_heavy_atoms)
        df = df.set_index("mol", drop=True)

    with st.spinner("Applying filters..."):
        df["num_heavy_atoms"] = df["atomic_numbers"].parallel_apply(
            lambda x: len(x) - np.sum(np.array(x) == 1)
        )

        if max_heavy_atoms is not None:
            df = df.query("num_heavy_atoms <= @max_heavy_atoms")

    with st.spinner("Calculating distances..."):
        df_dist = df.parallel_apply(
            # lambda row: group_distances(torch_cdist(row["coordinates"]).numpy(), row["atomic_numbers"]),
            lambda row: pdist_mol(row["coordinates"], row["atomic_numbers"], ATOMIC_PAIRS),
            axis=1,
            result_type="expand",
        )
    # df_dist = df.parallel_apply(
    #     lambda row: pdist_mol(
    #         row["coordinates"], row["atomic_numbers"]
    #     ),
    #     axis=1,
    #     result_type="expand",
    # )

    with st.spinner("Formatting distances..."):
        if average_by_empirical_formula:
            distances = df_dist.applymap(np.mean)
        else:
            distances = (
                df_dist.apply(
                    lambda x: np.concatenate([z for z in x if not np.isnan(z).any()]), axis=0
                )
                .apply(pd.Series)
                .T
            )

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


cutoff_overrides = {(1, 1): [0, 1.2], (8, 8): [0, 1.5]}


def main():
    st.set_page_config(page_title="DFTB Repulsive", layout="wide")
    st.title("DFTB Repulsive")

    with st.sidebar:
        st.subheader("Dataset")
        subset_data = False  # st.checkbox("Subset Data", value=False)
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

        max_heavy_atoms = st.number_input(
            "Max Heavy Atoms", value=8, min_value=1, max_value=20, step=1
        )

        st.subheader("Histogram")
        nbins = st.number_input("Histogram Bins", value=125, min_value=1, max_value=1000, step=1)

        st.subheader("Peak Finding")
        min_height = st.number_input(
            "Minimum Height", value=0.1, min_value=0.0, max_value=1.0, step=0.01
        )
        min_dist = st.number_input(
            "Minimum Horizontal Distance",
            value=0.35,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.3f",
            help="Required minimal horizontal distance between neighbouring peaks in Angstrom",
        )
        prominence = st.number_input(
            "Prominence",
            value=0.085,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.3f",
            help="The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.",
        )

    if True or st.button("Load Data"):
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
                histograms[pair]["counts"],
                histograms[pair]["bins"],
                min_dist=min_dist,
                height=min_height,
                prominence=prominence,
            )
            for pair in atomic_pairs
        }

        # check if there are any peaks
        no_minima = [
            str(x)
            for x in itertools.compress(atomic_pairs, [not len(x[0]) for x in extrema.values()])
        ]
        no_maxima = [
            str(x)
            for x in list(
                itertools.compress(atomic_pairs, [not len(x[1]) for x in extrema.values()])
            )
        ]

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

        with st.sidebar:
            with st.expander("Cutoff Types"):
                cutoff_types = {}

                to_override = st.multiselect(
                    "Override Cutoffs", ATOMIC_PAIRS, list(cutoff_overrides.keys())
                )

                for pair in to_override:
                    col1, col2 = st.columns(2)
                    cutoff_types[pair] = [
                        col1.number_input(
                            f"{pair} Min", value=cutoff_overrides[pair][0], key=f"{pair}_min"
                        ),
                        col2.number_input(
                            f"{pair} Max", value=cutoff_overrides[pair][1], key=f"{pair}_max"
                        ),
                    ]
                for pair in atomic_pairs:
                    if pair in to_override:
                        continue
                    else:
                        cutoff_types[pair] = st.selectbox(
                            f"{pair}",
                            CutoffType,
                            format_func=lambda x: x.name,
                            key=f"{pair}_cutofftype",
                        )

        with st.expander("Interatomic Distances and Cutoffs", expanded=True):
            tabs = st.tabs([str(pair) for pair in atomic_pairs])
            tab_empties = []
            for tab, (pair, hist) in zip(tabs, histograms.items()):
                with tab:
                    tab_empties.append(st.empty())

            cols = st.columns(2)
            lower_bound_add = cols[0].number_input(
                "Add to Lower Bound",
                value=0.03,
                min_value=-1.0,
                max_value=1.0,
                step=0.01,
                format="%.3f",
            )
            upper_bound_add = cols[1].number_input(
                "Add to Upper Bound",
                value=0.00,
                min_value=-1.0,
                max_value=1.0,
                step=0.01,
                format="%.3f",
            )
            cutoff_strs = []
            cutoffs = {}
            for pair, pair_extrema in extrema.items():
                cutoff_type = cutoff_types[pair]

                if isinstance(cutoff_type, (list, tuple)):
                    lower, upper = cutoff_type
                    cutoffs[pair] = (lower, upper)
                else:
                    lower, upper = get_cutoffs(
                        pair_extrema, histograms[pair]["bins"], cutoff_type=cutoff_type
                    )
                    lower += lower_bound_add
                    upper += upper_bound_add
                    cutoffs[pair] = (lower, upper)

                cutoff_strs.append(f"{pair}: ({lower:.5f}, {upper:.5f})")

            st.code("cutoffs = {\n    " + ",\n    ".join(cutoff_strs) + "\n}")

            with st.spinner("Plotting Histograms..."):
                for tab_empty, (pair, hist) in zip(tabs, histograms.items()):
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax = plot_annotated_histogram(**hist, ax=ax, extrema=extrema[pair])
                    ax.set_title(f"{pair}", y=1.13, fontsize=20)
                    # plt.subplots_adjust(wspace=0.1, hspace=0.5)
                    tab_empty.pyplot(fig)
                    tab_empty.caption(f"Distribution of {pair}-Pairwise Interatomic Distances")

        with st.expander("Summary", expanded=False):
            # Bar chart of heavy atom counts
            fig = (
                df["num_heavy_atoms"]
                .value_counts()
                .sort_index()
                .plot.bar(backend="plotly", title="Heavy Atom Counts")
                .update_layout(
                    showlegend=False,
                    xaxis_title="Heavy Atom Count",
                    yaxis_title="Number of Conformations",
                )
            )
            fig.update_traces(
                marker_color="rgb(158,202,225)",
                marker_line_color="rgb(8,48,107)",
                marker_line_width=1.5,
                opacity=0.8,
            )
            st.plotly_chart(fig)

            # Bar chart of atoms counts
            df["num_atoms"] = df["atomic_numbers"].apply(len)
            fig = (
                df["num_atoms"]
                .value_counts()
                .sort_index()
                .plot.bar(backend="plotly", title="Atom Counts")
                .update_layout(
                    showlegend=False,
                    xaxis_title="Atom Count",
                    yaxis_title="Number of Conformations",
                )
            )
            fig.update_traces(
                marker_color="rgb(158,202,225)",
                marker_line_color="rgb(8,48,107)",
                marker_line_width=1.5,
                opacity=0.8,
            )
            st.plotly_chart(fig)

            # Bar chart of empirical formulas
            fig = (
                df.index.value_counts()
                .plot(
                    kind="bar",
                    title="Number of Conformations per Empirical Formula",
                    backend="plotly",
                )
                .update_layout(
                    showlegend=False,
                    xaxis_title="Empirical Formula",
                    yaxis_title="Number of Conformations",
                )
            )
            fig.update_traces(
                marker_color="rgb(158,202,225)",
                marker_line_color="rgb(8,48,107)",
                marker_line_width=0.1,
                opacity=1,
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        st.subheader("Model Options")
        nknots = st.number_input("Number of Knots", value=20, min_value=1, max_value=100, step=1)
        deg = st.number_input("Degree", value=3, min_value=1, max_value=10, step=1)
        maxder = st.number_input("Max Derivative", value=2, min_value=0, max_value=10, step=1)
        bconds = st.selectbox("Boundary Conditions", BCONDS.keys(), index=2)

        if st.checkbox("Derivative Constraints"):
            constr = tuple(st.text_input("Constraints", "-1,+2").split(","))
            ngrid = st.number_input(
                "Number of Grid Points", value=500, min_value=1, max_value=10000, step=1
            )
            constr = {"constr": constr, "ngrid": ngrid}
        else:
            constr = None
        weighted_sampling = st.checkbox("Weighted Knot Sampling", value=True)

        if weighted_sampling:
            grids = {}
            st.subheader(f"Knot Sampling Density")
            cols = st.columns(2)
            lower_density = cols[0].number_input(
                "Minimum Density",
                value=0.001,
                min_value=0.0,
                max_value=10.0,
                step=0.001,
                format="%.5f",
                key=f"denselower",
            )
            upper_density = cols[1].number_input(
                "Maximum Density",
                value=0.01,
                min_value=0.0,
                max_value=10.0,
                step=0.001,
                format="%.5f",
                key=f"denseupper",
            )
            for pair in atomic_pairs:
                pair_dists = distances[pair]
                pair_dists = pair_dists[
                    (pair_dists < cutoffs[pair][1]) & (pair_dists > cutoffs[pair][0])
                ]
                grids[pair] = density_sensitive_grid(
                    pair_dists, nknots, lower_density, upper_density
                )
        else:
            grids = {}

        st.subheader("Data Splitting")
        reverse_cross_validation = st.checkbox("Reverse Cross Validation", value=False)
        cross_validation = st.checkbox("Cross Validation", value=False)
        if reverse_cross_validation or cross_validation:
            num_folds = st.number_input("Number of Folds", value=5, min_value=2, max_value=10)
            cv_seed = st.number_input("Random Seed", value=42)
        else:
            num_folds = 1
            cv_seed = 1

        st.subheader("Visualization")
        view_correlation_heatmap = st.checkbox("Feature Correlation Heatmap", value=False)
        view_residuals_vs_predicted = st.checkbox("Residuals vs. Predicted", value=False)
        view_residuals_vs_heavy_atoms = st.checkbox("Residuals vs. Heavy Atoms", value=False)
        view_scaled_residuals_vs_predicted = st.checkbox(
            "Scaled Residuals vs. Predicted", value=True
        )
        view_residuals_vs_predictors = st.checkbox("Residuals vs. Predictors", value=True)

    if weighted_sampling:
        with st.expander("Weighted Knot Sampling", expanded=False):
            fig, ax = plt.subplots(
                nrows=len(atomic_pairs),
                ncols=1,
                figsize=(8, 1 * len(atomic_pairs)),
                constrained_layout=True,
                sharex=True,
            )
            for i, (pair, grid) in enumerate(grids.items()):
                # plot density of points
                # vertical lines at knots
                ax[i].vlines(grid, 0, 1, color="red", alpha=0.5)
                ax[i].set_ylim(-0.1, 1.1)
                # remove y axis
                ax[i].set_yticks([])
                ax[i].set_ylabel(f"{pair}", rotation=0, labelpad=20, fontsize=10)
                ax[i].set_xlim(0.5, 2)

                # hide x axis ticks
                if i < len(atomic_pairs) - 1:
                    # ax[i].set_xticks([])
                    ax[i].set_xticklabels([])

            min_lower_cutoff = min([cutoffs[pair][0] for pair in atomic_pairs])
            max_upper_cutoff = max([cutoffs[pair][1] for pair in atomic_pairs])
            xticks = np.linspace(min_lower_cutoff, max_upper_cutoff, 11)
            ax[-1].set_xlabel("Interatomic Distance")
            ax[-1].set_xticks(xticks)
            ax[-1].set_xticklabels([f"{x:.2f}" for x in xticks])

            st.pyplot(fig)

    with st.expander("Model", expanded=True):

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
        }

        if constr:
            opts_dict["constr"] = constr

        if grids:
            override = {"xknots": grids}
        else:
            override = None

        # os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DFTBrepulsive"))
        cache_filename = (
            hashlib.sha256(f"{opts_dict}_{override}".encode("utf-8")).hexdigest() + ".pkl"
        )

        cache_may_be_invalid = True
        if st.button("Train Model"):
            importlib.reload(DFTBrepulsive)
            splitter = DFTBrepulsive.data.dataloader.DataSplitter(
                "heavy", [list(range(1, max_heavy_atoms + 1))]
            )
            dset = DFTBrepulsive.data.dataset.AniDataset.from_hdf(data_path)  # type: ignore
            dset = splitter.split(dset)[0]
            opts = DFTBrepulsive.options.Options(opts_dict, override=override)
            try:
                print(opts["model"]["xknots"])
            except Exception as e:
                print(e)

            model = DFTBrepulsive.model.RepulsiveModel(opts)
            generator = DFTBrepulsive.generator.Generator(model.models, model.loc)

            with st.spinner("Generating gammas..."):
                gammas = generator.get_gammas(dset, num_workers=16)  # type: ignore

                total = np.hstack([moldata["ccsd(t)_cbs.energy"] for moldata in dset.values()])
                electronic = np.hstack([moldata["dftb.elec_energy"] for moldata in dset.values()])
                target = total - electronic
                gamma = np.vstack([moldata["gammas"] for moldata in gammas.values()])

                # with open(cache_filename, "wb") as f:
                #     pickle.dump([gamma, generator, target], f)

            st.session_state["gamma"] = gamma
            st.session_state["target"] = target

            cache_may_be_invalid = False

        if "gamma" in st.session_state:
            if cache_may_be_invalid:
                st.warning("Using cached model. Press 'Train Model' to retrain.")
            gamma = st.session_state["gamma"]
            target = st.session_state["target"]
            # os.chdir(os.path.dirname(os.path.abspath(__file__)))
            np.random.seed(cv_seed)
            if reverse_cross_validation:
                folds = np.arange(num_folds)
                cv_shuffle_idx = np.random.permutation(len(target))
                fold_indices = np.array_split(cv_shuffle_idx, num_folds)
                coefs_folds = []
                preds_folds = []
                residuals_folds = []
                for train_fold in folds:
                    test_folds = np.where(folds != train_fold).reshape(-1)
                    test_idx = np.hstack([fold_indices[i] for i in test_folds])
                    train_idx = fold_indices[train_fold]
                    coefs, residuals, rank, sing_values = np.linalg.lstsq(
                        gamma[train_idx], target[train_idx], rcond=1.0e-6
                    )
                    coefs_folds.append(coefs)
                    pred = np.dot(gamma[test_idx], coefs)
                    preds_folds.append(pred)
                    residuals_folds.append(target[test_idx] - pred)
                coefs = np.concatenate(coefs_folds, axis=1)
                pred = np.concatenate(preds_folds)
                resid = np.concatenate(residuals_folds)
            elif cross_validation:
                folds = np.arange(num_folds)
                cv_shuffle_idx = np.random.permutation(len(target))
                fold_indices = np.array_split(cv_shuffle_idx, num_folds)
                coefs_folds = []
                preds_folds = []
                residuals_folds = []
                for test_fold in folds:
                    train_folds = np.where(folds != test_fold)[0]
                    train_idx = np.hstack([fold_indices[int(i)] for i in train_folds])
                    test_idx = fold_indices[test_fold]
                    coefs, residuals, rank, sing_values = np.linalg.lstsq(
                        gamma[train_idx], target[train_idx], rcond=1.0e-6
                    )
                    coefs_folds.append(coefs)
                    pred = np.dot(gamma[test_idx], coefs)
                    preds_folds.append(pred)
                    residuals_folds.append(target[test_idx] - pred)

                coefs = np.concatenate(coefs_folds, axis=1)
                pred = np.concatenate(preds_folds)
                resid = np.concatenate(residuals_folds)
            else:
                coefs, residuals, rank, sing_values = np.linalg.lstsq(gamma, target, rcond=1.0e-6)
                pred = np.dot(gamma, coefs)
                resid = target - pred

            mae_fit = np.mean(np.abs(resid))
            mae_per_atom = np.mean(resid / df["num_atoms"])
            mae_per_heavy_atom = np.mean(resid / df["num_heavy_atoms"])
            st.write(f"MAE {mae_fit:.4f} kcal/mol")
            st.write(f"MAE per Atom {mae_per_atom:.4f} kcal/mol")
            st.write(f"MAE per Heavy Atom {mae_per_heavy_atom:.4f} kcal/mol")
            coefs_labels = label_coefs(
                opts_dict["model"]["nknots"],
                opts_dict["model"]["deg"],
                BCONDS[opts_dict["model"]["bconds"]],
            )
            labeled_coefs = pd.Series(coefs.flatten(), index=coefs_labels)
            tabs = st.tabs(
                [
                    "Residuals vs. Predicted",
                    "Scaled Residuals vs. Predicted",
                    "Residuals vs. Heavy Atoms",
                    "Residuals vs. Predictors",
                ]
            )

            # plot residuals
            resid_df = pd.DataFrame({"pred": pred, "target": target, "resid": pred - target})
            resid_df["num_heavy_atoms"] = (
                df["num_heavy_atoms"].reset_index(drop=True).values.astype(int)
            )
            resid_df["resid_scaled"] = (pred - target) / resid_df["num_heavy_atoms"]

            resid_df.index = df.index
            with tabs[0]:
                if view_residuals_vs_predicted:
                    # fig, ax = plt.subplots(figsize=(10, 5))
                    # resid_df.plot.scatter(x="pred", y="resid", title="Residuals vs. Predicted", xlabel="Predicted (kcal/mol)", ylabel="Residual (kcal/mol)", ax=ax)
                    # # make points smaller with opacity matplotlib
                    # plt.setp(ax.collections, sizes=[10], alpha=0.3)
                    # st.pyplot(fig)
                    fig = resid_df.plot.scatter(
                        x="pred", y="resid", title="Residuals vs. Predicted", backend="plotly"
                    ).update_layout(
                        showlegend=False,
                        xaxis_title="Predicted (kcal/mol)",
                        yaxis_title="Residual (kcal/mol)",
                    )
                    fig.update_traces(
                        marker_color="rgb(158,202,225)",
                        marker_line_color="rgb(8,48,107)",
                        marker_line_width=0.1,
                        opacity=0.4,
                    )

                    # add molecule names to hover
                    fig.update_traces()
                    fig.update_layout(hovermode="closest")
                    fig.update_layout(
                        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
                    )
                    fig.update_traces(
                        hovertemplate="<b>%{text}</b><br><br>residual: %{y:.4f}<br>predicted: %{x:.4f}<extra></extra>"
                    )
                    fig.data[0].text = resid_df.index
                    fig.update_layout(height=800)
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                if view_scaled_residuals_vs_predicted:
                    # fig, ax = plt.subplots(figsize=(10, 5))
                    # resid_df.plot.scatter(x="pred", y="resid_scaled", title="Scaled Residuals vs. Predicted", xlabel="Predicted (kcal/mol)", ylabel="Residual (kcal/mol)", ax=ax)
                    # # make points smaller with opacity matplotlib
                    # plt.setp(ax.collections, sizes=[10], alpha=0.3)
                    # st.pyplot(fig)

                    fig = resid_df.plot.scatter(
                        x="pred",
                        y="resid_scaled",
                        title="Scaled Residuals vs. Predicted",
                        backend="plotly",
                    ).update_layout(
                        showlegend=False,
                        xaxis_title="Predicted (kcal/mol)",
                        yaxis_title="Residual (kcal/mol)",
                    )
                    fig.update_traces(
                        marker_color="rgb(158,202,225)",
                        marker_line_color="rgb(8,48,107)",
                        marker_line_width=0.1,
                        opacity=0.4,
                    )

                    # add molecule names to hover
                    fig.update_traces()
                    fig.update_layout(hovermode="closest")
                    fig.update_layout(
                        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
                    )
                    fig.update_traces(
                        hovertemplate="<b>%{text}</b><br><br>residual: %{y:.4f}<br>predicted: %{x:.4f}<extra></extra>"
                    )
                    fig.data[0].text = resid_df.index
                    fig.update_layout(height=800)
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption("Scaled by number of heavy atoms")

            with tabs[2]:
                if view_residuals_vs_heavy_atoms:
                    # residuals vs heavy atoms, plotly boxplot
                    fig = px.box(
                        resid_df,
                        x="num_heavy_atoms",
                        y="resid",
                        title="Residuals vs. Heavy Atoms",
                        labels={"num_heavy_atoms": "Heavy Atoms", "resid": "Residual (kcal/mol)"},
                        color="num_heavy_atoms",
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[3]:
                if view_residuals_vs_predictors:
                    pair_selection = st.selectbox("Select Pair", ATOMIC_PAIRS_LETTERS)

                    coef_idxs = [
                        i
                        for i in range(len(coefs_labels))
                        if "-".join(pair_selection) in coefs_labels[i]
                    ]

                    for coef_idx in coef_idxs:
                        # coefficient_selection = st.selectbox(
                        #     "Select a coefficient",
                        #     range(len(coefs_labels)),
                        #     format_func=lambda i: coefs_labels[i],
                        # )
                        coefficient_selection = coef_idx
                        coef_selection_label = coefs_labels[coefficient_selection]

                        non_zero_idx = (gamma[:, coefficient_selection] > 0).flatten()
                        predictor_values = gamma[non_zero_idx, coefficient_selection]

                        if predictor_values.shape[0] == 0:
                            st.warning(f"No non-zero values for {coef_selection_label}")
                        else:
                            # residuals vs covariate, plotly scatter
                            fig = px.scatter(
                                x=predictor_values,
                                y=resid_df["resid"].iloc[non_zero_idx],
                                # trendline="ols",
                                title=f"Residuals vs. Predictor {coef_selection_label}",
                                labels={
                                    "x": f"Predictor Variable {coef_selection_label}",
                                    "y": "Residual (kcal/mol)",
                                },
                            )
                            fig.update_traces(
                                marker_color="rgb(158,202,225)",
                                marker_line_color="rgb(8,48,107)",
                                marker_line_width=0.1,
                                opacity=0.4,
                            )

                            # Add molecule names to hover
                            fig.update_layout(hovermode="closest")
                            fig.update_layout(
                                hoverlabel=dict(
                                    bgcolor="white", font_size=16, font_family="Rockwell"
                                )
                            )
                            fig.update_traces(
                                hovertemplate="<b>%{text}</b><br><br>residual: %{y:.4f}<br>predictor: %{x:.4f}<extra></extra>"
                            )
                            fig.data[0].text = resid_df.index[non_zero_idx]
                            st.plotly_chart(fig, use_container_width=True)

            # # View the top 20 correlated features
            columns = ["_".join([str(x) for x in lab]) for lab in coefs_labels]
            gamma_df = pd.DataFrame(gamma, columns=columns)
            # top_abs_correlations = get_top_abs_correlations(gamma_df, 20)
            # # set column name to correlation
            # top_abs_correlations.columns = ["Correlation"]
            # st.dataframe(top_abs_correlations)
            # st.caption("Top 20 Correlated Features")

            if view_correlation_heatmap:
                with st.spinner("Generating feature correlation plot..."):
                    fig = px.imshow(gamma_df.corr())
                    fig.update_layout(
                        title="Correlation heatmap",
                        yaxis_nticks=len(gamma_df.columns),
                        height=1000,
                        width=1000,
                    )
                    # remove colorbar
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
