# ============================================================
# FiTabR Retrieval Analysis
#
# Compares the neighborhoods retrieved by TabR and FiTabR.
#
# Test samples = queries
# Training samples = retrieval candidate bank
#
# Analysis is paired by random seed.
# ============================================================

from pathlib import Path
import os
import sys


# ------------------------------------------------------------
# Project setup
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

os.environ.setdefault(
    "PROJECT_DIR",
    str(PROJECT_DIR),
)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import faiss
import numpy as np
import pandas as pd
import torch

import lib

from bin.tabr import Model as TabRModel
from bin.fitabr import Model as FiTabRModel


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "heart"

TABR_EVAL_DIR = (
    PROJECT_DIR
    / "exp"
    / "tabr"
    / DATASET_NAME
    / "0-evaluation"
)

FITABR_EVAL_DIR = (
    PROJECT_DIR
    / "exp"
    / "fitabr"
    / DATASET_NAME
    / "0-evaluation"
)


SEEDS = range(15)

# 96 is the actual context size used by your experiments.
K_VALUES = [
    5,
    10,
    20,
    96,
]

DEVICE = torch.device("cpu")


OUTPUT_DIR = (
    PROJECT_DIR
    / "notebooks"
    / "retrieval_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_dataset(
    run_dir: Path,
    model_class,
):

    report = lib.load_report(
        run_dir
    )

    config = report["config"]


    # --------------------------------------------------------
    # Recreate exactly the dataset preprocessing used by run
    # --------------------------------------------------------

    dataset = lib.build_dataset(
        **config["data"]
    ).to_torch(
        DEVICE
    )


    # --------------------------------------------------------
    # Recreate model
    # --------------------------------------------------------

    model = model_class(
        n_num_features=
            dataset.n_num_features,

        n_bin_features=
            dataset.n_bin_features,

        cat_cardinalities=
            dataset.cat_cardinalities(),

        n_classes=
            dataset.n_classes(),

        **config["model"],
    )


    model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = lib.load_checkpoint(
        run_dir,
        map_location=DEVICE,
    )

    state_dict = checkpoint["model"]


    # Handle checkpoints created with DataParallel
    if any(
        key.startswith("module.")
        for key in state_dict
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value
            in state_dict.items()
        }


    model.load_state_dict(
        state_dict
    )

    model.eval()


    return (
        model,
        dataset,
        config,
    )


# ============================================================
# DATA HELPERS
# ============================================================

def get_x(
    dataset,
    part: str,
):

    return {
        key[2:]:
            dataset.data[key][part]

        for key in dataset.data

        if key.startswith("X_")
    }


# ============================================================
# ENCODE INTO RETRIEVAL KEY SPACE
# ============================================================

def encode_keys(
    model,
    dataset,
    part: str,
):

    x = get_x(
        dataset,
        part,
    )

    with torch.inference_mode():

        _, keys = model._encode(
            x
        )


    keys = (
        keys
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


    return np.ascontiguousarray(
        keys
    )


# ============================================================
# NEAREST-NEIGHBOR RETRIEVAL
# ============================================================

def retrieve_neighbors(
    train_keys,
    query_keys,
    k,
):

    dimension = train_keys.shape[1]


    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        train_keys
    )


    distances, indices = index.search(
        query_keys,
        k,
    )


    return (
        distances,
        indices,
    )


# ============================================================
# PURITY
# ============================================================

def purity_per_query(
    neighbor_indices,
    y_train,
    y_query,
):

    neighbor_labels = y_train[
        neighbor_indices
    ]


    same_label = (
        neighbor_labels
        ==
        y_query[:, None]
    )


    return same_label.mean(
        axis=1
    )


# ============================================================
# SOFTMAX RETRIEVAL-WEIGHT PURITY
#
# TabR similarity is:
#
#     - || k_q - k_i ||^2
#
# FAISS IndexFlatL2 returns || k_q - k_i ||^2,
# therefore similarity = -distance.
# ============================================================

def weighted_purity_per_query(
    distances,
    neighbor_indices,
    y_train,
    y_query,
):

    similarities = -distances


    # Numerically stable softmax
    similarities = (
        similarities
        - similarities.max(
            axis=1,
            keepdims=True,
        )
    )


    weights = np.exp(
        similarities
    )

    weights = (
        weights
        /
        weights.sum(
            axis=1,
            keepdims=True,
        )
    )


    neighbor_labels = y_train[
        neighbor_indices
    ]


    same_label = (
        neighbor_labels
        ==
        y_query[:, None]
    )


    return (
        weights
        * same_label
    ).sum(
        axis=1
    )


# ============================================================
# TOP-k OVERLAP
# ============================================================

def overlap_per_query(
    tabr_indices,
    fitabr_indices,
):

    k = tabr_indices.shape[1]


    overlaps = []


    for tabr_neighbors, fitabr_neighbors in zip(
        tabr_indices,
        fitabr_indices,
    ):

        intersection = len(
            set(tabr_neighbors.tolist())
            &
            set(fitabr_neighbors.tolist())
        )


        overlaps.append(
            intersection / k
        )


    return np.asarray(
        overlaps,
        dtype=float,
    )


# ============================================================
# SAFE CLASS MEAN
# ============================================================

def class_mean(
    values,
    mask,
):

    if mask.sum() == 0:
        return np.nan

    return values[
        mask
    ].mean()


# ============================================================
# MAIN ANALYSIS
# ============================================================

rows = []


for seed in SEEDS:

    print(
        f"\nAnalyzing seed {seed}..."
    )


    tabr_run = (
        TABR_EVAL_DIR
        / str(seed)
    )

    fitabr_run = (
        FITABR_EVAL_DIR
        / str(seed)
    )


    if not tabr_run.exists():
        print(
            f"Missing TabR run: {tabr_run}"
        )
        continue


    if not fitabr_run.exists():
        print(
            f"Missing FiTabR run: {fitabr_run}"
        )
        continue


    # --------------------------------------------------------
    # Load TabR
    # --------------------------------------------------------

    (
        tabr_model,
        tabr_dataset,
        tabr_config,
    ) = load_model_and_dataset(
        tabr_run,
        TabRModel,
    )


    # --------------------------------------------------------
    # Load FiTabR
    # --------------------------------------------------------

    (
        fitabr_model,
        fitabr_dataset,
        fitabr_config,
    ) = load_model_and_dataset(
        fitabr_run,
        FiTabRModel,
    )


    # --------------------------------------------------------
    # Verify that both models use same dataset split
    # --------------------------------------------------------

    y_train_tabr = (
        tabr_dataset
        .Y["train"]
        .cpu()
        .numpy()
        .astype(int)
    )

    y_test_tabr = (
        tabr_dataset
        .Y["test"]
        .cpu()
        .numpy()
        .astype(int)
    )


    y_train_fitabr = (
        fitabr_dataset
        .Y["train"]
        .cpu()
        .numpy()
        .astype(int)
    )

    y_test_fitabr = (
        fitabr_dataset
        .Y["test"]
        .cpu()
        .numpy()
        .astype(int)
    )


    assert np.array_equal(
        y_train_tabr,
        y_train_fitabr,
    ), "TabR and FiTabR training labels differ."


    assert np.array_equal(
        y_test_tabr,
        y_test_fitabr,
    ), "TabR and FiTabR test labels differ."


    y_train = y_train_tabr
    y_test = y_test_tabr


    # --------------------------------------------------------
    # Encode retrieval representations
    # --------------------------------------------------------

    tabr_train_keys = encode_keys(
        tabr_model,
        tabr_dataset,
        "train",
    )

    tabr_test_keys = encode_keys(
        tabr_model,
        tabr_dataset,
        "test",
    )


    fitabr_train_keys = encode_keys(
        fitabr_model,
        fitabr_dataset,
        "train",
    )

    fitabr_test_keys = encode_keys(
        fitabr_model,
        fitabr_dataset,
        "test",
    )


    max_k = max(
        K_VALUES
    )


    if max_k > len(y_train):
        raise ValueError(
            f"k={max_k} is greater than "
            f"training size={len(y_train)}"
        )


    # --------------------------------------------------------
    # Retrieve maximum number once
    # --------------------------------------------------------

    (
        tabr_distances,
        tabr_neighbors,
    ) = retrieve_neighbors(
        tabr_train_keys,
        tabr_test_keys,
        max_k,
    )


    (
        fitabr_distances,
        fitabr_neighbors,
    ) = retrieve_neighbors(
        fitabr_train_keys,
        fitabr_test_keys,
        max_k,
    )


    positive_mask = (
        y_test == 1
    )

    negative_mask = (
        y_test == 0
    )


    # --------------------------------------------------------
    # k-specific analysis
    # --------------------------------------------------------

    for k in K_VALUES:

        tab_idx = (
            tabr_neighbors[:, :k]
        )

        fit_idx = (
            fitabr_neighbors[:, :k]
        )


        tab_dist = (
            tabr_distances[:, :k]
        )

        fit_dist = (
            fitabr_distances[:, :k]
        )


        # ----------------------------------------------------
        # Neighborhood overlap
        # ----------------------------------------------------

        overlap = overlap_per_query(
            tab_idx,
            fit_idx,
        )


        # ----------------------------------------------------
        # Unweighted label purity
        # ----------------------------------------------------

        tab_purity = purity_per_query(
            tab_idx,
            y_train,
            y_test,
        )

        fit_purity = purity_per_query(
            fit_idx,
            y_train,
            y_test,
        )


        # ----------------------------------------------------
        # Retrieval-weighted label purity
        # ----------------------------------------------------

        tab_weighted_purity = (
            weighted_purity_per_query(
                tab_dist,
                tab_idx,
                y_train,
                y_test,
            )
        )

        fit_weighted_purity = (
            weighted_purity_per_query(
                fit_dist,
                fit_idx,
                y_train,
                y_test,
            )
        )


        # ----------------------------------------------------
        # Save one row per seed and k
        # ----------------------------------------------------

        rows.append(
            {
                "Dataset":
                    DATASET_NAME,

                "Seed":
                    seed,

                "k":
                    k,

                # Neighborhood change
                "Top-k Overlap":
                    overlap.mean(),

                "Neighbor Change Rate":
                    1.0 - overlap.mean(),

                # Overall purity
                "TabR Purity":
                    tab_purity.mean(),

                "FiTabR Purity":
                    fit_purity.mean(),

                "Delta Purity":
                    (
                        fit_purity.mean()
                        -
                        tab_purity.mean()
                    ),

                # Positive queries
                "TabR Positive-query Purity":
                    class_mean(
                        tab_purity,
                        positive_mask,
                    ),

                "FiTabR Positive-query Purity":
                    class_mean(
                        fit_purity,
                        positive_mask,
                    ),

                "Delta Positive-query Purity":
                    (
                        class_mean(
                            fit_purity,
                            positive_mask,
                        )
                        -
                        class_mean(
                            tab_purity,
                            positive_mask,
                        )
                    ),

                # Negative queries
                "TabR Negative-query Purity":
                    class_mean(
                        tab_purity,
                        negative_mask,
                    ),

                "FiTabR Negative-query Purity":
                    class_mean(
                        fit_purity,
                        negative_mask,
                    ),

                "Delta Negative-query Purity":
                    (
                        class_mean(
                            fit_purity,
                            negative_mask,
                        )
                        -
                        class_mean(
                            tab_purity,
                            negative_mask,
                        )
                    ),

                # Similarity-weighted purity
                "TabR Weighted Purity":
                    tab_weighted_purity.mean(),

                "FiTabR Weighted Purity":
                    fit_weighted_purity.mean(),

                "Delta Weighted Purity":
                    (
                        fit_weighted_purity.mean()
                        -
                        tab_weighted_purity.mean()
                    ),
            }
        )


# ============================================================
# SEED-LEVEL RESULTS
# ============================================================

seed_df = pd.DataFrame(
    rows
)


if seed_df.empty:
    raise RuntimeError(
        "No retrieval results were generated."
    )


seed_output = (
    OUTPUT_DIR
    / f"{DATASET_NAME}_retrieval_seed_results.csv"
)

seed_df.to_csv(
    seed_output,
    index=False,
)


print(
    f"\nSaved seed results: {seed_output}"
)


# ============================================================
# SUMMARY ACROSS 15 SEEDS
# ============================================================

metric_columns = [
    "Top-k Overlap",
    "Neighbor Change Rate",

    "TabR Purity",
    "FiTabR Purity",
    "Delta Purity",

    "TabR Positive-query Purity",
    "FiTabR Positive-query Purity",
    "Delta Positive-query Purity",

    "TabR Negative-query Purity",
    "FiTabR Negative-query Purity",
    "Delta Negative-query Purity",

    "TabR Weighted Purity",
    "FiTabR Weighted Purity",
    "Delta Weighted Purity",
]


publication_rows = []


for k in K_VALUES:

    subset = seed_df[
        seed_df["k"] == k
    ]


    row = {
        "k": k
    }


    for metric in metric_columns:

        mean = subset[
            metric
        ].mean()

        std = subset[
            metric
        ].std(
            ddof=1
        )


        row[metric] = (
            f"{mean:.4f} ± {std:.4f}"
        )


    publication_rows.append(
        row
    )


summary_df = pd.DataFrame(
    publication_rows
)


summary_output = (
    OUTPUT_DIR
    / f"{DATASET_NAME}_retrieval_summary.csv"
)


summary_df.to_csv(
    summary_output,
    index=False,
)


print(
    f"\nSaved summary: {summary_output}"
)


print(
    "\nRetrieval summary:\n"
)

print(
    summary_df.to_string(
        index=False
    )
)