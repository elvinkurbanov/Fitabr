from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Script location:
# Fitabr/notebooks/learning_analysis/compare_tabr_fitabr_seeds.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABR_ROOT = (
    PROJECT_ROOT
    / "exp"
    / "tabr"
)

FITABR_ROOT = (
    PROJECT_ROOT
    / "exp"
    / "fitabr"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "notebooks"
    / "learning_analysis"
    / "tabr_vs_fitabr"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASETS / SEEDS
# ============================================================

DATASETS = {
    "heart": "Heart Disease",
    "pima": "Pima Diabetes",
    "stroke": "Stroke Prediction",
}

SEEDS = list(range(15))


# ============================================================
# LOAD ONE REPORT
# ============================================================

def load_report(
    experiment_root,
    dataset,
    seed,
):

    report_path = (
        experiment_root
        / dataset
        / "0-evaluation"
        / str(seed)
        / "report.json"
    )

    if not report_path.exists():
        print(
            f"[WARNING] Missing: {report_path}"
        )
        return None

    with open(report_path, "r") as f:

        data = json.load(
            f,
            parse_constant=lambda x: float("inf")
        )

    try:

        result = {
            "seed":
                seed,

            "best_epoch":
                data["best_epoch"],

            # -------------------------
            # Cross-entropy
            # -------------------------

            "train_ce":
                data["metrics"]
                ["train"]
                ["cross-entropy"],

            "val_ce":
                data["metrics"]
                ["val"]
                ["cross-entropy"],

            "test_ce":
                data["metrics"]
                ["test"]
                ["cross-entropy"],

            # -------------------------
            # ROC-AUC
            # -------------------------

            "train_auc":
                data["metrics"]
                ["train"]
                ["roc-auc"],

            "val_auc":
                data["metrics"]
                ["val"]
                ["roc-auc"],

            "test_auc":
                data["metrics"]
                ["test"]
                ["roc-auc"],

            # -------------------------
            # Accuracy / score
            # -------------------------

            "train_score":
                data["metrics"]
                ["train"]
                ["score"],

            "val_score":
                data["metrics"]
                ["val"]
                ["score"],

            "test_score":
                data["metrics"]
                ["test"]
                ["score"],
        }

    except KeyError as error:

        print(
            f"[WARNING] Missing key "
            f"{error} in {report_path}"
        )

        return None

    return result


# ============================================================
# LOAD PAIRED TABR / FITABR RESULTS
# ============================================================

def load_paired_results(dataset):

    paired_results = []

    for seed in SEEDS:

        tabr = load_report(
            TABR_ROOT,
            dataset,
            seed,
        )

        fitabr = load_report(
            FITABR_ROOT,
            dataset,
            seed,
        )

        # Use only seeds for which BOTH models exist.
        if (
            tabr is None
            or fitabr is None
        ):
            continue

        paired_results.append(
            {
                "seed": seed,
                "tabr": tabr,
                "fitabr": fitabr,
            }
        )

    return paired_results


# ============================================================
# HELPERS
# ============================================================

def mean_sd(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    mean = values.mean()

    if len(values) > 1:
        sd = values.std(
            ddof=1
        )
    else:
        sd = 0.0

    return mean, sd


def get_array(
    paired_results,
    model,
    metric,
):

    return np.asarray(
        [
            x[model][metric]
            for x in paired_results
        ],
        dtype=float,
    )


def get_seeds(
    paired_results
):

    return np.asarray(
        [
            x["seed"]
            for x in paired_results
        ]
    )


# ============================================================
# 1. VALIDATION CROSS-ENTROPY BY SEED
# ============================================================

def plot_validation_ce(
    dataset,
    dataset_name,
    paired_results,
):

    seeds = get_seeds(
        paired_results
    )

    tabr = get_array(
        paired_results,
        "tabr",
        "val_ce",
    )

    fitabr = get_array(
        paired_results,
        "fitabr",
        "val_ce",
    )

    tabr_mean, tabr_sd = mean_sd(tabr)
    fitabr_mean, fitabr_sd = mean_sd(fitabr)

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    # Slight horizontal offset only
    # to avoid overlapping points.
    ax.scatter(
        seeds - 0.08,
        tabr,
        s=75,
        label="TabR",
    )

    ax.scatter(
        seeds + 0.08,
        fitabr,
        s=75,
        label="FiTabR",
    )

    ax.axhline(
        tabr_mean,
        linestyle="--",
        linewidth=1.4,
        label=(
            f"TabR mean = "
            f"{tabr_mean:.3f}"
        ),
    )

    ax.axhline(
        fitabr_mean,
        linestyle="--",
        linewidth=1.4,
        label=(
            f"FiTabR mean = "
            f"{fitabr_mean:.3f}"
        ),
    )

    ax.set_xlabel(
        "Random Seed"
    )

    ax.set_ylabel(
        "Validation Cross-Entropy"
    )

    ax.set_title(
        (
            "Validation Cross-Entropy Across "
            f"Random Seeds — {dataset_name}"
        )
    )

    ax.set_xticks(
        seeds
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    summary = (
        f"TabR: {tabr_mean:.3f} ± "
        f"{tabr_sd:.3f}\n"
        f"FiTabR: {fitabr_mean:.3f} ± "
        f"{fitabr_sd:.3f}"
    )

    ax.text(
        0.99,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "alpha": 0.15
        },
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{dataset}_validation_ce_tabr_vs_fitabr.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 2. PER-SEED CHANGE IN VALIDATION CROSS-ENTROPY
# ============================================================

def plot_validation_ce_difference(
    dataset,
    dataset_name,
    paired_results,
):

    seeds = get_seeds(
        paired_results
    )

    tabr = get_array(
        paired_results,
        "tabr",
        "val_ce",
    )

    fitabr = get_array(
        paired_results,
        "fitabr",
        "val_ce",
    )

    # Negative means FiTabR has LOWER CE,
    # which is favorable.
    delta = (
        fitabr - tabr
    )

    delta_mean, delta_sd = mean_sd(
        delta
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.scatter(
        seeds,
        delta,
        s=80,
    )

    # Zero = no difference.
    ax.axhline(
        0,
        linewidth=1.5,
    )

    ax.axhline(
        delta_mean,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Mean Δ = "
            f"{delta_mean:.4f}"
        ),
    )

    ax.set_xlabel(
        "Random Seed"
    )

    ax.set_ylabel(
        "Δ Validation Cross-Entropy\n"
        "(FiTabR − TabR)"
    )

    ax.set_title(
        (
            "FiTabR − TabR Validation "
            f"Cross-Entropy — {dataset_name}"
        )
    )

    ax.set_xticks(
        seeds
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    ax.text(
        0.99,
        0.02,
        (
            "Negative = lower CE for FiTabR\n"
            f"Mean Δ = "
            f"{delta_mean:.4f} ± "
            f"{delta_sd:.4f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "alpha": 0.15
        },
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{dataset}_validation_ce_difference.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 3. TEST ROC-AUC BY SEED
# ============================================================

def plot_test_auc(
    dataset,
    dataset_name,
    paired_results,
):

    seeds = get_seeds(
        paired_results
    )

    tabr = get_array(
        paired_results,
        "tabr",
        "test_auc",
    )

    fitabr = get_array(
        paired_results,
        "fitabr",
        "test_auc",
    )

    tabr_mean, tabr_sd = mean_sd(tabr)
    fitabr_mean, fitabr_sd = mean_sd(fitabr)

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.scatter(
        seeds - 0.08,
        tabr,
        s=75,
        label="TabR",
    )

    ax.scatter(
        seeds + 0.08,
        fitabr,
        s=75,
        label="FiTabR",
    )

    ax.axhline(
        tabr_mean,
        linestyle="--",
        linewidth=1.4,
        label=(
            f"TabR mean = "
            f"{tabr_mean:.3f}"
        ),
    )

    ax.axhline(
        fitabr_mean,
        linestyle="--",
        linewidth=1.4,
        label=(
            f"FiTabR mean = "
            f"{fitabr_mean:.3f}"
        ),
    )

    ax.set_xlabel(
        "Random Seed"
    )

    ax.set_ylabel(
        "Test ROC-AUC"
    )

    ax.set_title(
        (
            "Test ROC-AUC Across Random Seeds — "
            f"{dataset_name}"
        )
    )

    ax.set_xticks(
        seeds
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    ax.text(
        0.99,
        0.02,
        (
            f"TabR: "
            f"{tabr_mean:.3f} ± "
            f"{tabr_sd:.3f}\n"
            f"FiTabR: "
            f"{fitabr_mean:.3f} ± "
            f"{fitabr_sd:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "alpha": 0.15
        },
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{dataset}_test_auc_tabr_vs_fitabr.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 4. TEST ROC-AUC DIFFERENCE
# ============================================================

def plot_test_auc_difference(
    dataset,
    dataset_name,
    paired_results,
):

    seeds = get_seeds(
        paired_results
    )

    tabr = get_array(
        paired_results,
        "tabr",
        "test_auc",
    )

    fitabr = get_array(
        paired_results,
        "fitabr",
        "test_auc",
    )

    # Positive means FiTabR has higher AUC.
    delta = (
        fitabr - tabr
    )

    delta_mean, delta_sd = mean_sd(
        delta
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.scatter(
        seeds,
        delta,
        s=80,
    )

    ax.axhline(
        0,
        linewidth=1.5,
    )

    ax.axhline(
        delta_mean,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Mean Δ = "
            f"{delta_mean:.4f}"
        ),
    )

    ax.set_xlabel(
        "Random Seed"
    )

    ax.set_ylabel(
        "Δ Test ROC-AUC\n"
        "(FiTabR − TabR)"
    )

    ax.set_title(
        (
            "FiTabR − TabR Test ROC-AUC — "
            f"{dataset_name}"
        )
    )

    ax.set_xticks(
        seeds
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    ax.text(
        0.99,
        0.02,
        (
            "Positive = higher AUC for FiTabR\n"
            f"Mean Δ = "
            f"{delta_mean:.4f} ± "
            f"{delta_sd:.4f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "alpha": 0.15
        },
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{dataset}_test_auc_difference.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 5. BEST EPOCH TABR VS FITABR
# ============================================================

def plot_best_epoch(
    dataset,
    dataset_name,
    paired_results,
):

    seeds = get_seeds(
        paired_results
    )

    tabr = get_array(
        paired_results,
        "tabr",
        "best_epoch",
    )

    fitabr = get_array(
        paired_results,
        "fitabr",
        "best_epoch",
    )

    tabr_mean, tabr_sd = mean_sd(tabr)
    fitabr_mean, fitabr_sd = mean_sd(fitabr)

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.scatter(
        seeds - 0.08,
        tabr,
        s=75,
        label="TabR",
    )

    ax.scatter(
        seeds + 0.08,
        fitabr,
        s=75,
        label="FiTabR",
    )

    ax.axhline(
        tabr_mean,
        linestyle="--",
        linewidth=1.4,
        label=(
            f"TabR mean = "
            f"{tabr_mean:.1f}"
        ),
    )

    ax.axhline(
        fitabr_mean,
        linestyle="--",
        linewidth=1.4,
        label=(
            f"FiTabR mean = "
            f"{fitabr_mean:.1f}"
        ),
    )

    ax.set_xlabel(
        "Random Seed"
    )

    ax.set_ylabel(
        "Selected Best Epoch"
    )

    ax.set_title(
        (
            "Selected Best Epoch Across "
            f"Random Seeds — {dataset_name}"
        )
    )

    ax.set_xticks(
        seeds
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    ax.text(
        0.99,
        0.02,
        (
            f"TabR: "
            f"{tabr_mean:.1f} ± "
            f"{tabr_sd:.1f}\n"
            f"FiTabR: "
            f"{fitabr_mean:.1f} ± "
            f"{fitabr_sd:.1f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "alpha": 0.15
        },
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{dataset}_best_epoch_tabr_vs_fitabr.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 6. MEAN ± SD VALIDATION CE SUMMARY
# ============================================================

def plot_ce_summary(
    all_results,
):

    dataset_names = []

    tabr_means = []
    tabr_stds = []

    fitabr_means = []
    fitabr_stds = []

    for dataset, dataset_name in (
        DATASETS.items()
    ):

        paired = all_results[
            dataset
        ]

        tabr = get_array(
            paired,
            "tabr",
            "val_ce",
        )

        fitabr = get_array(
            paired,
            "fitabr",
            "val_ce",
        )

        tm, ts = mean_sd(tabr)
        fm, fs = mean_sd(fitabr)

        dataset_names.append(
            dataset_name
        )

        tabr_means.append(tm)
        tabr_stds.append(ts)

        fitabr_means.append(fm)
        fitabr_stds.append(fs)

    positions = np.arange(
        len(dataset_names)
    )

    offset = 0.12

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.errorbar(
        positions - offset,
        tabr_means,
        yerr=tabr_stds,
        fmt="o",
        capsize=6,
        markersize=8,
        label="TabR",
    )

    ax.errorbar(
        positions + offset,
        fitabr_means,
        yerr=fitabr_stds,
        fmt="o",
        capsize=6,
        markersize=8,
        label="FiTabR",
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        dataset_names
    )

    ax.set_ylabel(
        "Validation Cross-Entropy"
    )

    ax.set_title(
        (
            "TabR vs FiTabR Validation "
            "Cross-Entropy — Mean ± SD"
        )
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "validation_ce_mean_sd_tabr_vs_fitabr.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 7. TEST ROC-AUC MEAN ± SD SUMMARY
# ============================================================

def plot_auc_summary(
    all_results,
):

    dataset_names = []

    tabr_means = []
    tabr_stds = []

    fitabr_means = []
    fitabr_stds = []

    for dataset, dataset_name in (
        DATASETS.items()
    ):

        paired = all_results[
            dataset
        ]

        tabr = get_array(
            paired,
            "tabr",
            "test_auc",
        )

        fitabr = get_array(
            paired,
            "fitabr",
            "test_auc",
        )

        tm, ts = mean_sd(tabr)
        fm, fs = mean_sd(fitabr)

        dataset_names.append(
            dataset_name
        )

        tabr_means.append(tm)
        tabr_stds.append(ts)

        fitabr_means.append(fm)
        fitabr_stds.append(fs)

    positions = np.arange(
        len(dataset_names)
    )

    offset = 0.12

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.errorbar(
        positions - offset,
        tabr_means,
        yerr=tabr_stds,
        fmt="o",
        capsize=6,
        markersize=8,
        label="TabR",
    )

    ax.errorbar(
        positions + offset,
        fitabr_means,
        yerr=fitabr_stds,
        fmt="o",
        capsize=6,
        markersize=8,
        label="FiTabR",
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        dataset_names
    )

    ax.set_ylabel(
        "Test ROC-AUC"
    )

    ax.set_title(
        (
            "TabR vs FiTabR Test ROC-AUC "
            "— Mean ± SD"
        )
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "test_auc_mean_sd_tabr_vs_fitabr.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PRINT PAIRED SUMMARY
# ============================================================

def print_dataset_summary(
    dataset_name,
    paired_results,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"TabR vs FiTabR — "
        f"{dataset_name}"
    )

    print(
        "=" * 70
    )

    print(
        f"Paired seeds: "
        f"{len(paired_results)}"
    )

    # --------------------------------------------------------
    # Validation CE
    # --------------------------------------------------------

    tabr_ce = get_array(
        paired_results,
        "tabr",
        "val_ce",
    )

    fitabr_ce = get_array(
        paired_results,
        "fitabr",
        "val_ce",
    )

    ce_delta = (
        fitabr_ce - tabr_ce
    )

    tabr_ce_mean, tabr_ce_sd = mean_sd(
        tabr_ce
    )

    fitabr_ce_mean, fitabr_ce_sd = mean_sd(
        fitabr_ce
    )

    ce_delta_mean, ce_delta_sd = mean_sd(
        ce_delta
    )

    print()
    print(
        "VALIDATION CROSS-ENTROPY"
    )

    print(
        f"TabR:   "
        f"{tabr_ce_mean:.4f} ± "
        f"{tabr_ce_sd:.4f}"
    )

    print(
        f"FiTabR: "
        f"{fitabr_ce_mean:.4f} ± "
        f"{fitabr_ce_sd:.4f}"
    )

    print(
        f"Δ FiTabR − TabR: "
        f"{ce_delta_mean:.4f} ± "
        f"{ce_delta_sd:.4f}"
    )

    print(
        "FiTabR lower CE in "
        f"{np.sum(ce_delta < 0)} / "
        f"{len(ce_delta)} seeds"
    )

    # --------------------------------------------------------
    # Test AUC
    # --------------------------------------------------------

    tabr_auc = get_array(
        paired_results,
        "tabr",
        "test_auc",
    )

    fitabr_auc = get_array(
        paired_results,
        "fitabr",
        "test_auc",
    )

    auc_delta = (
        fitabr_auc - tabr_auc
    )

    tabr_auc_mean, tabr_auc_sd = mean_sd(
        tabr_auc
    )

    fitabr_auc_mean, fitabr_auc_sd = mean_sd(
        fitabr_auc
    )

    auc_delta_mean, auc_delta_sd = mean_sd(
        auc_delta
    )

    print()
    print(
        "TEST ROC-AUC"
    )

    print(
        f"TabR:   "
        f"{tabr_auc_mean:.4f} ± "
        f"{tabr_auc_sd:.4f}"
    )

    print(
        f"FiTabR: "
        f"{fitabr_auc_mean:.4f} ± "
        f"{fitabr_auc_sd:.4f}"
    )

    print(
        f"Δ FiTabR − TabR: "
        f"{auc_delta_mean:.4f} ± "
        f"{auc_delta_sd:.4f}"
    )

    print(
        "FiTabR higher AUC in "
        f"{np.sum(auc_delta > 0)} / "
        f"{len(auc_delta)} seeds"
    )

    # --------------------------------------------------------
    # Best epoch
    # --------------------------------------------------------

    tabr_epoch = get_array(
        paired_results,
        "tabr",
        "best_epoch",
    )

    fitabr_epoch = get_array(
        paired_results,
        "fitabr",
        "best_epoch",
    )

    tabr_epoch_mean, tabr_epoch_sd = mean_sd(
        tabr_epoch
    )

    fitabr_epoch_mean, fitabr_epoch_sd = mean_sd(
        fitabr_epoch
    )

    print()
    print(
        "SELECTED BEST EPOCH"
    )

    print(
        f"TabR:   "
        f"{tabr_epoch_mean:.1f} ± "
        f"{tabr_epoch_sd:.1f}"
    )

    print(
        f"FiTabR: "
        f"{fitabr_epoch_mean:.1f} ± "
        f"{fitabr_epoch_sd:.1f}"
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    all_results,
):

    output_path = (
        OUTPUT_DIR
        / "tabr_vs_fitabr_seed_comparison.csv"
    )

    columns = [
        "dataset",
        "seed",

        "tabr_best_epoch",
        "fitabr_best_epoch",

        "tabr_val_ce",
        "fitabr_val_ce",
        "delta_val_ce",

        "tabr_test_ce",
        "fitabr_test_ce",
        "delta_test_ce",

        "tabr_val_auc",
        "fitabr_val_auc",
        "delta_val_auc",

        "tabr_test_auc",
        "fitabr_test_auc",
        "delta_test_auc",
    ]

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        for dataset in DATASETS:

            for pair in all_results[
                dataset
            ]:

                tabr = pair["tabr"]
                fitabr = pair["fitabr"]

                writer.writerow(
                    {
                        "dataset":
                            dataset,

                        "seed":
                            pair["seed"],

                        "tabr_best_epoch":
                            tabr["best_epoch"],

                        "fitabr_best_epoch":
                            fitabr["best_epoch"],

                        "tabr_val_ce":
                            tabr["val_ce"],

                        "fitabr_val_ce":
                            fitabr["val_ce"],

                        "delta_val_ce":
                            (
                                fitabr["val_ce"]
                                - tabr["val_ce"]
                            ),

                        "tabr_test_ce":
                            tabr["test_ce"],

                        "fitabr_test_ce":
                            fitabr["test_ce"],

                        "delta_test_ce":
                            (
                                fitabr["test_ce"]
                                - tabr["test_ce"]
                            ),

                        "tabr_val_auc":
                            tabr["val_auc"],

                        "fitabr_val_auc":
                            fitabr["val_auc"],

                        "delta_val_auc":
                            (
                                fitabr["val_auc"]
                                - tabr["val_auc"]
                            ),

                        "tabr_test_auc":
                            tabr["test_auc"],

                        "fitabr_test_auc":
                            fitabr["test_auc"],

                        "delta_test_auc":
                            (
                                fitabr["test_auc"]
                                - tabr["test_auc"]
                            ),
                    }
                )

    print(
        f"Saved CSV: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "TabR vs FiTabR "
        "Seed-Level Comparison"
    )

    print(
        "=" * 70
    )

    print(
        f"TabR root:\n"
        f"{TABR_ROOT}"
    )

    print(
        f"\nFiTabR root:\n"
        f"{FITABR_ROOT}"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    all_results = {}

    for dataset, dataset_name in (
        DATASETS.items()
    ):

        print()
        print(
            f"Loading {dataset_name}..."
        )

        paired_results = (
            load_paired_results(
                dataset
            )
        )

        if len(
            paired_results
        ) == 0:

            print(
                "No paired results found."
            )

            continue

        print(
            f"Loaded "
            f"{len(paired_results)} "
            "paired seeds."
        )

        all_results[
            dataset
        ] = paired_results

        # ----------------------------------------
        # Validation CE
        # ----------------------------------------

        plot_validation_ce(
            dataset,
            dataset_name,
            paired_results,
        )

        plot_validation_ce_difference(
            dataset,
            dataset_name,
            paired_results,
        )

        # ----------------------------------------
        # Test ROC-AUC
        # ----------------------------------------

        plot_test_auc(
            dataset,
            dataset_name,
            paired_results,
        )

        plot_test_auc_difference(
            dataset,
            dataset_name,
            paired_results,
        )

        # ----------------------------------------
        # Best epoch
        # ----------------------------------------

        plot_best_epoch(
            dataset,
            dataset_name,
            paired_results,
        )

        # ----------------------------------------
        # Print numerical results
        # ----------------------------------------

        print_dataset_summary(
            dataset_name,
            paired_results,
        )

    # ========================================================
    # Combined figures
    # ========================================================

    if all(
        dataset in all_results
        for dataset in DATASETS
    ):

        plot_ce_summary(
            all_results
        )

        plot_auc_summary(
            all_results
        )

        save_csv(
            all_results
        )

    print()
    print(
        "=" * 70
    )

    print(
        "Finished."
    )

    print(
        f"Results saved in:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()