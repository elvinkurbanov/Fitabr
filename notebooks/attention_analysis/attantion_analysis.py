import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# FiTabR tuning reports
# ============================================================

REPORT_PATHS = {
    "heart": Path("exp/fitabr/heart/0-tuning/report.json"),
    "pima": Path("exp/fitabr/pima/0-tuning/report.json"),
    "stroke": Path("exp/fitabr/stroke/0-tuning/report.json"),
}


# ============================================================
# Numerical feature names
# IMPORTANT:
# Order matches the FiTabR attention vector in report.json
# ============================================================

FEATURE_NAMES = {

    "heart": [
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak",
    ],

    "pima": [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "Diabetes\nPedigreeFunction",
        "Age",
    ],

    "stroke": [
        "Age",
        "Hypertension",
        "Heart disease",
        "Average glucose\nlevel",
        "BMI",
    ],
}


# ============================================================
# Load attention mean and std
# ============================================================

def load_attention(report_path):

    if not report_path.exists():
        raise FileNotFoundError(
            f"Report not found: {report_path}"
        )

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    # Tuning report
    if (
        "best" in report
        and isinstance(report["best"], dict)
        and "attention" in report["best"]
    ):
        attention = report["best"]["attention"]

    # Normal experiment report
    elif "attention" in report:
        attention = report["attention"]

    else:
        raise KeyError(
            f"No attention section found in: {report_path}"
        )

    mean = np.asarray(
        attention["mean"],
        dtype=float,
    )

    std = np.asarray(
        attention["std"],
        dtype=float,
    )

    return mean, std


# ============================================================
# Output directory
# ============================================================

figure_dir = Path("notebooks/attention_analysis/figures")
figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Draw figure for every dataset
# ============================================================

for dataset, report_path in REPORT_PATHS.items():

    attention_mean, attention_std = load_attention(
        report_path
    )

    feature_names = FEATURE_NAMES[dataset]

    if len(attention_mean) != len(feature_names):
        raise ValueError(
            f"{dataset}: report contains "
            f"{len(attention_mean)} attention values, "
            f"but {len(feature_names)} feature names were supplied."
        )

    # --------------------------------------------------------
    # Uniform attention
    # --------------------------------------------------------

    n_features = len(feature_names)

    uniform_attention = np.full(
        n_features,
        1.0 / n_features,
    )

    # --------------------------------------------------------
    # Plot positions
    # --------------------------------------------------------

    x = np.arange(n_features)

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    # --------------------------------------------------------
    # Learned FiTabR attention
    # Mean + standard deviation
    # --------------------------------------------------------

    bars_attention = ax.bar(
        x - width / 2,
        attention_mean,
        width,
        yerr=attention_std,
        capsize=4,
        label="FiTabR Attention (Mean ± 1 SD)",
    )

    # --------------------------------------------------------
    # Uniform baseline
    # --------------------------------------------------------

    bars_uniform = ax.bar(
        x + width / 2,
        uniform_attention,
        width,
        label="Uniform Attention",
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_ylabel(
        "Attention Weight"
    )

    ax.set_xlabel(
        "Feature"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        feature_names,
        rotation=30,
        ha="right",
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Value labels: FiTabR attention
    # --------------------------------------------------------

    for bar, mean, std in zip(
        bars_attention,
        attention_mean,
        attention_std,
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.006,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Value labels: uniform baseline
    # --------------------------------------------------------

    for bar, value in zip(
        bars_uniform,
        uniform_attention,
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Y limit
    # --------------------------------------------------------

    max_attention = np.max(
        attention_mean + attention_std
    )

    max_uniform = np.max(
        uniform_attention
    )

    upper_limit = max(
        max_attention,
        max_uniform,
    )

    ax.set_ylim(
        0,
        upper_limit * 1.25,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pdf_path = (
        figure_dir
        / f"{dataset}_attention_analysis.pdf"
    )

    png_path = (
        figure_dir
        / f"{dataset}_attention_analysis.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"{dataset.capitalize()} attention figure saved:"
    )
    print(f"  {pdf_path}")
    print(f"  {png_path}")

    plt.show()

    plt.close(fig)