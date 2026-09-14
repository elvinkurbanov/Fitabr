import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_DIRS = {
      "TabR": "exp/tabr/pima/0-ensemble-5",
    "FitabR": "exp/fitabr/pima/0-ensemble-5",
    "FitabR_No_Residual": "exp/fitabr_no_residual/pima/0-ensemble-5",
    "FitabR_No_Scale": "exp/fitabr_no_scale/pima/0-ensemble-5",
    "FitabR_Uniform": "exp/fitabr_uniform/pima/0-ensemble-5",
}

ENSEMBLE_IDS = [0, 1, 2]

rows = []

for model_name, ensemble_dir in MODEL_DIRS.items():
    ensemble_dir = Path(ensemble_dir)

    for ensemble_id in ENSEMBLE_IDS:
        report_path = ensemble_dir / str(ensemble_id) / "report.json"

        if not report_path.exists():
            print(f"Missing: {report_path}")
            continue

        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        test = report["metrics"]["test"]

        rows.append(
            {
                "Model": model_name,
                "Accuracy": test["accuracy"],
                "Precision": test["weighted avg"]["precision"],
                "Recall": test["weighted avg"]["recall"],
                "F1-score": test["weighted avg"]["f1-score"],
                "ROC-AUC": test["roc-auc"],
                "Cross-Entropy": test["cross-entropy"],
            }
        )


results_df = pd.DataFrame(rows)

if results_df.empty:
    raise RuntimeError("No ensemble reports were found.")


# Mean and sample standard deviation across three ensembles
mean_df = results_df.groupby("Model", sort=False).mean()
std_df = results_df.groupby("Model", sort=False).std(ddof=1)


publication_rows = []

for model_name in mean_df.index:
    publication_rows.append(
        {
            "Model": model_name,
            "Accuracy": (
                f"{mean_df.loc[model_name, 'Accuracy']:.4f} ± "
                f"{std_df.loc[model_name, 'Accuracy']:.4f}"
            ),
            "Precision": (
                f"{mean_df.loc[model_name, 'Precision']:.4f} ± "
                f"{std_df.loc[model_name, 'Precision']:.4f}"
            ),
            "Recall": (
                f"{mean_df.loc[model_name, 'Recall']:.4f} ± "
                f"{std_df.loc[model_name, 'Recall']:.4f}"
            ),
            "F1-score": (
                f"{mean_df.loc[model_name, 'F1-score']:.4f} ± "
                f"{std_df.loc[model_name, 'F1-score']:.4f}"
            ),
            "ROC-AUC": (
                f"{mean_df.loc[model_name, 'ROC-AUC']:.4f} ± "
                f"{std_df.loc[model_name, 'ROC-AUC']:.4f}"
            ),
            "Cross-Entropy": (
                f"{mean_df.loc[model_name, 'Cross-Entropy']:.4f} ± "
                f"{std_df.loc[model_name, 'Cross-Entropy']:.4f}"
            ),
        }
    )


publication_df = pd.DataFrame(publication_rows)

output_dir = Path("notebook_ablation")
output_dir.mkdir(parents=True, exist_ok=True)

publication_df.to_csv(
    output_dir / "pima_comparison_table.csv",
    index=False,
)

publication_df.to_latex(
    output_dir / "pima_comparison_table.tex",
    index=False,
    escape=False,
)

print(publication_df.to_string(index=False))


# ============================================================
# Plot Pima ablation results
# ============================================================

models = list(mean_df.index)

x = np.arange(len(models))
width = 0.24

fig, ax = plt.subplots(figsize=(8, 4.8))

ax.bar(
    x - width,
    mean_df["Accuracy"],
    width,
    yerr=std_df["Accuracy"],
    capsize=3,
    label="Accuracy",
)

ax.bar(
    x,
    mean_df["F1-score"],
    width,
    yerr=std_df["F1-score"],
    capsize=3,
    label="F1-score",
)

ax.bar(
    x + width,
    mean_df["ROC-AUC"],
    width,
    yerr=std_df["ROC-AUC"],
    capsize=3,
    label="ROC-AUC",
)

ax.set_ylabel("Score")

ax.set_xticks(x)
ax.set_xticklabels(
    [
        "TabR",
        "FiTabR",
        "No Residual",
        r"$\alpha=1$",
        "Uniform",
    ]
)

# Good range for the Pima results
ax.set_ylim(0.72, 0.82)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3,
)

ax.set_axisbelow(True)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
)

fig.tight_layout()

# Create figure directory
figure_dir = Path("notebook_ablation/figures")
figure_dir.mkdir(parents=True, exist_ok=True)

# Vector version for thesis
fig.savefig(
    figure_dir / "pima_ablation_performance.pdf",
    bbox_inches="tight",
)

# PNG for preview / GitHub
fig.savefig(
    figure_dir / "pima_ablation_performance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
