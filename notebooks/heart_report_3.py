import json
from pathlib import Path

import pandas as pd


MODEL_DIRS = {
    "CatBoost": "exp/catboost_/heart/0-ensemble-5",
    "XGBoost": "exp/xgboost_/heart/0-ensemble-5",
    "KNN": "exp/knn/heart/0-ensemble-5",
    "LightGBM": "exp/lightgbm_/heart/0-ensemble-5",
    "MLP": "exp/mlp/heart/0-ensemble-5",
    "SAINT": "exp/saint/heart/0-ensemble-5",
    "FT-Transformer": "exp/ft_transformer/heart/0-ensemble-5",
    "TabR": "exp/tabr/heart/0-ensemble-5",
    "FiTabR": "exp/fitabr/heart/0-ensemble-5",
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

output_dir = Path("notebooks")
output_dir.mkdir(parents=True, exist_ok=True)

publication_df.to_csv(
    output_dir / "heart_ensemble_publication_table.csv",
    index=False,
)

publication_df.to_latex(
    output_dir / "heart_ensemble_publication_table.tex",
    index=False,
    escape=False,
)

print(publication_df.to_string(index=False))