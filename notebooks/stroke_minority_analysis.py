import json
from pathlib import Path

import pandas as pd


MODEL_DIRS = {
    "TabR": Path("exp/tabr/stroke/0-ensemble-5"),
    "FiTabR": Path("exp/fitabr/stroke/0-ensemble-5"),
}

ENSEMBLE_IDS = [0, 1, 2]

rows = []

for model_name, model_dir in MODEL_DIRS.items():

    for ensemble_id in ENSEMBLE_IDS:

        report_path = (
            model_dir
            / str(ensemble_id)
            / "report.json"
        )

        if not report_path.exists():
            print(f"Missing report: {report_path}")
            continue

        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        test = report["metrics"]["test"]

        # Positive class = class 1
        positive = test["1"]

        # Balanced accuracy =
        # (specificity + sensitivity) / 2
        balanced_accuracy = (
            test["0"]["recall"]
            + test["1"]["recall"]
        ) / 2.0

        rows.append(
            {
                "Model": model_name,
                "Ensemble": ensemble_id,

                "Accuracy":
                    test["accuracy"],

                "Positive Precision":
                    positive["precision"],

                "Positive Recall":
                    positive["recall"],

                "Positive F1":
                    positive["f1-score"],

                "Macro F1":
                    test["macro avg"]["f1-score"],

                "Balanced Accuracy":
                    balanced_accuracy,

                "ROC-AUC":
                    test["roc-auc"],

                "Cross-Entropy":
                    test["cross-entropy"],
            }
        )


df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError(
        "No Stroke report.json files found."
    )


print("\nIndividual ensemble results:\n")
print(df.to_string(index=False))


metric_columns = [
    "Accuracy",
    "Positive Precision",
    "Positive Recall",
    "Positive F1",
    "Macro F1",
    "Balanced Accuracy",
    "ROC-AUC",
    "Cross-Entropy",
]


mean_df = (
    df.groupby("Model", sort=False)[metric_columns]
    .mean()
)

std_df = (
    df.groupby("Model", sort=False)[metric_columns]
    .std(ddof=1)
)


publication_rows = []

for model in mean_df.index:

    row = {"Model": model}

    for metric in metric_columns:

        mean = mean_df.loc[model, metric]
        std = std_df.loc[model, metric]

        row[metric] = (
            f"{mean:.4f} ± {std:.4f}"
        )

    publication_rows.append(row)


publication_df = pd.DataFrame(
    publication_rows
)

print("\nPublication table:\n")
print(publication_df.to_string(index=False))


output_path = Path(
    "notebooks/stroke_minority_metrics.csv"
)

publication_df.to_csv(
    output_path,
    index=False,
)

print(f"\nSaved: {output_path}")