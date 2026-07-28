import json
import pandas as pd
from pathlib import Path

reports = {
    "TabR": "exp/tabr/heart/0-ensemble-5/0/report.json",
    "FitabR": "exp/fitabr/heart/0-ensemble-5/0/report.json",
    "FitabR_No_Residual": "exp/fitabr_no_residual/heart/0-ensemble-5/0/report.json",
    "FitabR_No_Scale": "exp/fitabr_no_scale/heart/0-ensemble-5/0/report.json",
    "FitabR_Uniform": "exp/fitabr_uniform/heart/0-ensemble-5/0/report.json",
}

rows = []

for model, path in reports.items():
    path = Path(path)
    if not path.exists():
        continue

    r = json.load(open(path))
    test = r["metrics"]["test"]

    rows.append({
        "Model": model,
        "Accuracy": test["accuracy"],
        "Precision": test["weighted avg"]["precision"],
        "Recall": test["weighted avg"]["recall"],
        "F1-score": test["weighted avg"]["f1-score"],
        "ROC-AUC": test["roc-auc"],
        "Cross-Entropy": test["cross-entropy"],
    })

df = pd.DataFrame(rows)
df = df.round(4)
# display(df)

df.to_csv("notebook_ablation/heart_comparison_table.csv", index=False)
df.to_latex("notebook_ablation/heart_comparison_table.tex", index=False)