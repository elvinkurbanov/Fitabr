import json
import pandas as pd
from pathlib import Path

reports = {
    "TabR": "exp/tabr/stroke/0-ensemble-5/0/report.json",
    "FitabR": "exp/fitabr/stroke/0-ensemble-5/0/report.json",
    "FitabR_No_Residual": "exp/fitabr_no_residual/stroke/0-ensemble-5/0/report.json",
    "FitabR_No_Scale": "exp/fitabr_no_scale/stroke/0-ensemble-5/0/report.json",
    "FitabR_Uniform": "exp/fitabr_uniform/stroke/0-ensemble-5/0/report.json",
}

rows = []

for model, path in reports.items():
    path = Path(path)
    if not path.exists():
        print(f"Missing: {path}")
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
if df.empty:
    raise ValueError("No report.json files were found. Check your paths.")


df.to_csv("notebook_ablation/stroke_comparison_table.csv", index=False)
df.to_latex("notebook_ablation/stroke_comparison_table.tex", index=False)


print("Saved stroke_comparison_table.csv")
print("Saved stroke_comparison_table.tex")