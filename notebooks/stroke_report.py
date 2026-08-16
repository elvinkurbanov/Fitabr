import json
import pandas as pd
from pathlib import Path

reports = {
    "CatBoost": "exp/catboost_/stroke/0-ensemble-5/0/report.json",
    "XGBoost": "exp/xgboost_/stroke/0-ensemble-5/0/report.json",
    "Lightgbm": "exp/lightgbm_/stroke/0-ensemble-5/0/report.json",
    "MLP": "exp/mlp/stroke/0-ensemble-5/0/report.json",
    "Saint": "exp/saint/stroke/0-ensemble-5/0/report.json",
    "FT-Transformer": "exp/ft_transformer/stroke/0-ensemble-5/0/report.json",
    "TabR": "exp/tabr/stroke/0-ensemble-5/0/report.json",
    "FiTabR": "exp/fitabr/stroke/0-ensemble-5/0/report.json",
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


df.to_csv("notebooks/stroke_comparison_table.csv", index=False)
df.to_latex("notebooks/stroke_comparison_table.tex", index=False)


print("Saved stroke_comparison_table.csv")
print("Saved stroke_comparison_table.tex")