import json
import pandas as pd
from pathlib import Path

reports = {
    "CatBoost": "exp/catboost_/heart/0-ensemble-5/0/report.json",
    "XGBoost": "exp/xgboost_/heart/0-ensemble-5/0/report.json",
    "KNN": "exp/knn/heart/0-ensemble-5/0/report.json",
    "Lightgbm": "exp/lightgbm_/heart/0-ensemble-5/0/report.json",
    "MLP": "exp/mlp/heart/0-ensemble-5/0/report.json",
    "Saint": "exp/saint/heart/0-ensemble-5/0/report.json",
    "FT-Transformer": "exp/ft_transformer/heart/0-ensemble-5/0/report.json",
    "TabR": "exp/tabr/heart/0-ensemble-5/0/report.json",
    "FiTabR": "exp/fitabr/heart/0-ensemble-5/0/report.json",
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

df.to_csv("notebooks/heart_comparison_table.csv", index=False)
df.to_latex("notebooks/heart_comparison_table.tex", index=False)