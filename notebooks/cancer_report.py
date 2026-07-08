import json
import pandas as pd
from pathlib import Path

reports = {
    "CatBoost": "exp/catboost_/breast_cancer/0-ensemble-5/0/report.json",
    "XGBoost": "exp/xgboost_/breast_cancer/0-ensemble-5/0/report.json",
    "KNN": "exp/knn/breast_cancer/0-ensemble-5/0/report.json",
    "Lightgbm": "exp/lightgbm_/breast_cancer/0-ensemble-5/0/report.json",
    "MLP": "exp/mlp/breast_cancer/0-ensemble-5/0/report.json",
    "Saint": "exp/saint/breast_cancer/0-ensemble-5/0/report.json",
    "FT-Transformer": "exp/ft_transformer/breast_cancer/0-ensemble-5/0/report.json",
    "TabR": "exp/tabr/breast_cancer/0-ensemble-5/0/report.json",
    "FiTabR": "exp/fitabr/breast_cancer/0-ensemble-5/0/report.json",
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

df.to_csv("notebooks/cancer_comparison_table.csv", index=False)
df.to_latex("notebooks/cancer_comparison_table.tex", index=False)