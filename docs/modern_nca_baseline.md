# ModernNCA baseline

This repository adaptation adds a deep learned embedding and a soft nearest-neighbor classifier to the existing experiment runner. It supports binary and multiclass classification, uses the repository data preprocessing and train/validation/test partitions, and keeps the FiTabR and TabR implementations unchanged.

Training uses stochastic candidate neighborhoods. For each query batch, the sampled candidate set is augmented so every query has at least one same-class neighbor; the query itself is excluded from its neighbor distribution. Validation and test predictions use the full training set as the candidate memory.

This is an in-framework ModernNCA-style implementation of the paper's learned metric and soft-neighbor prediction approach. It is an independently tuned baseline, not a claim of bit-for-bit reproduction of the authors' TALENT implementation or benchmark numbers. The current runner supports classification tasks only.

## Run

Tune each dataset with the same runner used by other models:

```bash
python bin/go.py exp/modern_nca/heart/0-tuning.toml
python bin/go.py exp/modern_nca/pima/0-tuning.toml
python bin/go.py exp/modern_nca/stroke/0-tuning.toml
```

Each configuration uses 100 Optuna trials, then the standard 15-seed evaluation and three five-model ensembles through `bin/go.py`. For a fair comparison, use the same data split and report the same metrics as the other baselines.
