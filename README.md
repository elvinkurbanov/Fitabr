# FiTabR: Feature-Aware Representation Learning for Retrieval-Augmented Tabular Learning

FiTabR is a research implementation developed for a Master's thesis on retrieval-augmented learning for medical tabular classification.

The method extends TabR by introducing a sample-specific feature-attention module for numerical features before the TabR encoder and retrieval stage. The original TabR retrieval, contextual aggregation, and prediction framework is retained.

This design allows the effect of feature-aware input representation to be evaluated while preserving the underlying retrieval-based architecture.

---

## 1. Research Motivation

TabR combines deep tabular representation learning with retrieval from neighboring training examples. FiTabR investigates whether the numerical representation supplied to this retrieval framework can be improved through sample-specific feature reweighting.

For a numerical feature vector \(x_i\), FiTabR learns an attention vector \(a_i\) and applies residual feature modulation:

\[
\tilde{x}_i = x_i \odot (1 + \alpha a_i),
\]

where:

- \(a_i\) is a sample-specific attention distribution;
- \(\alpha = 2.0\) is the attention scaling coefficient;
- \(\odot\) denotes element-wise multiplication.

The resulting numerical representation is subsequently processed by the standard TabR encoder and retrieval pipeline.

---

## 2. Research Objectives

The project addresses the following objectives:

1. Evaluate whether feature-aware numerical representation can improve retrieval-augmented tabular classification.
2. Compare FiTabR directly with the original TabR architecture under a controlled experimental setting.
3. Compare FiTabR with classical machine-learning and deep tabular baselines.
4. Analyze the contribution of individual FiTabR design components through ablation experiments.
5. Examine learned feature-attention patterns to characterize how FiTabR reweights numerical clinical variables.
6. Investigate how model behavior changes across medical datasets with different predictive and class-distribution characteristics.

---

## 3. Proposed FiTabR Architecture

FiTabR modifies only the numerical input representation before the TabR encoder.

The main processing sequence is:

```text
Numerical Features
       |
       v
Feature-Attention Network
       |
       v
Softmax Attention Weights
       |
       v
Attention Scaling
   alpha = 2.0
       |
       v
Residual Feature Weighting
x_weighted = x * (1 + attention)
       |
       v
TabR Encoder
       |
       v
Candidate Retrieval
       |
       v
Contextual Aggregation
       |
       v
Prediction Head
```

---

## 4. Repository Structure

```text
Fitabr/
|
├── bin/
│   ├── fitabr.py                 # Proposed FiTabR model
│   ├── tabr.py                   # Original TabR baseline
│   ├── fitabr_no_residual.py     # Ablation: no residual weighting
│   ├── fitabr_no_scale.py        # Ablation: no attention scaling
│   ├── fitabr_uniform.py         # Ablation: uniform feature weighting
│   ├── catboost_.py              # CatBoost baseline
│   ├── lightgbm_.py              # LightGBM baseline
│   ├── xgboost_.py               # XGBoost baseline
│   ├── ft_transformer.py         # FT-Transformer baseline
│   ├── saint.py                  # SAINT baseline
│   ├── knn.py                    # k-NN baseline
│   ├── ensemble.py               # Ensemble construction
│   ├── evaluate.py               # Model evaluation
│   └── go.py                     # Tuning/evaluation/ensemble pipeline
│
├── exp/
│   ├── fitabr/                   # FiTabR experiment configurations/results
│   ├── tabr/                     # TabR experiments
│   ├── fitabr_no_residual/       # No-residual experiments
│   ├── fitabr_no_scale/          # No-scale experiments
│   ├── fitabr_uniform/           # Uniform-weight experiments
│   └── ...                       # Other baseline experiments
│
├── lib/
│   └── Shared data, training, preprocessing, and utility modules
│
├── data/
│   └── Experimental datasets
│
├── notebooks/
│   └── Experimental and interpretability analysis notebooks
│
├── notebook_ablation/
│   ├── heart_ablation_report.py
│   ├── pima_ablation_report.py
│   ├── stroke_ablation_report.py
│   └── Publication-ready ablation tables
│
├── environment-simple.yaml       # Recommended Conda research environment
├── environment.yaml              # Full environment snapshot
├── requirements.txt              # Core Python dependencies
└── README.md
```

---

## 5. Environment Setup

Clone the repository:

```bash
git clone https://github.com/elvinkurbanov/Fitabr.git
cd Fitabr
```

The recommended experimental environment is provided in `environment-simple.yaml`:

```bash
conda env create -f environment-simple.yaml
conda activate tabr
```

Alternatively, the Python dependencies can be installed with:

```bash
pip install -r requirements.txt
```

For reproducing the original GPU-based experimental environment, the Conda environment is recommended.

---

## 6. Experimental Protocol

Final evaluation uses **15 trained models** divided into **three five-model ensembles**.

Predictions are averaged within each ensemble, and results are reported as the **mean and sample standard deviation across the three ensembles**.

Unless otherwise specified, **precision, recall, and F1-score are computed as support-weighted averages across classes**. Positive-class and class-balanced metrics are reported separately for the Stroke Prediction analysis.

---

## 7. Running Experiments

Experiment configurations are stored under `exp/`.

For example, the FiTabR Heart Disease experiment can be started with:

```bash
python bin/go.py exp/fitabr/heart/0-tuning.toml
```

Equivalent configurations are provided for TabR, FiTabR ablation variants, and comparison models.

---

## 8. Research Status

The experimental implementation used for the Master's thesis is finalized.

The repository is retained for:

- reproducibility of the reported experiments;
- generation of thesis tables and figures;
- ablation analysis;
- attention analysis;
- documentation of the FiTabR implementation.

Further model development is outside the scope of the current thesis unless a numerical or reproducibility issue is identified.

---

## 9. Base Method

FiTabR is built on the TabR architecture introduced by Gorishniy et al.:

> **TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning**

FiTabR retains the core TabR retrieval framework while introducing feature-aware numerical representation before the encoder and retrieval stage.

---

## 10. License

See the repository `LICENSE` file for licensing information.
