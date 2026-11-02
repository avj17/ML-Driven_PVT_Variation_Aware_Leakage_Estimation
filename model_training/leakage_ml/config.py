"""Central configuration: paths, seeds, cases, features, split ratios and search spaces.

Every number that a script uses lives here so the report can quote one source of truth.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]           # model_training/
DATA_DIR = ROOT.parent / "data_set_generattion"      # CSVs from generate_data.py
OUT_DIR = Path(os.environ.get("LEAKAGE_OUT", ROOT))   # override for test runs
RESULTS_DIR = OUT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = OUT_DIR / "saved_models"
SPLITS_DIR = RESULTS_DIR / "splits"

# ----------------------------------------------------------------------------
# Reproducibility (Deliverable 9). Separate streams so changing one stage
# (e.g. MLP init) never changes another (e.g. the data split).
# ----------------------------------------------------------------------------
SEED = 42
SEED_SPLIT = SEED + 0
SEED_CV = SEED + 1
SEED_MODEL = SEED + 2
SEED_TORCH = SEED + 3

# ----------------------------------------------------------------------------
# Technology (team StaticSilicon: 45nm_HP, VDD 0.9 V, 3-bit CLA)
# ----------------------------------------------------------------------------
L_MIN = 45e-9
VDD_NOM = 0.9
K_BOLTZMANN = 1.380649e-23
Q_ELECTRON = 1.602176634e-19

# case -> (device type, gate state). Polarity p=+1 NMOS, -1 PMOS.
CASES = {
    "NMOS_ON":  {"mos": "nmos", "on": True},
    "NMOS_OFF": {"mos": "nmos", "on": False},
    "PMOS_ON":  {"mos": "pmos", "on": True},
    "PMOS_OFF": {"mos": "pmos", "on": False},
}

TARGETS = ["id", "ig", "is", "ib"]

# vg and vb are fixed by the case (0 or vdd), so they carry no information
# beyond vdd and are dropped (see data.drop_uninformative). Early CSVs named
# the vth0 column "vtho"; that name is still accepted and renamed on load.
RAW_FEATURES = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
                "vth0", "toxe", "ndep", "u0", "xj"]

# ----------------------------------------------------------------------------
# Split (spec: 70/15/15 with a fixed seed; 80/10/10 also allowed)
# ----------------------------------------------------------------------------
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
CV_FOLDS = 5

# ----------------------------------------------------------------------------
# Target transform: y = sign(I) * log10(1 + |I| / I0).
# I0 is fitted per target on the TRAIN set as I0_FRAC * percentile_1(|I|), so
# 99 % of samples sit in the logarithmic regime where an error dy maps to a
# relative current error of ~ln(10)*dy, while the sign is kept and the
# transform stays smooth through I = 0 (Id/Is change sign with Vds).
# ----------------------------------------------------------------------------
I0_FRAC = 0.1

# ----------------------------------------------------------------------------
# Model selection on the validation set: lowest mean (over the 4 terminals)
# median relative error; ties broken by mean 95th-percentile error.
# ----------------------------------------------------------------------------
MODEL_FAMILIES = ["poly_elasticnet", "random_forest", "xgboost", "mlp"]

SEARCH_SPACES = {
    # Pipeline: scale -> polynomial expansion -> scale -> ElasticNetCV.
    # alpha path and l1_ratio are chosen by CV inside ElasticNetCV; degree by CV score.
    "poly_elasticnet": {
        "degree": [2, 3],
        "l1_ratio": [0.1, 0.5, 0.9, 1.0],
        "n_alphas": 40,
        "max_iter": 20000,
    },
    # Random forest on the GPU (XGBoost RF mode: num_parallel_tree bagged deep
    # trees with per-node feature subsampling; no boosting). Grid search with
    # K-fold CV on TRAIN. RF_BACKEND = "sklearn" switches to the CPU
    # RandomForestRegressor tuned on out-of-bag R^2 instead.
    # The first full run (12-point grid incl. max_depth [12, 16] and
    # min_child_weight [1, 4]) took ~1.5 h per terminal; its 5-fold CV chose
    # max_depth 16 and min_child_weight 1 for Id in all four cases, so those are
    # fixed and only the per-node feature fraction is searched.
    "random_forest": {
        "grid": {
            "num_parallel_tree": [300],
            "max_depth": [16],
            "colsample_bynode": [0.33, 0.6, 1.0],
            "min_child_weight": [1],
            "subsample": [0.632],
        },
        "sklearn_grid": {
            "n_estimators": [400],
            "max_depth": [None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [0.33, 0.6, 1.0],
        },
    },
    # Randomised search with K-fold CV on the training set.
    "xgboost": {
        "n_iter": 25,
        "param_distributions": {
            "n_estimators": [800, 1500, 3000],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [4, 6, 8, 10],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [0.1, 1.0, 10.0],
            "reg_alpha": [0.0, 1e-3, 1e-1],
        },
    },
    # Each config is trained on 90 % of TRAIN with early stopping on the other
    # 10 % ("inner validation"); the official validation set is never seen.
    "mlp": {
        "configs": [
            {"hidden": [256, 256, 256], "activation": "silu", "dropout": 0.0, "weight_decay": 1e-6, "lr": 2e-3, "batch_size": 256},
            {"hidden": [512, 512, 256, 128], "activation": "silu", "dropout": 0.0, "weight_decay": 1e-6, "lr": 2e-3, "batch_size": 256},
            {"hidden": [256, 256, 256], "activation": "relu", "dropout": 0.0, "weight_decay": 1e-6, "lr": 2e-3, "batch_size": 256},
            {"hidden": [512, 512, 256, 128], "activation": "silu", "dropout": 0.05, "weight_decay": 1e-5, "lr": 2e-3, "batch_size": 256},
            {"hidden": [512, 512, 512, 256], "activation": "gelu", "dropout": 0.0, "weight_decay": 1e-6, "lr": 1e-3, "batch_size": 128},
            {"hidden": [1024, 512, 256], "activation": "silu", "dropout": 0.0, "weight_decay": 1e-6, "lr": 3e-3, "batch_size": 512},
        ],
        "max_epochs": 3000,
        "patience": 150,
        "inner_val_frac": 0.10,
    },
}

# Smaller budgets for a fast smoke test (python train.py --quick).
QUICK_OVERRIDES = {
    "poly_elasticnet": {"degree": [2], "l1_ratio": [0.5], "n_alphas": 10},
    "random_forest": {"grid": {"num_parallel_tree": [50], "max_depth": [12], "colsample_bynode": [0.6],
                               "min_child_weight": [1], "subsample": [0.632]},
                      "sklearn_grid": {"n_estimators": [100], "max_depth": [None],
                                       "min_samples_leaf": [1], "max_features": [0.6]}},
    "xgboost": {"n_iter": 2, "param_distributions": {**SEARCH_SPACES["xgboost"]["param_distributions"],
                                                     "n_estimators": [200]}},
    "mlp": {"configs": SEARCH_SPACES["mlp"]["configs"][:1], "max_epochs": 30, "patience": 10},
}

RF_BACKEND = "xgboost_gpu"          # or "sklearn" (CPU, OOB-tuned)

# Dataset-size learning curve (spec: smallest dataset that keeps accuracy).
SIZE_FRACTIONS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
