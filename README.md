# ML-Driven PVT-Variation-Aware Leakage Estimation

Digital VLSI Design course project, team **StaticSilicon** (Chanda Akshay Kumar,
G Hemanth, Jai Srikar M, Arun Venkat).

The project estimates the leakage current of CMOS circuits under process,
voltage and temperature (PVT) variation using machine learning instead of full
SPICE simulation. It runs in three parts:

- **Part A:** train per-transistor models.
- **Part B:** compose them into logic gates.
- **Part C:** compose the gates into a 3-bit carry-lookahead adder.

| Assignment | Value |
|---|---|
| Model file | 45nm_HP (PTM, BSIM4) |
| Nominal VDD | 0.9 V (sampled ±10 %) |
| Channel length / widths | L = 45 nm, W = 1–8 × L |
| Test circuit | 3-bit carry-lookahead adder |
| Simulator | ngspice |

## Status

| Part | State |
|---|---|
| A: transistor models | ✅ Done. The MLP was chosen for all 4 cases, with a test median error of 1.1–2.9 % (mean over Id/Ig/Is/Ib). See [`model_training/train_4.md`](model_training/train_4.md) |
| B: gate-level validation | ⏳ To do |
| C: 3-bit CLA validation | ⏳ To do |

## Folder structure

```
.
├── README.md                     this file
├── data_set_generattion/         Part A step 1: SPICE dataset generation (folder name as created)
│   ├── generate_data.py          draws 10,000 random samples per case (seed 42), runs ngspice, writes the CSVs
│   ├── check_data.py             Deliverable 3: KCL check + the three trend plots
│   ├── 45nm_HP.pm                the assigned PTM model card
│   ├── {nmos,pmos}_{on,off}.net  example single-transistor netlists (Deliverable 2)
│   ├── {NMOS,PMOS}_{ON,OFF}.csv  the four datasets (Deliverable 1)
│   └── plot*.png                 trend plots: OFF current vs T, OFF current vs vth0, Ig vs toxe
│
└── model_training/               Part A steps 2 to 5: model training and evaluation
    ├── train_4.md                full write-up: models, hyperparameters, all metrics, why MLP
    ├── requirements.txt          exact package versions used
    ├── setup_venv_ada.sh         builds the venv on an Ada compute node (GPU builds + NCCL fix)
    ├── run_all.sh                Slurm job: train → raw-feature ablation → evaluate → size curve
    ├── submit_by_model.sh        same pipeline as a chain of jobs, one per model family
    │
    ├── train.py                  step 1: tune and fit each (case × family) on TRAIN, score on VAL
    ├── evaluate.py               step 2: comparison table, pick best per case, evaluate ONCE on TEST, plots
    ├── data_size_curve.py        step 3: validation error vs training-set size for the chosen models
    │
    ├── leakage_ml/               the library the scripts share
    │   ├── config.py             single source of truth: paths, seeds, split, search spaces, selection rule
    │   ├── data.py               load CSV, physics features, drop constant columns, 70/15/15 split
    │   ├── transforms.py         signed-log target transform sign(I)·log10(1+|I|/I0)
    │   ├── models.py             the 4 families: Poly Elastic Net, Random Forest, XGBoost, MLP
    │   ├── metrics.py            R², median / p95 / mean relative error, sign accuracy
    │   ├── plots.py              parity, error histogram, importance, learning-curve plots
    │   └── predictor.py          LeakagePredictor: saved-model API used by Parts B and C
    │
    ├── saved_models/
    │   ├── {NMOS,PMOS}_{ON,OFF}.joblib   the 4 final models (Deliverable 5)
    │   └── candidates/                   all 24 trained candidates (git-ignored, ~590 MB, kept on Ada)
    ├── results/
    │   ├── model_comparison_val.md/.csv          Deliverable 4: 4 models × 4 cases on VAL
    │   ├── model_comparison_val_per_target.csv   the same, per terminal current
    │   ├── chosen_models_test.md/.csv            Deliverable 5: chosen models on TEST
    │   ├── feature_ablation_val.md               physics features vs raw inputs
    │   ├── data_size_curve.csv                   error vs training-set size
    │   ├── <case>_permutation_importance.csv     feature importance of the chosen model
    │   ├── candidates/*.json                     per-candidate metrics + full hyperparameter search trace
    │   ├── splits/*.json                         saved train/val/test indices (same rows for every model)
    │   └── plots/                                parity, error histograms, importance, MLP learning curves, size curve
    └── logs/                     training logs (git-ignored)
```

## How to run

**1. Generate the data** (local machine; needs `ngspice` on `PATH`):

```bash
cd data_set_generattion
python generate_data.py      # writes the 4 CSVs
python check_data.py         # KCL check + trend plots
```

**2. Train and evaluate** (GPU; we used the Ada HPC cluster with an RTX 2080 Ti):

```bash
cd model_training
bash setup_venv_ada.sh       # once, on a compute node
sbatch run_all.sh            # all 4 families × 4 cases, ablation, evaluation, size curve
# or one Slurm job per family:  bash submit_by_model.sh
```

The steps can also be run one by one: `python train.py`, then
`python evaluate.py`, then `python data_size_curve.py`.
`python train.py --quick` is a fast smoke test.

**3. Use a trained model** (Parts B and C):

```python
from leakage_ml.predictor import LeakagePredictor
m = LeakagePredictor.load("saved_models/NMOS_OFF.joblib")
I = m.predict(df)   # df columns: vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj → (n, 4) amperes [id, ig, is, ib]
```

Loading a model on a CPU-only machine needs plain `torch` and `xgboost`; the
CUDA builds are not required.

## Method in brief

- **Split:** 70 / 15 / 15 (7,000 / 1,500 / 1,500) with seed 42.
  - Hyperparameters are tuned on TRAIN only: 5-fold CV, or an inner 10 % hold-out for the MLP.
  - VAL is used only to pick the model family.
  - TEST is used once.
- **Target:** signed log `sign(I)·log10(1+|I|/I0)`. The currents span 12 decades and Id/Is change sign.
- **Features:** raw PVT inputs plus physics-derived ones (Vgs, Vds, current direction, (Vgs−Vth)/V_T, |Vds|/V_T, ln W, 1/toxe, …).
  - These cut the MLP median error by 1.2–11× compared with raw inputs.
- **Models:** Polynomial Elastic Net (scikit-learn), Random Forest (XGBoost RF mode, GPU), XGBoost (GPU), MLP (PyTorch, GPU).
- **Choice:** the MLP had the lowest validation error in every case. The test results (median error averaged over the 4 currents) are:

| Case | Median error | p95 error |
|---|---|---|
| NMOS_ON | 1.13 % | 7.3 % |
| NMOS_OFF | 2.81 % | 13.7 % |
| PMOS_ON | 2.60 % | 12.8 % |
| PMOS_OFF | 2.94 % | 15.4 % |

The full details are in [`model_training/train_4.md`](model_training/train_4.md).
