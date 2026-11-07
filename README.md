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
| A: transistor models | ✅ Done. The final MLP (v3) has a test **mean** error of 0.13–0.29 % (mean over Id/Ig/Is/Ib; below 1 % for every current) and R² = 1.000. See [`model_training/train_4.md`](model_training/train_4.md) |
| B: gate-level validation | ⏳ To do |
| C: 3-bit CLA validation | ⏳ To do |

## Folder structure

```
.
├── README.md                     this file
├── data_set_generattion/         Part A step 1: SPICE dataset generation (folder name as created)
│   ├── generate_data.py          draws the random samples (seed 42; extra rows seed 4242), runs ngspice, writes the CSVs
│   ├── check_data.py             Deliverable 3: KCL check + the three trend plots
│   ├── 45nm_HP.pm                the assigned PTM model card
│   ├── {CASE}_new.csv            current datasets, 10,000 rows each (Deliverable 1): junction geometry
│   │                             AD=AS=W·Ldiff, PD=PS=2(W+Ldiff), Ldiff = 2·Lmin = 90 nm
│   ├── {CASE}_new_extra.csv      30,000 extra training rows per case (same distributions)
│   ├── {case}_new.net            example single-transistor netlists (Deliverable 2), body node as ground
│   ├── plot*_new.png             trend plots for the current data: OFF current vs T and vth0, Ig vs toxe
│   └── {CASE}.csv, {case}.net,   first datasets (no junction geometry; used by MLP v1/v2) and their
│       plot*.png                 netlists and plots
│
└── model_training/               Part A steps 2 to 5: model training and evaluation
    ├── train_4.md                full write-up: models, MLP architecture and training method, all metrics,
    │                             why MLP, error analysis, and the v1 → v2 → v3 MLP improvements
    ├── v3/                       FINAL v3 MLP (new data): saved_models/<CASE>.joblib, results/ (test,
    │                             validation, recipe study, size curve, plots)
    ├── requirements.txt          exact package versions used
    ├── setup_venv_ada.sh         builds the venv on an Ada compute node (GPU builds + NCCL fix)
    ├── run_all.sh                Slurm job: train → raw-feature ablation → evaluate → size curve
    ├── submit_by_model.sh        same pipeline as a chain of jobs, one per model family
    ├── run_experiments.sh        Slurm job for the MLP recipe study (4 cases in parallel)
    │
    ├── train.py                  step 1: tune and fit each (case × family) on TRAIN, score on VAL
    ├── evaluate.py               step 2: comparison table, pick best per case, evaluate on TEST, plots
    ├── data_size_curve.py        step 3: validation error vs training-set size for the chosen models
    ├── analyze_errors.py         where the error comes from: per current and operating region (TRAIN/VAL only)
    ├── mlp_experiments.py        MLP recipe study: target form × loss × early-stopping monitor
    │
    ├── leakage_ml/               the library the scripts share
    │   ├── config.py             single source of truth: paths, seeds, split, search spaces, MLP recipe, selection rule
    │   ├── data.py               load CSV, physics features, drop constant columns, 70/15/15 split
    │   ├── transforms.py         signed-log transform; KCLTransform (predict A, Ig, Ib and rebuild Id, Is)
    │   ├── models.py             the 4 families: Poly Elastic Net, Random Forest, XGBoost, MLP (+ ensemble)
    │   ├── metrics.py            R², median / p95 / mean relative error, sign accuracy
    │   ├── plots.py              parity, error histogram, importance, learning-curve plots
    │   └── predictor.py          LeakagePredictor: saved-model API used by Parts B and C
    │
    ├── saved_models/             v2 models (first data); v3 models are in v3/saved_models/
    │   ├── {NMOS,PMOS}_{ON,OFF}.joblib   5-network MLP ensembles
    │   └── candidates/                   all trained candidates (git-ignored, ~600 MB, kept on Ada)
    ├── results/                  all-families comparison and v1/v2 results (first data)
    │   ├── model_comparison_val.md/.csv          Deliverable 4: 4 models × 4 cases on VAL
    │   ├── model_comparison_val_per_target.csv   the same, per terminal current
    │   ├── chosen_models_test.md/.csv            Deliverable 5: chosen models on TEST
    │   ├── feature_ablation_val.md               physics features vs raw inputs
    │   ├── data_size_curve.csv                   error vs training-set size
    │   ├── error_analysis.md                     error breakdown of the final models
    │   ├── <case>_permutation_importance.csv     feature importance of the chosen model
    │   ├── candidates/*.json                     per-candidate metrics + full hyperparameter search trace
    │   ├── experiments/mlp_<case>.md/.json       MLP recipe study results
    │   ├── v1/                                   results of the first MLP version, kept for comparison
    │   ├── splits/*.json                         saved train/val/test indices (same rows for every model)
    │   └── plots/                                parity, error histograms, importance, MLP learning curves, size curve
    └── logs/                     training logs (git-ignored)
```

## How to run

**1. Generate the data** (local machine; needs `ngspice` on `PATH`):

```bash
cd data_set_generattion
python generate_data.py --extra 30000   # <CASE>_new.csv (10,000 rows) + <CASE>_new_extra.csv (30,000 rows)
python check_data.py                    # KCL check + trend plots
```

On Windows, `generate_data.py` uses the console build `ngspice_con` (the plain
`ngspice.exe` is a GUI build that prints nothing).

**2. Train and evaluate** (GPU; we used the Ada HPC cluster with an RTX 2080 Ti):

```bash
cd model_training
bash setup_venv_ada.sh                            # once, on a compute node
sbatch run_experiments.sh                         # v3 MLP recipe study
MODELS=mlp STEPS="train eval size" sbatch run_all.sh    # final v3 MLPs -> v3/
LEAKAGE_DATASET=orig sbatch run_all.sh            # all 4 families on the first data (v2 comparison)
```

`LEAKAGE_DATASET` selects the data: `new` (the default) or `orig`. The steps
can also be run one by one: `python train.py`, then `python evaluate.py`, then
`python data_size_curve.py`. `python train.py --quick` is a fast smoke test.

**3. Use a trained model** (Parts B and C):

```python
from leakage_ml.predictor import LeakagePredictor
m = LeakagePredictor.load("v3/saved_models/NMOS_OFF.joblib")
I = m.predict(df)   # df columns: vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj → (n, 4) amperes [id, ig, is, ib]
```

Loading a model on a CPU-only machine needs plain `torch` and `xgboost`; the
CUDA builds are not required.

## Method in brief

- **Data:** 10,000 SPICE samples per case (spec §4 distributions), with drain/source junction geometry (Ldiff = 90 nm), plus 30,000 extra training rows per case.
- **Split:** 70 / 15 / 15 (7,000 / 1,500 / 1,500) of the 10,000 with seed 42; the extra rows go to TRAIN only.
  - Hyperparameters are tuned on TRAIN only: 5-fold CV, or an inner 10 % hold-out for the MLP.
  - VAL is used only to pick the model family.
  - TEST is used only to report the final metrics.
- **Target:** signed log `sign(I)·log10(1+|I|/I0)`. The currents span 12 decades and Id/Is change sign.
- **Features:** raw PVT inputs plus physics-derived ones (Vgs, Vds, current direction, (Vgs−Vth)/V_T, |Vds|/V_T, ln W, 1/toxe, …).
  - These cut the MLP median error by 2.2–3.4× compared with raw inputs.
- **Models:** Polynomial Elastic Net (scikit-learn), Random Forest (XGBoost RF mode, GPU), XGBoost (GPU), MLP (PyTorch, GPU).
- **Choice:** the MLP had the lowest validation error in every case (on the first data, against all four families).
- **Final MLP (v3):** a 5-network ensemble per case of SiLU networks (256-256-256; 512-512-256-128 for NMOS_ON).
  - Each network predicts A = (Id−Is)/2, Ig and Ib. Id and Is are rebuilt exactly by KCL, so the predicted currents always satisfy Id+Ig+Is+Ib = 0.
  - Training: AdamW with Huber loss, early stopping on the mean relative error, and a κ-weighted loss in the OFF cases.

The test results of v3, averaged over the 4 currents, with v2 for comparison:

| Case | Median error | **Mean error** | p95 error | R² (log) |
|---|---|---|---|---|
| NMOS_ON | 0.093 % (v2 0.49) | **0.29 %** (v2 1.47) | 0.68 % | 1.000 |
| NMOS_OFF | 0.067 % (v2 0.54) | **0.19 %** (v2 2.97) | 0.35 % | 1.000 |
| PMOS_ON | 0.139 % (v2 0.68) | **0.24 %** (v2 1.89) | 0.57 % | 1.000 |
| PMOS_OFF | 0.058 % (v2 0.40) | **0.13 %** (v2 2.20) | 0.38 % | 1.000 |

The full details are in [`model_training/train_4.md`](model_training/train_4.md).
