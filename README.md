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
| A: transistor models | ✅ Done. The MLP was chosen for all 4 cases, with a test median error of 0.40–0.68 % (mean over Id/Ig/Is/Ib) and R² ≥ 0.997. See [`model_training/train_4.md`](model_training/train_4.md) |
| B: gate-level validation | ✅ Done. 5 gates × all inputs × 200 corners (3,600 ngspice runs): ML vs SPICE median error 2.75 %, p95 15.6 %, mean accuracy 95.3 %. See [`part_b_gate_level/README.md`](part_b_gate_level/README.md) |
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
├── part_b_gate_level/            Part B: 5 gates × all inputs × 200 corners, SPICE vs ML (see its README)
│   ├── spice/*.cir               ngspice gate netlists: draw corners, sweep inputs, write leakage
│   ├── spice/gates.lib           the gates as subcircuits (used by the CLA)
│   ├── spice/models/             model card with vth0/toxe/ndep/u0/xj as .params
│   ├── step1_collect_spice.py    check + collect the ngspice output
│   ├── step2_ml_gates.py         ML gate leakage (gate model -> transistor models)
│   ├── step3_evaluate_gates.py   gate results table + parity plot
│   └── gatelib/                  gate definitions, GateModel, output parsing/checks, metrics
│
└── model_training/               Part A steps 2 to 5: model training and evaluation
    ├── train_4.md                full write-up: models, MLP architecture and training method, all metrics,
    │                             why MLP, error analysis and the v1 → v2 MLP improvement
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
    ├── saved_models/
    │   ├── {NMOS,PMOS}_{ON,OFF}.joblib   the 4 final models: 5-network MLP ensembles (Deliverable 5)
    │   └── candidates/                   all trained candidates (git-ignored, ~600 MB, kept on Ada)
    ├── results/
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

**4. Gate-level validation (Part B)** (local machine; needs `ngspice`).
The SPICE side is ngspice netlists; the runner runs them and then the Python ML/analysis steps:

```bash
cd part_b_gate_level
python run_part_b.py               # spice/*.cir, then steps 1-3
python run_part_b.py --skip-spice  # after new Part A models: redo only the ML steps
```

For a quick check set `let nrun = 5` in the five `.cir` files first.

## Method in brief

- **Split:** 70 / 15 / 15 (7,000 / 1,500 / 1,500) with seed 42.
  - Hyperparameters are tuned on TRAIN only: 5-fold CV, or an inner 10 % hold-out for the MLP.
  - VAL is used only to pick the model family.
  - TEST is used only to report the final metrics.
- **Target:** signed log `sign(I)·log10(1+|I|/I0)`. The currents span 12 decades and Id/Is change sign.
- **Features:** raw PVT inputs plus physics-derived ones (Vgs, Vds, current direction, (Vgs−Vth)/V_T, |Vds|/V_T, ln W, 1/toxe, …).
  - These cut the MLP median error by 2.2–3.4× compared with raw inputs.
- **Models:** Polynomial Elastic Net (scikit-learn), Random Forest (XGBoost RF mode, GPU), XGBoost (GPU), MLP (PyTorch, GPU).
- **Final MLP:** a 5-network ensemble of 256-256-256 SiLU networks.
  - Each network predicts A = (Id−Is)/2, Ig and Ib. Id and Is are rebuilt exactly by KCL, so the predicted currents always satisfy Id+Ig+Is+Ib = 0.
  - Training: AdamW with Huber loss and early stopping on the median relative error.
  - Compared with the first version (one network predicting Id/Ig/Is/Ib directly with MSE loss), this cut the test median error of NMOS_OFF, PMOS_ON and PMOS_OFF by 4–7×.
- **Choice:** the MLP had the lowest validation error in every case. The test results (averaged over the 4 currents) are:

| Case | Median error | p95 error | R² (log) |
|---|---|---|---|
| NMOS_ON | 0.49 % | 4.2 % | 1.000 |
| NMOS_OFF | 0.54 % | 5.7 % | 0.999 |
| PMOS_ON | 0.68 % | 4.9 % | 1.000 |
| PMOS_OFF | 0.40 % | 4.0 % | 0.999 |

The full details are in [`model_training/train_4.md`](model_training/train_4.md).

## Part B results in brief

- **Setup:**
  - INV, NAND2, NOR2, AND2 and OR2 at every input combination and 200 random PVT corners: 18 cases × 200 = 3,600 runs.
  - SPICE: transistor-level ngspice netlists. Their `.control` scripts draw the corners (seed 142), apply them with `alterparam`/`reset`, `option temp` and `alter`, and sweep the inputs.
  - ML: the four Part A models, fed each transistor's terminal voltages from the recorded SPICE node voltages (spec Step 4).
- **Leakage** = supply power / VDD:
  - SPICE: −I(VDD) plus the current of each input held at VDD;
  - ML: (1/VDD)·Σ(Vds·Id + Vgs·Ig + Vbs·Ib) over the transistors.
- **Accuracy:**

  | | Median error | p95 error | Mean accuracy |
  |---|---|---|---|
  | All runs | 2.75 % | 15.6 % | 95.3 % |
  | Best gate (OR2) | | | 97.0 % |
  | Worst gate (NAND2) | | | 93.0 % |

  - The main outlier is NAND2 AB=00 (two OFF NMOS in series), at 83.4 % mean accuracy.
- **SPICE consistency:** the Part B way of setting a corner reproduces the Part A dataset currents to within 0.0005 %, so the remaining error is the ML models' own.
- **The models are less accurate inside gates than on their own test set:** 0.4–0.7 % median error on the uniform test set, 2.75 % inside gates. Gates use the transistors at rail and stack-node voltages, which uniform sampling rarely covers; more training data there is the planned improvement.

The full table (Deliverable 6), parity plot (Deliverable 7) and file-by-file description are in [`part_b_gate_level/README.md`](part_b_gate_level/README.md).
