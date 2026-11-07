# Part A: Training the four transistor models

Team StaticSilicon. Model file 45nm_HP, VDD = 0.9 V, L = 45 nm.

This document records how the four model families were trained for the four
transistor cases (NMOS ON/OFF, PMOS ON/OFF), the architecture and
hyperparameters of each, every metric, why the MLP was chosen, and how the MLP
was then improved. All numbers come from `results/` (the JSON files in
`results/candidates/` hold the full search traces).

**Short version.**
- The MLP is the best family in all four cases on the validation set.
- The final MLP (**v3**, §13) is a 5-network ensemble per case. Each network predicts the channel-like current A = (Id−Is)/2 together with Ig and Ib, and Id and Is are rebuilt exactly with KCL.
- v3 is trained on the new dataset: transistors with drain/source junction geometry (AD/AS/PD/PS), and 37,000 training rows per case.
- On the test set, its **mean** error is 0.06–0.40 % per terminal current and 0.13–0.29 % averaged over the four currents. The median error is 0.06–0.14 %.
- R² is 1.000 (log and raw) and the current's sign is right in 100 % of samples.
- This meets the target of R² > 0.99 and mean error < 1 % (TA's annotated slides, page 4) in every case and current.

**Three versions of the MLP** (test set, average over the four currents):

| Version | Data | What it is | Median error, NMOS_ON / NMOS_OFF / PMOS_ON / PMOS_OFF | Mean error, same order |
|---|---|---|---|---|
| v1 | original CSVs, 7,000 training rows | One network per case, predicting Id, Ig, Is and Ib directly | 1.13 / 2.81 / 2.60 / 2.94 % | 5.27 / 5.11 / 4.45 / 5.68 % |
| v2 | original CSVs, 7,000 training rows | KCL outputs, Huber loss, early stopping on the median error, 5-network ensemble | 0.49 / 0.54 / 0.68 / 0.40 % | 1.47 / 2.97 / 1.89 / 2.20 % |
| **v3 (final)** | new CSVs (junction geometry), 37,000 training rows | v2 + early stopping on the **mean** error, κ-weighted loss in the OFF cases | **0.093 / 0.067 / 0.139 / 0.058 %** | **0.29 / 0.19 / 0.24 / 0.13 %** |

- §6 explains the error analysis that led to v2; §13 covers v3.
- Sections 1–12 describe the model families and the v2 MLP in detail. v3 uses the same features, model code and evaluation.
- v1 results are kept in `results/v1/`, v2 results in `results/`, and v3 results in `v3/results/`.

---

## 1. Setup

| Item | Value |
|---|---|
| Data | 4 CSVs (one per case), 10,000 ngspice samples each (`data_set_generattion/`) |
| Split | 70 / 15 / 15 → 7,000 train / 1,500 val / 1,500 test, `train_test_split`, seed 42. Indices saved in `results/splits/<case>_split.json` so every model uses the same rows |
| Targets | The four terminal currents Id, Ig, Is, Ib |
| Hardware | Ada HPC, one RTX 2080 Ti, 10 CPUs, 20 GB RAM |
| Software | Python 3.12.4, PyTorch 2.14.1 (CUDA 12.6), XGBoost 3.4.1 (`xgboost-cu12`), scikit-learn 1.9.1 |
| Seeds | `SEED_SPLIT=42`, `SEED_CV=43`, `SEED_MODEL=44`, `SEED_TORCH=45` (separate streams, `leakage_ml/config.py`) |

### Rule for using the three splits

- **TRAIN (7,000):** used for everything that is fitted or tuned: scalers, the target transforms, dropped-column decisions, and all hyperparameter and recipe search (5-fold CV, or an inner 10 % hold-out for the MLP).
- **VAL (1,500):** used only to compare the model families and pick one per case.
- **TEST (1,500):** used in `evaluate.py` only, to report the chosen models. It was never used for training, tuning or choosing anything (see §11).

## 2. Preprocessing (same for every family)

### 2.1 Target transform: signed log

The currents span about 12 decades (1e-17 to 5e-4 A), and Id/Is change sign
(about 50 % negative in the ON cases, 13–21 % in the OFF cases). So the models
work with

    y = sign(I) · log10(1 + |I| / I0)

with one I0 per current, set on TRAIN as 0.1 × the 1st percentile of |I|.

- For |I| ≫ I0 this is log10(|I|/I0). An error dy in y is then a relative current error of about ln(10)·dy, so fitting y approximately minimises relative error, which is what the spec scores.
- Unlike plain log|I|, it keeps the sign and is smooth through I = 0.
- The final MLP applies the same transform to A, Ig and Ib instead of to Id, Ig, Is and Ib (§3.4).

### 2.2 Features

**Raw inputs:** vdd, temp, W/L, vd, vs, vth0, toxe, ndep, u0, xj. vg and vb are
dropped because they are fixed by the case.

**Physics features** (`leakage_ml/data.py`) are added on top of the raw inputs:

| Group | Features | Why |
|---|---|---|
| Terminal differences | vgs, vds, vbs, vgd, vbd | A MOSFET responds to voltage differences, not absolute node voltages |
| Polarity-aware | \|vds\|, dir = sign(p·vds), vgs_eff, vbs_eff (measured from the lower-potential terminal for NMOS, higher for PMOS) | Handles the source/drain swap when the current reverses |
| Thermal / subthreshold | 1/V_T, ln T, (vgs_eff − vth0)/V_T, \|vds\|/V_T, ln(1 − e^(−\|vds\|/V_T)) | Subthreshold current ∝ exp((Vgs−Vth)/(nV_T))·(1 − exp(−Vds/V_T)) |
| Log / inverse | ln W, ln u0, 1/toxe | Current ∝ W·u0, and gate tunnelling ∝ exp(−B·toxe) |

On TRAIN, columns that are constant or perfectly correlated with a kept column
are then removed:

| Case | Inputs | Dropped |
|---|---|---|
| NMOS_ON | 25 | vg, vb, vbd, vbs |
| NMOS_OFF | 22 | vg, vb, vgd, vbd, vgs, vbs, vbs_eff |
| PMOS_ON | 25 | vg, vb, vgd, vgs |
| PMOS_OFF | 24 | vg, vb, vbs, vbd, vbs_eff |

### 2.3 How accuracy is measured (`leakage_ml/metrics.py`)

Predictions are converted back to amperes. Then, for each current:

| Metric | Definition |
|---|---|
| Relative error | \|I_pred − I_SPICE\| / \|I_SPICE\| × 100 %, per sample |
| Median error | Median of the relative error. **Accuracy = 100 − median error** |
| p95 error | 95th percentile of the relative error (the tail) |
| Mean error | Mean of the relative error. Very sensitive to samples where I_SPICE ≈ 0 |
| R² (log) | R² in signed-log space: how well the shape is captured across all 12 decades |
| R² (raw) | R² in amperes. Dominated by the few largest currents |
| Sign accuracy | % of samples where sign(I_pred) = sign(I_SPICE) |

**Trivial baseline:** always predict the training mean of y. It scores R² ≈ 0
and a median error of 97–117 %.

---

## 3. The four model families

### 3.1 Polynomial Elastic Net (scikit-learn, CPU)

**Pipeline**, one model per current:

`StandardScaler → PolynomialFeatures(degree) → StandardScaler → ElasticNetCV`

**Search, on TRAIN only:**
- Degree ∈ {2, 3} was compared by 5-fold CV MSE.
- Inside `ElasticNetCV`: l1_ratio ∈ {0.1, 0.5, 0.9, 1.0}, a 40-value alpha path, 5-fold CV, max_iter 20,000.

**Chosen:** degree 3 for every case and current.

| Case | Polynomial terms | Current | alpha | l1_ratio | Non-zero coefficients |
|---|---|---|---|---|---|
| NMOS_ON | 3275 | Id / Ig / Is / Ib | 7.7e-3 / 1.1e-3 / 7.7e-3 / 6.8e-4 | 0.9 / 1.0 / 0.9 / 1.0 | 57 / 194 / 56 / 344 |
| NMOS_OFF | 2299 | Id / Ig / Is / Ib | 2.2e-3 / 1.2e-3 / 2.3e-3 / 1.0e-3 | 1.0 / 0.5 / 1.0 / 1.0 | 405 / 118 / 402 / 215 |
| PMOS_ON | 3275 | Id / Ig / Is / Ib | 8.4e-3 / 1.0e-3 / 1.0e-2 / 6.0e-4 | 0.9 / 1.0 / 0.9 / 1.0 | 77 / 218 / 68 / 352 |
| PMOS_OFF | 2924 | Id / Ig / Is / Ib | 3.5e-3 / 1.3e-3 / 2.9e-3 / 8.9e-4 | 1.0 / 1.0 / 1.0 / 1.0 | 323 / 76 / 366 / 238 |

- **Role:** the interpretable linear-in-parameters reference.
- **Fit time:** 13–15 min per case.

### 3.2 Random Forest (XGBoost random-forest mode, GPU)

**Setup:**
- One boosting round of `num_parallel_tree` bagged trees, learning rate 1 (no shrinkage), λ = 1e-5. This is XGBoost's documented RF recipe.
- `tree_method="hist"`, `device="cuda"`.
- One model per current.

**Search:** `GridSearchCV`, 5-fold on TRAIN.
- Grid: colsample_bynode ∈ {0.33, 0.6, 1.0}.
- Fixed: 300 trees, max_depth 16, min_child_weight 1, subsample 0.632.
- Depth 16 and min_child_weight 1 were fixed because an earlier 12-point grid (depth {12, 16} × min_child_weight {1, 4} × colsample) chose them for Id in all four cases. That grid took about 6 h per case, so the final run searches only the feature fraction.

**Chosen:** 300 trees, depth 16, colsample_bynode **0.6** everywhere, except PMOS_OFF Ib, which chose 0.33.

**Fit time:** about 2.7 h per case.

### 3.3 XGBoost (gradient-boosted trees, GPU)

**Setup:** `objective="reg:squarederror"`, `tree_method="hist"`,
`device="cuda"`. One model per current.

**Search:** 25 random configs (`ParameterSampler`, seed 43). Each config is
scored with `xgb.cv`: 5-fold on TRAIN, with early stopping after 100 rounds
without improvement. The final model is then refit on all of TRAIN with the
CV-chosen number of trees.

| Hyperparameter | Range searched |
|---|---|
| n_estimators (maximum) | 800, 1500, 3000 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |
| max_depth | 4, 6, 8, 10 |
| min_child_weight | 1, 3, 5 |
| subsample | 0.7, 0.85, 1.0 |
| colsample_bytree | 0.6, 0.8, 1.0 |
| reg_lambda | 0.1, 1, 10 |
| reg_alpha | 0, 1e-3, 0.1 |

**Chosen:** in the table, "trees" is the number after early stopping, and
"η" is the learning rate.

| Case | Id / Is | Ig | Ib |
|---|---|---|---|
| NMOS_ON | 800 trees, η 0.03, depth 10, mcw 3, sub 0.85, col 1.0, λ 10, α 0.1 | 1500, η 0.05, depth 4, mcw 3, sub 0.7, col 0.6, λ 1, α 1e-3 | same as Ig |
| NMOS_OFF | 3000, η 0.01, depth 8, mcw 5, sub 0.7, col 1.0, λ 1, α 0.1 | 1500, η 0.05, depth 4, mcw 3, sub 0.7, col 0.6 | 1500, η 0.01, depth 4, mcw 3, sub 0.7, col 1.0, λ 0.1, α 0 |
| PMOS_ON | 3000, η 0.01, depth 8, mcw 5, sub 0.7, col 1.0, λ 1, α 0.1 | 1500, η 0.05, depth 4, mcw 3, sub 0.7, col 0.6 | same as Ig |
| PMOS_OFF | 2337 (Id) / 3000 (Is), η 0.01, depth 8, mcw 5, sub 0.7, col 1.0 | 1500, η 0.05, depth 4, mcw 3, sub 0.7, col 0.6 | 1059, η 0.1, depth 4, mcw 3, sub 0.7, col 0.8, λ 0.1 |

- **Pattern:** Id and Is need deep, slow-learning ensembles (depth 8–10, η 0.01–0.03). Ig and Ib are smoother and need only depth 4.
- **Fit time:** 3.3–3.6 h per case (25 configs × 5 folds × 4 currents).

### 3.4 MLP (PyTorch, GPU), the chosen model

#### 3.4.1 What the network predicts: KCL outputs

Instead of predicting Id, Ig, Is and Ib directly, each network predicts three
quantities:

| Output | Meaning | How its sign is handled |
|---|---|---|
| \|A\|, where A = (Id − Is)/2 | The "through" current, mostly channel current | sign(A) = sign(Vd − Vs) is imposed; the network learns only the magnitude |
| Ig | Gate current | Kept in signed-log form; its sign never changes within a case |
| Ib | Body current | Kept in signed-log form; its sign never changes within a case |

Each output is in signed-log form (§2.1). Id and Is are then rebuilt exactly
using Kirchhoff's current law:

    S  = (Id + Is)/2 = −(Ig + Ib)/2      (KCL: Id + Ig + Is + Ib = 0)
    Id =  A + S
    Is = −A + S

**Why this works better** (details in §6):
- In the OFF state, Id and Is are near-cancellations of A and S. A and S are each smooth with a fixed sign pattern, but their sum is hard to learn directly in log space.
- The sign of A follows sign(Vd−Vs) in 98.8–99.99 % of samples. Imposing it removes the sign flip at Vds = 0 from what the network must learn.
- The four predicted currents always satisfy KCL exactly (to about 1e-16), which keeps them physically consistent for Parts B and C.
- Rebuilding Id/Is from the *exact* A, Ig and Ib loses nothing: the median error is about 0.0003 %.

Code: `KCLTransform` in `leakage_ml/transforms.py`; the sign handling is in
`TorchMLPRegressor` in `leakage_ml/models.py`.

#### 3.4.2 Final architecture

The same architecture won in all four cases. One ensemble member:

```
22–25 physics features (standardised, StandardScaler fit on TRAIN)
  → Linear(n_in → 256) → SiLU
  → Linear(256 → 256)  → SiLU
  → Linear(256 → 256)  → SiLU
  → Linear(256 → 3)                       outputs: z_|A|, z_Ig, z_Ib (standardised signed-log)
  → undo standardisation; give A the sign of (Vd − Vs)
```

The final model averages the 5 members' outputs (in signed-log space) and then
rebuilds the four currents:

```
average of 5 members
  → inverse signed-log → A, Ig, Ib in amperes
  → S = −(Ig + Ib)/2;  Id = A + S;  Is = −A + S
  → [Id, Ig, Is, Ib]
```

| Case | Inputs | Parameters per network | Networks | Total parameters |
|---|---|---|---|---|
| NMOS_ON | 25 | 139,011 | 5 | 695,055 |
| NMOS_OFF | 22 | 138,243 | 5 | 691,215 |
| PMOS_ON | 25 | 139,011 | 5 | 695,055 |
| PMOS_OFF | 24 | 138,755 | 5 | 693,775 |

Per network the count is (n_in·256 + 256) + 2·(256·256 + 256) + (256·3 + 3).

#### 3.4.3 Training method (how the weights are updated)

| Item | What is used |
|---|---|
| Learning type | Supervised regression, trained by mini-batch gradient descent with backpropagation (PyTorch autograd, `loss.backward()`) |
| Weight initialisation | He (Kaiming) normal for the weights, zero biases. Every network starts from new random weights (training from scratch) |
| Input / output scaling | `StandardScaler` on the inputs and on the 3 targets, fit on that member's training portion |
| Loss | **Huber loss** with δ = 0.1, on the standardised targets: ½e² when \|e\| ≤ δ, and δ(\|e\| − ½δ) otherwise, averaged over the batch and the 3 outputs |
| Optimiser (update rule) | **AdamW**: Adam with decoupled weight decay. Per step, for gradient g: m ← β1·m + (1−β1)·g, v ← β2·v + (1−β2)·g², θ ← θ − lr·(m̂ / (√v̂ + ε) + λ·θ). Settings: β1 = 0.9, β2 = 0.999, ε = 1e-8 (PyTorch defaults), weight decay λ = 1e-6 |
| Learning rate | Starts at 2e-3. **ReduceLROnPlateau** halves it each time the inner-hold-out loss has not improved for 50 epochs, down to a minimum of 1e-6 |
| Mini-batches | 256 samples. The 6,300 training rows are reshuffled every epoch (seeded `torch.randperm`), giving 25 weight updates per epoch |
| Early stopping | After each epoch, the **inner-hold-out median relative error** is computed in amperes, on the rebuilt Id, Ig, Is and Ib. Training stops after 150 epochs without improvement, and the weights from the best epoch are restored (`copy.deepcopy`). The upper limit is 3,000 epochs |
| Inner hold-out | 10 % of TRAIN (700 rows); the network trains on the other 6,300 rows. The official VAL and TEST sets are never seen |
| Regularisation | Weight decay, early stopping and the ensemble. Dropout was tried and lost (§3.4.4) |
| Ensemble | 5 members. Member k uses its own random seed (45 + k) **and** its own inner hold-out (seed 43 + k), so together the members train on all of TRAIN. Their outputs are averaged in signed-log space before the currents are rebuilt |
| Precision / hardware | float32. All training data stays on the GPU (RTX 2080 Ti) |

**Training length per member** (epochs run; the best epoch is 150 earlier):

| Case | Members 1–5 | Fit time (6-architecture search + 4 extra members) |
|---|---|---|
| NMOS_ON | 1099, 1041, 1127, 1301, 1166 | 17.0 min |
| NMOS_OFF | 999, 1235, 800, 608, 819 | 15.6 min |
| PMOS_ON | 1006, 570, 948, 933, 774 | 14.6 min |
| PMOS_OFF | 1451, 1602, 1342, 1121, 1391 | 19.2 min |

That is roughly 15,000–40,000 weight updates per member.

**Learning curves** are in `results/plots/<case>_mlp_learning_curve.png`. They
show why early stopping uses the median error rather than the loss. The
hold-out *loss* flattens early (about epoch 520 for NMOS_OFF). The hold-out
*median error* keeps falling, in steps at each learning-rate halving, from
about 2 % to 0.85 % at epoch 849. A loss-based stop would have ended training
too early.

**Prediction speed:** 0.035–0.043 ms per sample for the whole 5-network
ensemble, on CPU.

#### 3.4.4 Architecture search

Six architectures were trained per case with the final recipe (first member
only). They were scored on the inner hold-out with the spec metric: the mean
over the four currents of the median relative error, with ties broken by p95.

| # | Hidden layers | Activation | Dropout | Weight decay | lr | Batch | NMOS_ON | NMOS_OFF | PMOS_ON | PMOS_OFF |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 256-256-256 | SiLU | 0 | 1e-6 | 2e-3 | 256 | **0.84 %** | **0.81 %** | **1.04 %** | **0.67 %** |
| 2 | 512-512-256-128 | SiLU | 0 | 1e-6 | 2e-3 | 256 | 1.28 | 1.24 | 1.33 | 1.03 |
| 3 | 256-256-256 | ReLU | 0 | 1e-6 | 2e-3 | 256 | 3.66 | 3.62 | 4.58 | 2.89 |
| 4 | 512-512-256-128 | SiLU | 0.05 | 1e-5 | 2e-3 | 256 | 1.22 | 1.14 | 1.36 | 0.83 |
| 5 | 512-512-512-256 | GELU | 0 | 1e-6 | 1e-3 | 128 | 1.52 | 1.59 | 1.89 | 1.68 |
| 6 | 1024-512-256 | SiLU | 0 | 1e-6 | 3e-3 | 512 | 1.70 | 1.47 | 1.85 | 1.28 |

(Inner-hold-out median error, mean over the four currents. **Bold** = chosen.)

**What the search shows:**
- **SiLU beats ReLU by 3–4×.** The currents are smooth exponentials of voltage, and a smooth activation suits them.
- **Bigger networks were worse in every case.** With about 6,300 samples, the larger networks fit less well within the training budget. The small 256-256-256 net won all four cases.
- **Dropout did not help.** The SPICE data is noise-free, so dropout only adds noise to training.

---

## 4. Validation results: all models, all metrics

These numbers are from the validation set (1,500 samples per case). "Train
median" is the same metric on TRAIN, so the gap between it and "Median"
shows over-fitting. The MLP rows are the final v2 model.

### 4.1 Summary (Deliverable 4): mean over Id, Ig, Is, Ib

| Case | Model | R² (log) | Median error % | p95 error % | Worst p95 % | Train median % |
|---|---|---|---|---|---|---|
| NMOS_ON | Baseline (mean) | 0.000 | 97.2 | 2405 | 7044 | – |
| | Poly Elastic Net | 0.9971 | 13.80 | 44.3 | 57.1 | 13.60 |
| | Random Forest | 0.9926 | 12.98 | 63.2 | 76.9 | 4.63 |
| | XGBoost | 0.9987 | 6.27 | 29.9 | 36.5 | 2.08 |
| | **MLP** | **0.9999** | **0.51** | **3.7** | **6.5** | 0.13 |
| NMOS_OFF | Baseline (mean) | 0.000 | 116.7 | 681 | 2238 | – |
| | Poly Elastic Net | 0.9710 | 35.54 | 185.6 | 380.1 | 33.04 |
| | Random Forest | 0.9914 | 10.29 | 57.3 | 92.3 | 3.99 |
| | XGBoost | 0.9951 | 5.24 | 39.5 | 72.5 | 1.49 |
| | **MLP** | **0.9989** | **0.53** | **6.1** | **10.0** | 0.17 |
| PMOS_ON | Baseline (mean) | 0.000 | 96.8 | 2592 | 8693 | – |
| | Poly Elastic Net | 0.9965 | 13.79 | 50.8 | 68.4 | 13.31 |
| | Random Forest | 0.9938 | 12.46 | 65.9 | 82.4 | 4.93 |
| | XGBoost | 0.9987 | 5.96 | 29.0 | 38.1 | 2.03 |
| | **MLP** | **0.9997** | **0.69** | **5.4** | **10.2** | 0.31 |
| PMOS_OFF | Baseline (mean) | -0.001 | 116.2 | 761 | 1905 | – |
| | Poly Elastic Net | 0.9607 | 31.85 | 116.6 | 215.6 | 29.54 |
| | Random Forest | 0.9857 | 8.28 | 53.2 | 83.2 | 3.25 |
| | XGBoost | 0.9920 | 4.22 | 41.3 | 73.6 | 1.49 |
| | **MLP** | **0.9993** | **0.40** | **4.7** | **7.5** | 0.10 |

### 4.2 Per current (validation)

Each cell gives median % / p95 % / mean %.

| Case | Current | Poly Elastic Net | Random Forest | XGBoost | MLP |
|---|---|---|---|---|---|
| NMOS_ON | Id | 17.1 / 56.5 / 377 | 11.5 / 66.4 / 44.0 | 6.11 / 36.5 / 16.8 | **0.55 / 3.1 / 1.2** |
| | Ig | 6.97 / 24.7 / 9.5 | 13.4 / 43.1 / 16.9 | 5.31 / 17.5 / 6.8 | **0.48 / 6.5 / 1.8** |
| | Is | 17.1 / 57.1 / 123 | 11.4 / 66.2 / 27.5 | 5.69 / 35.1 / 11.4 | **0.55 / 3.1 / 1.1** |
| | Ib | 14.1 / 38.8 / 16.5 | 15.7 / 76.9 / 24.7 | 7.96 / 30.5 / 10.6 | **0.48 / 2.2 / 0.7** |
| NMOS_OFF | Id | 61.0 / 320 / 109 | 15.0 / 92.3 / 26.5 | 7.89 / 72.5 / 16.4 | **0.76 / 10.0 / 4.1** |
| | Ig | 1.67 / 6.0 / 2.4 | 9.84 / 39.2 / 14.3 | 4.93 / 16.1 / 6.2 | **0.37 / 1.9 / 0.7** |
| | Is | 66.5 / 380 / 108 | 16.1 / 87.5 / 30.3 | 7.63 / 61.5 / 17.9 | **0.69 / 8.5 / 3.8** |
| | Ib | 13.0 / 36.0 / 16.5 | **0.23** / 10.1 / 2.5 | 0.52 / 7.9 / 2.0 | 0.32 / **4.1** / **1.5** |
| PMOS_ON | Id | 18.0 / 61.9 / 25.6 | 13.2 / 81.0 / 30.5 | 5.95 / 38.1 / 12.3 | **0.78 / 4.3 / 1.8** |
| | Ig | 4.79 / 32.3 / 9.4 | 11.4 / 39.1 / 15.4 | 4.59 / 14.4 / 5.9 | **0.58 / 10.2 / 2.7** |
| | Is | 18.6 / 68.4 / 55.2 | 13.5 / 82.4 / 29.3 | 5.82 / 36.3 / 17.5 | **0.77 / 4.3 / 2.3** |
| | Ib | 13.8 / 40.7 / 16.3 | 11.8 / 61.2 / 20.3 | 7.50 / 27.0 / 10.0 | **0.63 / 2.7 / 1.0** |
| PMOS_OFF | Id | 55.1 / 216 / 104 | 10.6 / 77.3 / 21.4 | 5.43 / 73.6 / 16.0 | **0.44 / 6.5 / 2.6** |
| | Ig | 1.99 / 7.9 / 3.0 | 10.2 / 40.2 / 15.0 | 5.12 / 17.1 / 6.6 | **0.25 / 1.5 / 0.6** |
| | Is | 57.3 / 206 / 240 | 11.7 / 83.2 / 23.8 | 5.63 / 67.2 / 16.6 | **0.48 / 7.5 / 2.2** |
| | Ib | 13.0 / 37.5 / 18.8 | 0.64 / 12.0 / 3.6 | 0.69 / 7.1 / 2.1 | **0.42 / 3.3 / 2.2** |

**R² (log), MLP:** 0.9975–1.0000 on every current.

**Sign accuracy:**
- MLP: 99.7–100 %, with Ig and Ib always 100 %.
- XGBoost: 98.9–100 %.
- Random Forest: 98.8–100 %.
- Elastic Net: 98.4–100 %.

**Other observations:**
- **Elastic Net R² (raw):** negative for OFF-state Id/Is (down to −3.8e6 on PMOS_OFF Is). A few predictions are off by orders of magnitude in amperes even though R² (log) is 0.92–0.95. A degree-3 polynomial cannot follow the exponential subthreshold behaviour over 12 decades.
- **Over-fitting (train median vs val median):**
  - The tree models over-fit the most: XGBoost reaches 1.5–2 % on train but 4–6 % on val, and RF 3–5 % vs 8–13 %.
  - The MLP reaches 0.10–0.31 % on train and 0.40–0.69 % on val. A gap remains, but the absolute errors are an order of magnitude below the other families.
  - Elastic Net has no gap, but high error on both: it is under-fitting (high bias).

---

## 5. Why the MLP was chosen

**Selection rule** (fixed before looking at results, `config.py`): lowest mean
median error over the four currents on VAL; ties are broken by mean p95.

The MLP wins in all four cases, by a wide margin:

1. **Accuracy.** Its median error is 9–12× lower than XGBoost, the next-best model, and its p95 is 5–9× lower. It wins 15 of the 16 per-current comparisons on median error. The one exception is NMOS_OFF Ib, where RF reaches 0.23 % against the MLP's 0.32 %; even there, the MLP's p95 is lower.
2. **The tails matter for Parts B and C.** Gate and circuit leakage is a sum of transistor currents, mostly the OFF-transistor channel currents plus gate leakage. The MLP's worst p95 is 6.5–10 %, against 37–74 % for XGBoost and 77–92 % for RF.
3. **Smooth function.** The I–V relationship is smooth (exponential in Vgs/V_T, saturating in Vds), and an MLP with a smooth activation is a smooth function. Trees are piecewise constant: they need very many leaves to follow an exponential over 12 decades, and they step between training points.
4. **Physical structure can be built in.** One network predicts A, Ig and Ib together, and KCL is enforced exactly by construction (§3.4.1). The tree families predict each current separately and do not satisfy KCL.
5. **Cost.** Training takes 15–19 min per case including the ensemble, against 3.5 h for XGBoost and 2.7 h for RF. Prediction takes 0.04 ms per sample, which is fast for the many calls in Parts B and C.

XGBoost is kept as the documented runner-up.

---

## 6. Error analysis and how the MLP was improved (v1 → v2)

The first MLP (v1) predicted Id, Ig, Is and Ib directly, with MSE loss and
early stopping on the hold-out MSE. NMOS_ON reached 1.1 % median error, but the
other three cases were stuck at 2.6–3.0 %. `analyze_errors.py` breaks the
validation error down by current and operating region. It uses TRAIN and VAL
only, never test. The reports are `results/v1/error_analysis.md` (v1) and
`results/error_analysis.md` (v2).

### 6.1 What the analysis of v1 found

1. **Id and Is were the problem; Ig and Ib were not.**

   | Case | v1 val median: Id / Ig / Is / Ib |
   |---|---|
   | NMOS_ON | 1.54 / 0.65 / 1.59 / 0.58 % |
   | NMOS_OFF | 3.71 / 1.31 / 5.51 / 0.91 % |
   | PMOS_ON | 3.84 / 1.05 / 3.76 / 0.98 % |
   | PMOS_OFF | 4.22 / 1.51 / 4.77 / 1.37 % |

2. **The labels are clean.** The SPICE KCL residual \|Id+Ig+Is+Ib\| is about 0 % of \|Id\| and \|Is\|, so the error is not noise in the data.
3. **The models underfit.** Even on TRAIN, v1 reached only 2.7–4.7 % median error on OFF Id/Is, against 0.55 % for NMOS_ON. The error was spread evenly over Vds, temperature and current size.
4. **In the OFF state, Id is a near-cancellation.** Write Id = A + S, with A = (Id−Is)/2 and S = (Id+Is)/2 = −(Ig+Ib)/2.
   - The leakage part S dominates in 58 % (NMOS_OFF) and 72 % (PMOS_OFF) of samples.
   - The cancellation factor κ = (\|A\|+\|S\|)/\|Id\| has a 95th percentile of 14.7 (NMOS_OFF) and 24.6 (PMOS_OFF). In the worst 5 % of samples, Id is the difference of two currents 15–25× larger than Id itself.
   - A direct model must learn this difference in log space, including a sign flip of Id at a location that changes from sample to sample.
5. **A few samples dominated training.** In the OFF cases, 79–88 % of the squared error on Id/Is came from the worst 1 % of samples. Both MSE training and MSE-based early stopping chased those few points instead of the bulk.
6. **The pieces are simple.**
   - Ig and Ib never change sign within a case.
   - In the OFF cases, S never changes sign either.
   - sign(A) = sign(Vd−Vs) in 98.8–99.99 % of samples.

### 6.2 Recipe study (`mlp_experiments.py`)

Twelve recipes were trained on the same 256-256-256 SiLU network, one member
each:
- **Target:** direct, KCL, or KCL with imposed sign.
- **Loss:** MSE or Huber.
- **Early stopping monitor:** loss, or the spec metric.

The recipe was chosen on the **inner hold-out** (part of TRAIN). Validation
numbers are shown only for information.

| Target | Loss | Early stop on | Inner-hold-out median, NMOS_ON / NMOS_OFF / PMOS_ON / PMOS_OFF | Val median, same order |
|---|---|---|---|---|
| direct (v1) | MSE | loss | 1.14 / 2.95 / 2.32 / 3.05 | 1.09 / 2.86 / 2.41 / 2.97 |
| direct | MSE | metric | 1.08 / 1.63 / 1.67 / 1.52 | 1.09 / 1.64 / 1.81 / 1.49 |
| direct | Huber | loss | 1.33 / 1.27 / 1.30 / 1.57 | 1.40 / 1.22 / 1.28 / 1.51 |
| direct | Huber | metric | 1.18 / 1.20 / 1.06 / 1.24 | 1.30 / 1.15 / 1.13 / 1.25 |
| KCL | MSE | loss | 1.90 / 2.41 / 2.48 / 1.86 | 1.97 / 2.25 / 2.54 / 1.83 |
| KCL | MSE | metric | 1.79 / 1.86 / 2.35 / 1.28 | 1.97 / 1.82 / 2.39 / 1.29 |
| KCL | Huber | loss | 1.94 / 1.32 / 2.60 / 1.21 | 1.93 / 1.22 / 2.63 / 1.15 |
| KCL | Huber | metric | 2.13 / 1.19 / 1.87 / 1.13 | 2.25 / 1.06 / 1.94 / 1.13 |
| KCL + sign | MSE | loss | 1.18 / 2.71 / 1.72 / 1.04 | 1.36 / 2.76 / 1.76 / 0.98 |
| KCL + sign | MSE | metric | 1.11 / 1.18 / 1.29 / 1.09 | 1.25 / 1.19 / 1.26 / 1.10 |
| KCL + sign | Huber | loss | 0.92 / 0.84 / 1.19 / 0.67 | 0.94 / 0.89 / 1.16 / 0.63 |
| **KCL + sign** | **Huber** | **metric** | **0.84 / 0.81 / 1.04 / 0.67** | **0.88 / 0.85 / 1.06 / 0.62** |

(All values are mean median error in %, averaged over the four currents.)

**Takeaways:**
- **The v1 row reproduces the original v1 results:** 1.09 / 2.86 / 2.97 % on val for NMOS_ON / NMOS_OFF / PMOS_OFF, exactly as before. PMOS_ON gives 2.41 % here against v1's 2.63 %, because v1 had picked a different architecture (512-512-256-128) for that case. So the study is a fair comparison.
- **KCL without the imposed sign is *worse* than direct** in the ON cases. A signed A flips sign exactly at Vds = 0, and that jump is hard to learn. **Imposing sign(A) is the key step.**
- **Huber loss and metric-based early stopping each help,** and they add up.
- **The same recipe wins in all four cases,** so it became the default (`config.py`).
- **On top of the winning recipe,** the 6-architecture search and the 5-member ensemble cut the val median by a further 35–40 %: from 0.88 / 0.85 / 1.06 / 0.62 to 0.51 / 0.53 / 0.69 / 0.40 %.

### 6.3 Where the remaining error is (v2)

- **OFF Id/Is tail:** p95 8–10 %. It comes from samples where the leakage part dominates (\|A\|/(\|A\|+\|S\|) < 0.9): NMOS_OFF Id has a p95 of 12–15 % there, against about 3 % elsewhere. Two things contribute:
  - the cancellation of §6.1, item 4, where a 0.5 % error in A or S still becomes about 8 % on Id;
  - the 0.6–1.2 % of OFF samples where the imposed sign of A is wrong (§12).
- **ON Ig tail:** p95 6.5–10 %. It comes from deep subthreshold, where Vgs is more than 10·V_T below Vth. There, Ig has a median error of 11–15 % and a p95 of 44–76 %. That region holds 4.6 % of NMOS_ON samples and 7.4 % of PMOS_ON samples, which is why PMOS_ON's Ig tail is larger.
- **Everywhere else,** the error is 0.3–0.8 % median.

---

## 7. Do the MLPs differ between cases?

**Architecture:** no. All four cases chose the same 256-256-256 SiLU network.
The parameter counts differ only through the number of inputs (138–139k per
network, × 5 networks). The v1 MLP had chosen a bigger 512-512-256-128 net for
PMOS_ON; with the v2 recipe that net lost in every case (§3.4.4).

**Weights:** yes. Each case has its own five networks, trained from scratch on
its own data.

**Accuracy:** the remaining differences come from the physics, not the model:
- the OFF cases have the cancellation tail in Id/Is (§6.3);
- PMOS_ON has more deep-subthreshold samples, where Ig is hardest.

---

## 8. Test-set results of the chosen models (Deliverable 5)

These are evaluated on 1,500 test samples per case, after everything else was
fixed.

| Case | Current | R² (log) | R² (raw) | Median % | p95 % | Mean % | Sign accuracy % |
|---|---|---|---|---|---|---|---|
| NMOS_ON | Id | 1.0000 | 0.9999 | 0.51 | 3.63 | 1.51 | 99.87 |
| | Ig | 0.9993 | 0.9998 | 0.47 | 7.27 | 2.08 | 100 |
| | Is | 1.0000 | 0.9999 | 0.52 | 3.63 | 1.54 | 99.93 |
| | Ib | 1.0000 | 0.9995 | 0.44 | 2.42 | 0.76 | 100 |
| NMOS_OFF | Id | 0.9995 | 0.9998 | 0.74 | 8.32 | 3.55 | 99.87 |
| | Ig | 0.9999 | 0.9999 | 0.35 | 1.97 | 0.74 | 100 |
| | Is | 0.9977 | 0.9998 | 0.73 | 8.26 | 5.06 | 99.80 |
| | Ib | 0.9975 | 0.9988 | 0.35 | 4.18 | 2.53 | 100 |
| PMOS_ON | Id | 1.0000 | 0.9999 | 0.74 | 3.60 | 1.57 | 100 |
| | Ig | 0.9987 | 0.9999 | 0.59 | 9.67 | 2.64 | 100 |
| | Is | 1.0000 | 0.9999 | 0.74 | 3.68 | 2.19 | 100 |
| | Ib | 0.9996 | 0.9995 | 0.64 | 2.51 | 1.18 | 100 |
| PMOS_OFF | Id | 0.9997 | 0.9997 | 0.47 | 5.67 | 1.76 | 99.93 |
| | Ig | 0.9999 | 1.0000 | 0.27 | 1.52 | 0.61 | 100 |
| | Is | 0.9999 | 0.9997 | 0.43 | 4.91 | 2.31 | 99.93 |
| | Ib | 0.9971 | 0.9991 | 0.42 | 4.03 | 4.12 | 100 |

**Mean over the four currents (test), v1 → v2:**

| Case | Median % | p95 % | Worst-current p95 % | Mean % | Median accuracy |
|---|---|---|---|---|---|
| NMOS_ON | 1.13 → **0.49** | 7.28 → **4.24** | 8.96 → **7.27** | 5.27 → **1.47** | **99.5 %** |
| NMOS_OFF | 2.81 → **0.54** | 13.7 → **5.68** | 23.6 → **8.32** | 5.11 → **2.97** | **99.5 %** |
| PMOS_ON | 2.60 → **0.68** | 12.8 → **4.86** | 17.3 → **9.67** | 4.45 → **1.89** | **99.3 %** |
| PMOS_OFF | 2.94 → **0.40** | 15.4 → **4.03** | 25.6 → **5.67** | 5.68 → **2.20** | **99.6 %** |

**Takeaways:**
- **No over-fitting:** test agrees with validation to within about 0.03 percentage points on median error.
- **Against the targets:**
  - The TA's annotated target, R² > 0.99 and error < 1 %, is met on the median in every case and current. R² is ≥ 0.997 in log space and ≥ 0.998 in amperes.
  - The spec's "up to 99 % mean accuracy" is met as median accuracy (99.3–99.6 %). Mean accuracy (100 − mean error) is 97.0–98.5 %; it is pulled down by samples where \|I_SPICE\| is close to 0.
- **Against NMOS_ON's v1 level** (the bar the other cases were measured against): all three other cases now beat it on median, mean p95 and mean error. On worst-current p95, NMOS_OFF (8.32 %) and PMOS_OFF (5.67 %) beat it, and PMOS_ON is slightly above (9.67 % on Ig against 8.96 %).

**Plots** (`results/plots/`):
- `<case>_parity_test.png`: ML vs SPICE on log axes.
- `<case>_error_hist_test.png`: distribution of the relative error.
- `<case>_mlp_learning_curve.png`: loss and median error per epoch for ensemble member 1.

---

## 9. Feature ablation: physics vs raw inputs (validation)

The final MLP recipe and XGBoost were also trained on the 10 raw inputs only,
with the same search.

| Case | MLP physics | MLP raw | XGBoost physics | XGBoost raw |
|---|---|---|---|---|
| NMOS_ON | **0.51 %** (p95 3.7) | 1.72 % (p95 14.8) | 6.27 % | 12.67 % |
| NMOS_OFF | **0.53 %** (6.1) | 1.20 % (13.4) | 5.24 % | 5.92 % |
| PMOS_ON | **0.69 %** (5.4) | 1.81 % (19.0) | 5.96 % | 11.88 % |
| PMOS_OFF | **0.40 %** (4.7) | 1.09 % (12.9) | 4.22 % | 6.05 % |

(Each cell is the mean median error; MLP p95 in brackets.)

- The physics features cut the MLP's median error by 2.2–3.4× and its p95 by 2.2–3.9×.
- Even on raw inputs, the v2 MLP beats physics-feature XGBoost by 3–4×. Most of the gain comes from the KCL recipe; the physics features add the rest.

**Permutation importance (MLP, on VAL):**
- vd and vs are by far the most important inputs in every case. The sign of A is computed from them, so shuffling either one flips the sign of the channel current.
- Next come (Vgs_eff − Vth0)/V_T, vgs_eff, vbs_eff and \|vds\|: the subthreshold overdrive and the drain bias.

Files: `results/<case>_permutation_importance.csv` and
`plots/<case>_importance.png`.

## 10. How much data is needed? (`data_size_curve.py`)

The final MLP recipe, including the 5-member ensemble, was retrained on random
subsets of TRAIN and scored on VAL. Each cell is the mean median error, with the
mean p95 in brackets.

| Training samples | NMOS_ON | NMOS_OFF | PMOS_ON | PMOS_OFF |
|---|---|---|---|---|
| 700 (10 %) | 7.5 % (35) | 9.9 % (93) | 8.2 % (40) | 7.8 % (138) |
| 1,400 (20 %) | 4.6 % (23) | 5.6 % (49) | 4.7 % (26) | 4.9 % (59) |
| 2,450 (35 %) | 2.7 % (14) | 3.1 % (29) | 3.2 % (19) | 2.3 % (29) |
| 3,500 (50 %) | 1.7 % (10) | 1.8 % (17) | 1.7 % (15) | 1.2 % (15) |
| 5,250 (75 %) | 0.97 % (7.0) | 0.74 % (8.5) | 1.1 % (9.6) | 0.68 % (7.8) |
| 7,000 (100 %) | **0.51 % (3.8)** | **0.53 % (6.1)** | **0.69 % (5.4)** | **0.40 % (4.7)** |

**Takeaways:**
- Error falls steadily all the way to 7,000 training samples: the median drops by 28–47 % between 5,250 and 7,000, and the p95 by the same amount. The curve has not flattened.
- With 5,250 samples, the < 1 % median target is met in three of the four cases (PMOS_ON is at 1.1 %). With 7,000 it is met in all four.
- **Conclusion:** 10,000 samples per case (7,000 for training) meets the target and is what Deliverable 1 asks for, so the dataset was kept at that size. The spec does allow more samples ("may be increased or decreased as needed"). The curve suggests more data would shrink the remaining tails further, at the cost of deviating from the 10,000-row deliverable.

Plot: `results/plots/data_size_curve.png`.

---

## 11. Use of the test set

The test rows are read in exactly one place, `evaluate.py`, after each model is
fixed. They are used only to compute the reported metrics and plots. No
weight, hyperparameter, recipe or model choice ever used test data.

The test metrics were computed twice: once for v1 and once for v2. The v2
redesign was motivated by the error analysis on TRAIN and VAL (§6.1), and the
v2 recipe and architecture were chosen on the inner hold-out (part of TRAIN).
The close agreement between validation and test (§8) is consistent with no
over-fitting to the test set.

## 12. Limitations

- **OFF Id/Is tail (p95 about 8 %):** this comes from the near-cancellation of channel and leakage currents (§6.3). It can only shrink as A and S get more accurate, i.e. with more data.
- **ON Ig in deep subthreshold (p95 7–10 %):** this region is sparsely sampled, at 5–7 % of samples. A feature for the gradual weak-to-strong inversion transition (an EKV-style interpolation) is the targeted fix.
- **The mean relative error is inflated by samples with \|I\| ≈ 0.** Median and p95 describe typical accuracy better.
- **The imposed sign of A is wrong in a few samples:** 1.23 % of NMOS_OFF, 0.63 % of PMOS_OFF, 0.08 % of PMOS_ON and 0.01 % of NMOS_ON. In these, A has the opposite sign to Vd−Vs, probably because the gate-to-channel current is split unevenly between source and drain. Even with \|A\| predicted exactly, the wrong sign gives these samples a large Id/Is error: 19–30 % median in the OFF cases. They are too few to affect the median, but they add to the OFF tail (p95 and above). A possible fix is to let the network predict a signed correction to A near the sign change instead of imposing the sign everywhere.
- **The tree models were searched more narrowly than they could be** (RF depth fixed at 16), because of GPU time. Their gap to the MLP is large enough that this does not change the choice.

## 13. v3 (final): junction geometry, more data, mean-error target

The target is **mean** relative error < 1 % per current (TA's annotation), not
the median. v2 met it on the median but not the mean, which was 1.5–3.0 %.

### 13.1 New dataset (`data_set_generattion/`)

| File (per case) | Rows | Content | Used for |
|---|---|---|---|
| `<CASE>_new.csv` | 10,000 | The **same input rows** as `<CASE>.csv` (seed 42), re-simulated with junction geometry | Train / val / test (the same 70/15/15 rows as before) |
| `<CASE>_new_extra.csv` | 30,000 | New rows from the same distributions (seed 4242) | **Training only** |
| `<case>_new.net` | – | Example netlists (Deliverable 2) | – |
| `plot*_new.png` | – | KCL check and trend plots (Deliverable 3) | – |

So each case has 37,000 training rows (7,000 + 30,000). Validation and test
are the same 1,500 + 1,500 rows as before. The Deliverable 1 files stay at
10,000 rows; the extra rows are a disclosed supplement, as allowed by the
spec ("the number of samples may be increased … as needed").

**Junction geometry** (requested by the TA): every transistor now has
`AD = AS = W·Ldiff` and `PD = PS = 2·(W + Ldiff)`, with **Ldiff = 2·Lmin = 90 nm**.
- Without AD/AS the netlist had no junction area, so the reverse-biased drain/source junction leakage was missing.
- With them, Id, Ig and Is barely change (median change 0–0.3 %), but Ib changes completely in the OFF state: about 500× larger at 125 °C.
- The old data's constant Ib of about 1e-14 A at low temperature turns out to have been an artefact of the missing junction area.
- A useful side effect: sign(A) = sign(Vd−Vs) now holds in all but 1 of the 160,000 rows (before, it failed in up to 1.2 % of a case), which removes an error source of v2 (§12).

**Body-referenced netlists.**
- **Problem:** in the first new run, 53 % of PMOS_OFF Ib values (and 9 % of PMOS_ON Ib) were exact multiples of 2⁻⁵⁵ ≈ 2.8e-17 A, or exactly 0. ngspice solves voltages and currents together, so a branch current carries an absolute round-off of about machine-ε × the node voltage. The PMOS body sits at VDD ≈ 0.9 V, which puts about ±2.8e-17 A of floating-point noise on the tiny low-temperature body current.
- **Fix:** each netlist now uses the **body node as ground**, i.e. every source is shifted by −Vb. BSIM4 depends only on voltage differences, so the device biases are identical: Id and Is change by ≤ 2e-6, and Ig by ≤ 0.1 % (that was round-off in the old values too). PMOS Ib becomes smooth and positive (2e-19 to 1e-17 A at low temperature), with no zeros and no false sign changes. The CSVs keep the absolute voltages of spec §4. NMOS is unchanged, because its body is already at 0 V.
- **Metric safeguard:** relative error is undefined where the SPICE current is exactly 0. `metrics.relative_error` skips such rows and reports their count. After the fix there are none.

### 13.2 v3 recipe study (`mlp_experiments.py`, `v3/results/experiments/`)

Five recipes were trained on 256-256-256 SiLU, with one network each and all
37,000 training rows. The recipe was chosen on the **inner-hold-out mean
error** (part of TRAIN). Validation numbers are shown for information only.

| Recipe | Inner-hold-out mean error, NMOS_ON / NMOS_OFF / PMOS_ON / PMOS_OFF | Val mean error, same order |
|---|---|---|
| v2 recipe (KCL + sign, Huber, stop on median) | 0.34 / 0.52 / 0.63 / 0.31 % | 0.30 / 0.55 / 0.26 / 0.24 % |
| + stop on **mean** error | **0.34** / 0.52 / **0.63** / 0.30 % | 0.30 / 0.55 / 0.26 / 0.26 % |
| + stop on mean + **κ-weighted loss** | 0.39 / **0.33** / 0.92 / **0.29** % | 0.35 / 0.33 / 0.41 / 0.22 % |
| KCL without imposed sign, stop on mean | 0.61 / 0.72 / 0.95 / 0.70 % | 0.62 / 0.61 / 0.48 / 0.53 % |
| KCL without imposed sign, stop on mean, κ-weighted | 0.71 / 0.49 / 1.30 / 0.46 % | 0.93 / 0.47 / 0.73 / 0.36 % |

**κ-weighted loss:** each training row is weighted by κ = (\|A\|+\|S\|)/\|I\|,
averaged over Id and Is, clipped at 50 and normalised to mean 1
(`models.kappa_weights`). This is how strongly KCL amplifies errors in A and S
into Id/Is, so the weighting makes the network learn A and S most precisely
where Id or Is is a near-cancellation.

**Takeaways:**
- Even with the v2 recipe, the extra data alone cut the error by about 3–5× (val median 0.09–0.19 %, against 0.40–0.69 % for v2).
- **The κ-weighted loss helps the OFF cases** (NMOS_OFF inner mean 0.52 → 0.33 %), where the cancellation lives. **It hurts the ON cases**, where κ ≈ 1 and the weights only add noise.
- **Imposing the sign of A is still essential:** every unsigned variant is worse.
- **Chosen per case** (`config.MLP_CASE_RECIPES`): KCL + sign, Huber δ = 0.1, early stopping and selection on the mean error, κ weighting in NMOS_OFF and PMOS_OFF only.

### 13.3 Final v3 models

- **Architecture search:** the SiLU networks 256-256-256, 512-512-256-128 and 1024-512-256. ReLU, dropout and GELU lost in every case of the v2 search and were dropped to save GPU time.
- **Selection** is on the inner-hold-out mean error.
- **Ensemble:** 5 members per case, as in v2 (§3.4.3), each with its own seed and hold-out.
- **Training:** everything else is unchanged from v2: AdamW lr 2e-3, batch 256, ReduceLROnPlateau, patience 150, up to 3,000 epochs.

| Case | Recipe | Chosen network | Parameters per network (× 5) | Epochs of the 5 members | Fit time |
|---|---|---|---|---|---|
| NMOS_ON | no κ weighting | 512-512-256-128 | 440,579 | 1592, 3000, 3000, 859, 947 | 141 min |
| NMOS_OFF | κ-weighted | 256-256-256 | 138,243 | 741, 1339, 1498, 841, 1385 | 77 min |
| PMOS_ON | no κ weighting | 256-256-256 | 139,011 | 1764, 3000, 346, 2026, 1475 | 110 min |
| PMOS_OFF | κ-weighted | 256-256-256 | 138,755 | 1872, 1491, 2214, 991, 1373 | 107 min |

One PMOS_ON member stopped early, after 346 epochs, and is weaker than the
other four. The ensemble still reaches the target, but retraining that member
is a possible small gain.

Prediction takes 0.03–0.05 ms per sample for the whole ensemble.

### 13.4 v3 test results (Deliverable 5; `v3/results/chosen_models_test.md`)

| Case | Current | Median % | p95 % | **Mean %** | R² (log) | Sign % |
|---|---|---|---|---|---|---|
| NMOS_ON | Id | 0.100 | 0.61 | **0.32** | 1.000 | 100 |
| | Ig | 0.085 | 1.04 | **0.35** | 1.000 | 100 |
| | Is | 0.100 | 0.61 | **0.35** | 1.000 | 100 |
| | Ib | 0.088 | 0.47 | **0.15** | 1.000 | 100 |
| NMOS_OFF | Id | 0.063 | 0.37 | **0.40** | 1.000 | 100 |
| | Ig | 0.038 | 0.17 | **0.06** | 1.000 | 100 |
| | Is | 0.062 | 0.38 | **0.13** | 1.000 | 100 |
| | Ib | 0.103 | 0.49 | **0.18** | 1.000 | 100 |
| PMOS_ON | Id | 0.144 | 0.60 | **0.22** | 1.000 | 100 |
| | Ig | 0.120 | 0.60 | **0.23** | 1.000 | 100 |
| | Is | 0.144 | 0.58 | **0.31** | 1.000 | 100 |
| | Ib | 0.150 | 0.49 | **0.19** | 1.000 | 100 |
| PMOS_OFF | Id | 0.054 | 0.45 | **0.12** | 1.000 | 100 |
| | Ig | 0.033 | 0.17 | **0.07** | 1.000 | 100 |
| | Is | 0.052 | 0.40 | **0.14** | 1.000 | 100 |
| | Ib | 0.093 | 0.52 | **0.19** | 1.000 | 100 |

R² on raw amperes is ≥ 0.99995 everywhere.

**v2 → v3, test, mean over the four currents:**

| Case | Median % | p95 % | Worst-current p95 % | **Mean %** | Mean accuracy |
|---|---|---|---|---|---|
| NMOS_ON | 0.49 → **0.093** | 4.24 → **0.68** | 7.27 → **1.04** | 1.47 → **0.29** | **99.71 %** |
| NMOS_OFF | 0.54 → **0.067** | 5.68 → **0.35** | 8.32 → **0.49** | 2.97 → **0.19** | **99.81 %** |
| PMOS_ON | 0.68 → **0.139** | 4.86 → **0.57** | 9.67 → **0.60** | 1.89 → **0.24** | **99.76 %** |
| PMOS_OFF | 0.40 → **0.058** | 4.03 → **0.38** | 5.67 → **0.52** | 2.20 → **0.13** | **99.87 %** |

**Takeaways:**
- **The target is met everywhere:** mean error < 1 % for every case and every current (the highest is 0.40 %, NMOS_OFF Id), and R² > 0.99.
- **Every metric improved on v2:** median 5–8×, mean 5–17×, p95 6–16×.
- Validation and test agree closely: val mean over the currents is 0.21 / 0.14 / 0.26 / 0.14 %, against 0.29 / 0.19 / 0.24 / 0.13 % on test.
- **Caveat:** v2 and v3 are scored on different labels, the original and the junction-geometry data. Id, Ig and Is are essentially the same physics in both, but Ib is not.

### 13.5 How much did the extra data help? (`v3/results/data_size_curve.csv`)

The final v3 recipe (5-member ensemble), retrained on random subsets of the
37,000 training rows and scored on VAL:

| Training rows | NMOS_ON | NMOS_OFF | PMOS_ON | PMOS_OFF |
|---|---|---|---|---|
| 7,000 (= v2's training set) | 0.74 / **1.63 %** | 0.41 / 0.87 % | 0.54 / **1.88 %** | 0.50 / **1.39 %** |
| 18,500 | 0.17 / 0.37 % | 0.10 / 0.24 % | 0.13 / 0.36 % | 0.10 / 0.26 % |
| 37,000 | 0.092 / 0.21 % | 0.064 / 0.14 % | 0.144 / 0.26 % | 0.056 / 0.14 % |

(Each cell is the validation median / **mean** error, averaged over the four
currents. Plot: `v3/results/plots/data_size_curve.png`.)

**Takeaways:**
- **The extra data is necessary for the target.** With only the 7,000 rows of the original split, the v3 recipe misses mean < 1 % in three of the four cases (0.87–1.88 %). This is the evidence for going beyond the 10,000-row dataset.
- **18,500 training rows already meet the target** (mean 0.24–0.37 %). 37,000 halve the mean error again, and the p95 too (0.38–0.62 %).
- PMOS_ON's *median* is slightly lower at 18,500 than at 37,000 (0.126 vs 0.144 %). That is the weak early-stopped ensemble member of the final model (§13.3); its mean and p95 are still best at 37,000.

### 13.6 Test set and limitations (v3)

- **Test set:** the v3 test labels are new (junction geometry), so v3's test evaluation is the first on these labels. The test *input* rows are the same 1,500 per case as in v1/v2, and no model was trained or selected on them.
- **Disclosure:** the extra 30,000 rows per case go beyond the 10,000-row deliverable. They are uniform draws from the spec's §4 distributions with a different seed.
- **Remaining tail:** the largest p95 is NMOS_ON Ig (1.04 %), in deep subthreshold (§6.3).

## 14. Files and how to reproduce

The dataset is chosen with `LEAKAGE_DATASET`: `new` (the default, v3) reads
`<CASE>_new.csv` + `<CASE>_new_extra.csv` and writes to `v3/`; `orig` reads
`<CASE>.csv` and writes to `results/` and `saved_models/` (v1, v2).

| What | v3 (final, `new` data) | v2 (`orig` data) |
|---|---|---|
| Final models (Deliverable 5). Load with `LeakagePredictor.load(path).predict(df)`, which returns amperes | `v3/saved_models/<CASE>.joblib` | `saved_models/<CASE>.joblib` |
| Search traces and all metrics | `v3/results/candidates/<case>__mlp.json` | `results/candidates/<case>__<family>[__raw].json` |
| Validation table (Deliverable 4) | `v3/results/model_comparison_val.md` (MLP only) | `results/model_comparison_val.md` (all 4 families) |
| Test results (Deliverable 5) | `v3/results/chosen_models_test.md` / `.csv` | `results/chosen_models_test.md` / `.csv` |
| Size curve | `v3/results/data_size_curve.csv` | `results/data_size_curve.csv` |
| Recipe study | `v3/results/experiments/` | `results/experiments/` |
| Error analysis | – | `results/error_analysis.md`, `results/v1/error_analysis.md` |
| Plots | `v3/results/plots/` | `results/plots/` |
| Feature ablation | – | `results/feature_ablation_val.md` |

v1 results are in `results/v1/`. Logs are in `logs/` and `v3/logs/` (not in
git). The candidate models are on Ada.

```bash
# Data (laptop, needs ngspice): 10,000 main + 30,000 extra rows per case, then the data check
cd data_set_generattion && python generate_data.py --extra 30000 && python check_data.py

# On Ada (GPU), v3 = default dataset "new":
sbatch run_experiments.sh                                  # v3 recipe study (§13.2)
MODELS=mlp STEPS="train eval size" sbatch run_all.sh      # final v3 MLPs, test evaluation, size curve
# v2 / all four families on the original data:
LEAKAGE_DATASET=orig sbatch run_all.sh
python analyze_errors.py                                   # error breakdown (§6), after evaluate.py
```
