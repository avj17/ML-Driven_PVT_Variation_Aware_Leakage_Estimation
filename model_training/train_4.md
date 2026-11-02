# Part A: Training the four transistor models

Team StaticSilicon. Model file 45nm_HP, VDD = 0.9 V, L = 45 nm.

This document records how the four model families were trained for the four
transistor cases (NMOS ON/OFF, PMOS ON/OFF), the architecture and
hyperparameters of each, every metric, and why the MLP was chosen. All numbers
come from `results/` (the JSON files in `results/candidates/` hold the full
search traces).

**Short version.** The MLP is the best family in all four cases on the
validation set. On the test set its median error per terminal current is
0.56–5.4 %, R² (signed-log) is 0.995–1.000, and the sign of the current is right
99.5–100 % of the time.

---

## 1. Setup

| Item | Value |
|---|---|
| Data | 4 CSVs (one per case), 10,000 ngspice samples each (`data_set_generattion/`) |
| Split | 70 / 15 / 15 → 7,000 train / 1,500 val / 1,500 test, `train_test_split`, seed 42. Indices saved in `results/splits/<case>_split.json` so every model uses the same rows |
| Targets | The four terminal currents Id, Ig, Is, Ib |
| Hardware | Ada HPC, one RTX 2080 Ti (gnode043), 10 CPUs, 20 GB RAM |
| Software | Python 3.12.4, PyTorch 2.14.1 (CUDA 12.6), XGBoost 3.4.1 (`xgboost-cu12`), scikit-learn 1.9 |
| Seeds | `SEED_SPLIT=42`, `SEED_CV=43`, `SEED_MODEL=44`, `SEED_TORCH=45` (separate streams, `leakage_ml/config.py`) |

### Rule for using the three splits

- **TRAIN (7,000):** used for everything that is fitted or tuned: scalers, the target transform's I0, dropped-column decisions, and all hyperparameter search (5-fold CV, or an inner 10 % hold-out for the MLP).
- **VAL (1,500):** used only to compare the four families and pick one per case.
- **TEST (1,500):** used once, in `evaluate.py`, on the chosen model only.

## 2. Preprocessing (same for every family)

### 2.1 Target transform: signed log

The currents span about 12 decades (1e-17 to 5e-4 A), and Id/Is change sign
(about 50 % negative in the ON cases, 13–21 % in the OFF cases). So every model
predicts

    y = sign(I) · log10(1 + |I| / I0)

with one I0 per current, set on TRAIN as 0.1 × the 1st percentile of |I|.

- For |I| ≫ I0 this is log10(|I|/I0). An error dy in y is then a relative current error of about ln(10)·dy, so minimising MSE in y approximately minimises relative error, which is what the spec scores.
- Unlike plain log|I|, it keeps the sign and is smooth through I = 0.

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

Predictions are converted back to amperes with the inverse transform. Then, for
each current:

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
and a median error of 97–117 %, so every model below is far better than
chance.

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

**Architecture:** one multi-output network per case, which predicts all four
currents together:

    input (22–25 features) → [Linear → SiLU] × k → Linear(4)

| Setting | Value |
|---|---|
| Input / output scaling | `StandardScaler` on X and on y (fit on the training part) |
| Initialisation | He (Kaiming normal) weights, zero bias |
| Loss | MSE on standardised signed-log targets |
| Optimiser | AdamW, lr 2e-3, weight decay 1e-6 |
| LR schedule | ReduceLROnPlateau, factor 0.5, patience 50, minimum 1e-6 |
| Batch size | 256 |
| Epochs | Up to 3000. Early stopping with patience 150; the best weights are restored (`copy.deepcopy`) |
| Early-stopping data | An inner 10 % hold-out from TRAIN (6,300 fit / 700 hold-out). The official VAL set is never seen during training |

**Search:** 6 configurations are trained per case. The one with the lowest
inner hold-out MSE is kept.

| # | Hidden layers | Activation | Dropout | Weight decay | lr | Batch |
|---|---|---|---|---|---|---|
| 1 | 256-256-256 | SiLU | 0 | 1e-6 | 2e-3 | 256 |
| 2 | 512-512-256-128 | SiLU | 0 | 1e-6 | 2e-3 | 256 |
| 3 | 256-256-256 | ReLU | 0 | 1e-6 | 2e-3 | 256 |
| 4 | 512-512-256-128 | SiLU | 0.05 | 1e-5 | 2e-3 | 256 |
| 5 | 512-512-512-256 | GELU | 0 | 1e-6 | 1e-3 | 128 |
| 6 | 1024-512-256 | SiLU | 0 | 1e-6 | 3e-3 | 512 |

**Inner hold-out MSE** (standardised units; lower is better, **bold** = chosen):

| Config | NMOS_ON | NMOS_OFF | PMOS_ON | PMOS_OFF |
|---|---|---|---|---|
| 1. 256×3 SiLU | **9.21e-5** | **3.59e-3** | 6.92e-4 | **2.98e-3** |
| 2. 512-512-256-128 SiLU | 1.36e-4 | 3.76e-3 | **4.38e-4** | 3.08e-3 |
| 3. 256×3 ReLU | 3.91e-4 | 4.79e-3 | 1.11e-3 | 4.43e-3 |
| 4. #2 + dropout 0.05 | 1.40e-4 | 4.02e-3 | 5.20e-4 | 5.05e-3 |
| 5. 512×3-256 GELU | 1.82e-4 | 4.34e-3 | 4.81e-4 | 4.58e-3 |
| 6. 1024-512-256 SiLU | 3.31e-4 | 4.21e-3 | 9.82e-4 | 4.75e-3 |

**The four final networks:**

| Case | Inputs | Hidden layers | Trainable parameters | Epochs run | Fit time (all 6 configs) |
|---|---|---|---|---|---|
| NMOS_ON | 25 | 256-256-256 | 139,268 | 1642 | 11.3 min |
| NMOS_OFF | 22 | 256-256-256 | 138,500 | 594 | 6.5 min |
| PMOS_ON | 25 | 512-512-256-128 | 440,708 | 994 | 9.7 min |
| PMOS_OFF | 24 | 256-256-256 | 139,012 | 611 | 6.8 min |

**What the search shows:**
- SiLU beats ReLU in every case (2–4× lower hold-out MSE). The smooth activation suits currents that are smooth exponentials of voltage.
- Dropout never helped: with 6,300 noise-free SPICE samples the model is not overfitting much (see the train/val gap in §4).
- Bigger is not automatically better. The 1024-512-256 net lost in every case, and the 4-layer 512-wide net won only for PMOS_ON (see §6).

**Inference speed:** 0.01–0.03 ms per sample on CPU. That is the fastest of
the four families, which matters for Parts B and C.

---

## 4. Validation results: all models, all metrics

These numbers are from the validation set (1,500 samples per case). "Train
median" is the same metric on TRAIN, so the gap between it and "Median"
shows over-fitting.

### 4.1 Summary (Deliverable 4): mean over Id, Ig, Is, Ib

| Case | Model | R² (log) | Median error % | p95 error % | Worst p95 % | Train median % |
|---|---|---|---|---|---|---|
| NMOS_ON | Baseline (mean) | 0.000 | 97.2 | 2405 | 7044 | – |
| | Poly Elastic Net | 0.9971 | 13.80 | 44.3 | 57.1 | 13.60 |
| | Random Forest | 0.9926 | 12.98 | 63.2 | 76.9 | 4.63 |
| | XGBoost | 0.9987 | 6.27 | 29.9 | 36.5 | 2.08 |
| | **MLP** | **0.9999** | **1.09** | **6.3** | **7.8** | 0.34 |
| NMOS_OFF | Baseline (mean) | 0.000 | 116.7 | 681 | 2238 | – |
| | Poly Elastic Net | 0.9710 | 35.54 | 185.6 | 380.1 | 33.04 |
| | Random Forest | 0.9914 | 10.29 | 57.3 | 92.3 | 3.99 |
| | XGBoost | 0.9951 | 5.24 | 39.5 | 72.5 | 1.49 |
| | **MLP** | **0.9985** | **2.86** | **13.8** | **23.4** | 2.29 |
| PMOS_ON | Baseline (mean) | 0.000 | 96.8 | 2592 | 8693 | – |
| | Poly Elastic Net | 0.9965 | 13.79 | 50.8 | 68.4 | 13.31 |
| | Random Forest | 0.9938 | 12.46 | 65.9 | 82.4 | 4.93 |
| | XGBoost | 0.9987 | 5.96 | 29.0 | 38.1 | 2.03 |
| | **MLP** | **0.9997** | **2.63** | **13.9** | **20.0** | 1.22 |
| PMOS_OFF | Baseline (mean) | -0.001 | 116.2 | 761 | 1905 | – |
| | Poly Elastic Net | 0.9607 | 31.85 | 116.6 | 215.6 | 29.54 |
| | Random Forest | 0.9857 | 8.28 | 53.2 | 83.2 | 3.25 |
| | XGBoost | 0.9920 | 4.22 | 41.3 | 73.6 | 1.49 |
| | **MLP** | **0.9966** | **2.97** | **15.0** | **29.0** | 2.35 |

### 4.2 Per current (validation)

Each cell gives median % / p95 % / mean %. R² (log) and sign accuracy are below
the table.

| Case | Current | Poly Elastic Net | Random Forest | XGBoost | MLP |
|---|---|---|---|---|---|
| NMOS_ON | Id | 17.1 / 56.5 / 377 | 11.5 / 66.4 / 44.0 | 6.11 / 36.5 / 16.8 | **1.54 / 7.8 / 7.3** |
| | Ig | 6.97 / 24.7 / 9.5 | 13.4 / 43.1 / 16.9 | 5.31 / 17.5 / 6.8 | **0.65 / 7.2 / 2.0** |
| | Is | 17.1 / 57.1 / 123 | 11.4 / 66.2 / 27.5 | 5.69 / 35.1 / 11.4 | **1.59 / 7.7 / 3.7** |
| | Ib | 14.1 / 38.8 / 16.5 | 15.7 / 76.9 / 24.7 | 7.96 / 30.5 / 10.6 | **0.58 / 2.7 / 0.9** |
| NMOS_OFF | Id | 61.0 / 320 / 109 | 15.0 / 92.3 / 26.5 | 7.89 / 72.5 / 16.4 | **3.71 / 18.4 / 6.6** |
| | Ig | 1.67 / 6.0 / 2.4 | 9.84 / 39.2 / 14.3 | 4.93 / 16.1 / 6.2 | **1.31 / 4.8 / 1.8** |
| | Is | 66.5 / 380 / 108 | 16.1 / 87.5 / 30.3 | 7.63 / 61.5 / 17.9 | **5.51 / 23.4 / 13.2** |
| | Ib | 13.0 / 36.0 / 16.5 | **0.23** / 10.1 / 2.5 | 0.52 / **7.9** / **2.0** | 0.91 / 8.4 / 2.5 |
| PMOS_ON | Id | 18.0 / 61.9 / 25.6 | 13.2 / 81.0 / 30.5 | 5.95 / 38.1 / 12.3 | **4.10 / 19.2 / 7.0** |
| | Ig | 4.79 / 32.3 / 9.4 | 11.4 / 39.1 / 15.4 | 4.59 / 14.4 / 5.9 | **1.09 / 11.8 / 3.4** |
| | Is | 18.6 / 68.4 / 55.2 | 13.5 / 82.4 / 29.3 | 5.82 / 36.3 / 17.5 | **4.16 / 20.0 / 8.5** |
| | Ib | 13.8 / 40.7 / 16.3 | 11.8 / 61.2 / 20.3 | 7.50 / 27.0 / 10.0 | **1.17 / 4.6 / 1.6** |
| PMOS_OFF | Id | 55.1 / 216 / 104 | 10.6 / 77.3 / 21.4 | 5.43 / 73.6 / 16.0 | **4.22 / 17.6 / 7.6** |
| | Ig | 1.99 / 7.9 / 3.0 | 10.2 / 40.2 / 15.0 | 5.12 / 17.1 / 6.6 | **1.51 / 5.8 / 2.2** |
| | Is | 57.3 / 206 / 240 | 11.7 / 83.2 / 23.8 | 5.63 / 67.2 / 16.6 | **4.77 / 29.0 / 10.3** |
| | Ib | 13.0 / 37.5 / 18.8 | **0.64** / 12.0 / 3.6 | 0.69 / **7.1** / **2.1** | 1.37 / 7.7 / 4.0 |

**R² (log), MLP:** 0.9916–1.0000 on every current. The lowest values are the
OFF-state currents; the lowest of all is PMOS_OFF Is (0.9916).

**Sign accuracy:**
- MLP: 99.1–100 %, with Ig and Ib always 100 %.
- XGBoost: 98.9–100 %.
- Random Forest: 98.8–100 %.
- Elastic Net: 98.4–100 %.

**Other observations:**
- **Elastic Net R² (raw):** negative for OFF-state Id/Is (down to −3.8e6 on PMOS_OFF Is). A few predictions are off by orders of magnitude in amperes even though R² (log) is 0.92–0.95. A degree-3 polynomial cannot follow the exponential subthreshold behaviour over 12 decades.
- **Over-fitting (train median vs val median):**
  - The tree models over-fit the most: XGBoost reaches 1.5–2 % on train but 4–6 % on val, and RF 4–5 % vs 8–13 %.
  - The MLP gap is small: 0.3→1.1, 2.3→2.9, 1.2→2.6, 2.3→3.0.
  - Elastic Net has no gap, but high error on both: it is under-fitting (high bias).

---

## 5. Why the MLP was chosen

**Selection rule** (fixed before looking at results, `config.py`): lowest mean
median error over the four currents on VAL; ties are broken by mean p95.

The MLP wins in all four cases, and the win is not marginal:

1. **Accuracy.** Its median error is 1.4–5.8× lower than XGBoost, the next-best model, and its p95 is 2.1–4.7× lower. It wins 14 of the 16 per-current comparisons on median error. The two exceptions are OFF-state Ib, where RF and XGBoost reach 0.2–0.7 % against the MLP's 0.9–1.4 %, all already under 1.5 %.
2. **The tails matter for Parts B and C.** Gate and circuit leakage is a sum of transistor currents, mostly the OFF-transistor channel currents plus gate leakage. The MLP's worst p95 is 8–29 %, against 37–74 % for XGBoost and 77–92 % for RF. Fewer wild predictions mean the sums stay accurate.
3. **Smooth function.** The I–V relationship is smooth (exponential in Vgs/V_T, saturating in Vds), and an MLP with a smooth activation is a smooth function. Trees are piecewise constant: they need very many leaves to follow an exponential over 12 decades, and they step between training points. That shows up in their larger val–train gaps.
4. **One network for all four currents.** The currents are physically linked (KCL: Id + Ig + Is + Ib = 0), and a shared hidden representation learns that structure. The other families use four separate models.
5. **Cost.** Training takes 7–11 min per case, against 3.5 h for XGBoost and 2.7 h for RF. Prediction takes 0.01–0.03 ms per sample, which is fast for the many calls in Parts B and C.
6. **Over-fitting is controlled.** Early stopping uses a hold-out carved from TRAIN, and the train–val gap is the smallest of all the flexible models. Test results (§7) match validation.

XGBoost is kept as the documented runner-up.

---

## 6. Do the MLPs differ between cases?

Yes. Each case has its own network trained on its own data. Three cases chose
the same architecture (256-256-256, about 139k parameters; the small
differences come from the number of inputs). PMOS_ON chose the larger
512-512-256-128 net, with 441k parameters.

**The larger network is not why one case is more accurate than another:**
- The most accurate case, **NMOS_ON** (1.09 % median), uses the *small* net (139k).
- **PMOS_ON**, with the large net (441k), is at 2.63 %.
- For NMOS_ON, NMOS_OFF and PMOS_OFF, the larger 4-layer net was *worse* on the inner hold-out than 256×3.
- The 1024-512-256 net, with even more parameters, lost in every case.

The extra capacity helped only within PMOS_ON. There, the 512-512-256-128 net
had 37 % lower hold-out MSE than 256×3, which suggests the PMOS ON surface is
harder to fit.

The differences in accuracy between cases come from the physics:
- **OFF-state Id and Is are the hardest targets in every family.** Near Vds ≈ 0 the channel current becomes tiny and changes sign, and gate and junction leakage compete with it, so the relative error grows.
- **PMOS is harder than NMOS** for every family, not only the MLP.

All four cases went through the same 6-config search with the same rule, so
the comparison is fair. Each case simply gets the architecture its data
supports best.

---

## 7. Test-set results of the chosen models (Deliverable 5)

These are evaluated once, on 1,500 test samples per case.

| Case | Current | R² (log) | R² (raw) | Median % | p95 % | Mean % | Sign accuracy % |
|---|---|---|---|---|---|---|---|
| NMOS_ON | Id | 1.0000 | 0.9994 | 1.64 | 8.96 | 9.87 | 99.80 |
| | Ig | 0.9991 | 0.9998 | 0.63 | 8.60 | 2.48 | 100 |
| | Is | 1.0000 | 0.9994 | 1.68 | 8.93 | 7.87 | 99.80 |
| | Ib | 1.0000 | 0.9996 | 0.56 | 2.65 | 0.87 | 100 |
| NMOS_OFF | Id | 0.9990 | 0.9824 | 3.67 | 18.99 | 6.43 | 99.87 |
| | Ig | 0.9998 | 0.9996 | 1.29 | 4.68 | 1.75 | 100 |
| | Is | 0.9974 | 0.9917 | 5.36 | 23.55 | 8.91 | 99.80 |
| | Ib | 0.9970 | 0.9979 | 0.91 | 7.45 | 3.35 | 100 |
| PMOS_ON | Id | 1.0000 | 0.9972 | 4.16 | 17.32 | 6.36 | 100 |
| | Ig | 0.9988 | 0.9998 | 1.00 | 11.75 | 3.25 | 100 |
| | Is | 1.0000 | 0.9976 | 4.11 | 17.28 | 6.33 | 99.93 |
| | Ib | 0.9996 | 0.9993 | 1.12 | 4.66 | 1.84 | 100 |
| PMOS_OFF | Id | 0.9976 | 0.9889 | 4.21 | 21.50 | 7.13 | 99.80 |
| | Ig | 0.9994 | 0.9990 | 1.60 | 6.04 | 2.33 | 100 |
| | Is | 0.9950 | 0.9817 | 4.64 | 25.55 | 8.46 | 99.47 |
| | Ib | 0.9971 | 0.9967 | 1.29 | 8.34 | 4.79 | 100 |

**Mean over the four currents (test):**

| Case | Median % | p95 % | Mean % | Median accuracy |
|---|---|---|---|---|
| NMOS_ON | 1.13 | 7.3 | 5.3 | 98.9 % |
| NMOS_OFF | 2.81 | 13.7 | 5.1 | 97.2 % |
| PMOS_ON | 2.60 | 12.8 | 4.4 | 97.4 % |
| PMOS_OFF | 2.94 | 15.4 | 5.7 | 97.1 % |

**Takeaways:**
- **No over-fitting:** test agrees with validation to within about 0.1 percentage points on median error.
- **Gate and body currents are the most accurate:** 0.6–1.6 % median.
- **Channel currents are the least accurate:** Id/Is in the OFF and PMOS cases, at 3.7–5.4 % median.
- **Against the 99 % target:** median accuracy is 97–99 %. Mean accuracy (100 − mean error) is 90–99 % per current. The mean is pulled down by samples where |I_SPICE| is close to 0.

**Plots** (`results/plots/`):
- `<case>_parity_test.png`: ML vs SPICE on log axes.
- `<case>_error_hist_test.png`: distribution of the relative error.
- `<case>_mlp_learning_curve.png`: train and inner-validation loss per epoch.

---

## 8. Feature ablation: physics vs raw inputs (validation)

The two strongest families were retrained on the 10 raw inputs only, with the
same search.

| Case | MLP physics | MLP raw | XGBoost physics | XGBoost raw |
|---|---|---|---|---|
| NMOS_ON | **1.09 %** (p95 6.3) | 12.45 % (p95 58.8) | 6.27 % | 12.67 % |
| NMOS_OFF | **2.86 %** (13.8) | 5.67 % (34.9) | 5.24 % | 5.92 % |
| PMOS_ON | **2.63 %** (13.9) | 6.09 % (42.0) | 5.96 % | 11.88 % |
| PMOS_OFF | **2.97 %** (15.0) | 3.52 % (27.8) | 4.22 % | 6.05 % |

(Each cell is the mean median error; MLP p95 in brackets.)

- The physics features cut the MLP error by 1.2–11×, and the p95 by 1.9–9×.
- The ON cases gain the most: the network no longer has to discover Vgs, Vds and the current direction by itself.

**Permutation importance (MLP, on VAL):**
- The most important feature in every case is `dir`, the current direction.
- It is followed by `vov_over_vt` / `vgs_eff`, i.e. the subthreshold overdrive (Vgs − Vth)/V_T.

Files: `results/<case>_permutation_importance.csv` and
`plots/<case>_importance.png`.

## 9. How much data is needed? (`data_size_curve.py`)

The chosen MLP was refit with its tuned hyperparameters on random subsets of
TRAIN and scored on VAL. Each cell is the mean median error, with the mean p95
in brackets.

| Training samples | NMOS_ON | NMOS_OFF | PMOS_ON | PMOS_OFF |
|---|---|---|---|---|
| 700 (10 %) | 34.5 % (245) | 34.6 % (237) | 34.2 % (251) | 32.9 % (170) |
| 1,400 (20 %) | 23.8 % (126) | 24.8 % (138) | 20.3 % (86) | 22.3 % (121) |
| 2,450 (35 %) | 8.7 % (37) | 8.8 % (52) | 13.1 % (56) | 9.9 % (58) |
| 3,500 (50 %) | 5.6 % (25) | 6.9 % (39) | 7.7 % (36) | 6.3 % (41) |
| 5,250 (75 %) | 8.0 % (26) | 2.5 % (16) | 4.9 % (25) | 3.4 % (21) |
| 7,000 (100 %) | **1.1 % (6.3)** | 2.9 % (13.8) | **2.6 % (13.9)** | 3.0 % (15.0) |

**Takeaways:**
- Error falls steeply up to about 3,500 training samples, then keeps improving.
- For the ON cases, the full 7,000 samples give a clear further gain: NMOS_ON goes from 5.6 % to 1.1 %, PMOS_ON from 4.9 % to 2.6 %.
- For the OFF cases, 5,250 and 7,000 are about equal. NMOS_OFF is even slightly better at 5,250, which is within the run-to-run variation of a single MLP training.
- The NMOS_ON bump at 5,250 (8.0 %) is the same kind of single-run variation.
- **Conclusion:** 10,000 samples per case (7,000 for training) is the smallest size tried that gives the best accuracy in all four cases, so the dataset was not reduced. The ON curves have not flattened, so more data would likely help further.

Plot: `results/plots/data_size_curve.png`.

---

## 10. Limitations

- The weakest predictions are OFF-state Id/Is near Vds ≈ 0, with a median of 3.7–5.4 % and a p95 of 19–26 %. These currents are what gate leakage is built from.
- The **mean** relative error is inflated by samples with |I| ≈ 0. Median and p95 describe typical accuracy better.
- Each MLP is a single training run (seed 45). The bumps in the size curve show run-to-run variation; an ensemble of several seeds would reduce it.
- The tree models were searched more narrowly than they could be (RF depth fixed at 16), because of GPU time. Their gap to the MLP is large enough that this does not change the choice.

## 11. Files and how to reproduce

| What | Where |
|---|---|
| Final models (Deliverable 5) | `saved_models/<CASE>.joblib`. Load with `LeakagePredictor.load(path).predict(df)`, which returns amperes |
| All 24 candidate models (on Ada) | `saved_models/candidates/` |
| Search traces and all metrics | `results/candidates/<case>__<family>[__raw].json` |
| Comparison table (Deliverable 4) | `results/model_comparison_val.md` / `.csv`, plus `model_comparison_val_per_target.csv` |
| Test results (Deliverable 5) | `results/chosen_models_test.md` / `.csv` |
| Feature ablation | `results/feature_ablation_val.md` |
| Size curve | `results/data_size_curve.csv` |
| Logs | `logs/` |

```bash
# On Ada (GPU job): train → raw ablation → evaluate → size curve
sbatch run_all.sh                 # everything in one job
bash submit_by_model.sh           # or one chained job per family
```
