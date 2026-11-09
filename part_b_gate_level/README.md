# Part B: gate-level validation

This part predicts the leakage of INV, NAND2, NOR2, AND2 and OR2 for every input combination at 200 random PVT corners, and checks each prediction against SPICE. There are 18 gate/input cases × 200 corners = 3,600 runs.

- **SPICE side:** ngspice netlists in `spice/`. Their `.control` scripts draw the corners, sweep the inputs and compute the leakage.
- **ML side:** Python. It uses only the four Part A transistor models.

## How to run

You need:
- ngspice (`ngspice_con` on Windows; set `NGSPICE=<path>` to override);
- the Part A packages from `model_training/requirements.txt` (the CPU builds of `torch` and `xgboost` are enough).

```bash
cd part_b_gate_level
python run_part_b.py            # ngspice on the 5 netlists, then steps 1-3
```

Or run each piece by hand:

```bash
cd part_b_gate_level/spice      # the netlists use paths relative to this folder
ngspice_con -b inv.cir          # same for nand2.cir nor2.cir and2.cir or2.cir
cd ..
python step1_collect_spice.py   # check + collect the ngspice output
python step2_ml_gates.py        # spec Step 4: ML leakage via the gate model
python step3_evaluate_gates.py  # spec Step 5: results table, summary and plots
```

After new Part A models, `python run_part_b.py --skip-spice` redoes only the ML steps.

**Quick check:** set `let nrun = 5` in the five netlists (and in `cla3.cir` for Part C). Only the first 5 corners are simulated; the corners themselves stay the same.

## Current results

These results use the **v3 Part A models** (`model_training/v3/saved_models/`: 5-network MLP ensembles trained on transistors with junction geometry and 37,000 training rows per case). The gate netlists carry the same junction geometry (AD = AS = W·90 nm, PD = PS = 2(W + 90 nm)). There are 3,600 runs: 18 gate/input cases × 200 corners, all with the correct output logic level in SPICE. The ML uses the node voltages recorded by SPICE (spec Step 4).

Leakage is total leakage (supply power / VDD). Mean accuracy = 100 % − mean |relative error|. Bias = mean signed relative error. The full tables are in `results/gate_results_table.md`.

| | Median error | p95 error | **Mean accuracy** |
|---|---|---|---|
| **All 3,600 runs (v3)** | **0.26 %** | **1.20 %** | **99.61 %** |
| Earlier, v2 models | 2.75 % | 15.6 % | 95.3 % |
| Earlier, v1 models | 5.60 % | 17.0 % | 93.3 % |

**All 18 cases are above the spec's 99 % mean-accuracy target** (lowest: NOR2 01, 99.46 %).

**Per gate and input case** (Deliverable 6):

| Gate | Inputs | SPICE median (nA) | ML median (nA) | Bias | Median error | p95 error | Mean accuracy |
|---|---|---|---|---|---|---|---|
| INV | 0 | 1.828 | 1.833 | +0.34 % | 0.36 % | 1.29 % | 99.54 % |
| INV | 1 | 0.886 | 0.878 | −0.28 % | 0.30 % | 1.52 % | 99.49 % |
| NAND2 | 00 | 0.691 | 0.692 | +0.13 % | 0.24 % | 1.01 % | 99.63 % |
| NAND2 | 01 | 3.687 | 3.687 | +0.24 % | 0.27 % | 1.26 % | 99.59 % |
| NAND2 | 10 | 1.840 | 1.830 | −0.05 % | 0.18 % | 0.70 % | 99.76 % |
| NAND2 | 11 | 1.807 | 1.809 | −0.23 % | 0.29 % | 1.51 % | 99.51 % |
| NOR2 | 00 | 4.383 | 4.394 | +0.21 % | 0.31 % | 1.29 % | 99.55 % |
| NOR2 | 01 | 2.505 | 2.494 | −0.18 % | 0.40 % | 1.46 % | 99.46 % |
| NOR2 | 10 | 0.688 | 0.690 | −0.04 % | 0.15 % | 0.53 % | 99.79 % |
| NOR2 | 11 | 0.348 | 0.349 | +0.06 % | 0.12 % | 1.09 % | 99.72 % |
| AND2 | 00 | 1.902 | 1.906 | −0.09 % | 0.28 % | 1.02 % | 99.63 % |
| AND2 | 01 | 4.947 | 4.969 | +0.11 % | 0.25 % | 0.94 % | 99.64 % |
| AND2 | 10 | 3.184 | 3.171 | −0.16 % | 0.25 % | 0.79 % | 99.69 % |
| AND2 | 11 | 4.529 | 4.520 | +0.05 % | 0.32 % | 1.05 % | 99.59 % |
| OR2 | 00 | 5.628 | 5.642 | +0.11 % | 0.31 % | 1.15 % | 99.59 % |
| OR2 | 01 | 5.102 | 5.081 | +0.04 % | 0.35 % | 1.18 % | 99.55 % |
| OR2 | 10 | 2.858 | 2.873 | +0.23 % | 0.22 % | 1.07 % | 99.66 % |
| OR2 | 11 | 2.289 | 2.294 | +0.32 % | 0.26 % | 1.18 % | 99.60 % |

**Per gate:**

| Gate | Median error | p95 error | Mean accuracy |
|---|---|---|---|
| INV | 0.33 % | 1.46 % | 99.52 % |
| NAND2 | 0.24 % | 1.28 % | 99.62 % |
| NOR2 | 0.24 % | 1.22 % | 99.63 % |
| AND2 | 0.27 % | 0.97 % | 99.64 % |
| OR2 | 0.28 % | 1.14 % | 99.60 % |
| **All** | **0.26 %** | **1.20 %** | **99.61 %** |

**Observations:**
- **The ML follows the input dependence of the leakage.** The SPICE leakage varies about 16× between cases, and the ML median is within 0.8 % of the SPICE median in every case. The lowest leakage is in the stacked cases, NOR2 11 (two series PMOS OFF, 0.35 nA) and NAND2 00 (two series NMOS OFF, 0.69 nA): the stack effect. With only one series device OFF (NAND2 01) the leakage is about 5× higher.
- **No systematic offset:** the bias is within ±0.34 % in every case and +0.05 % overall.
- **Most and least accurate:** the best cases (NOR2 10, NAND2 10, NOR2 11: 99.7–99.8 %) are dominated by OFF-transistor current. The lowest (NOR2 01, INV 1, NAND2 11: 99.46–99.51 %) still clear the target.
- **Gate size does not add error:** AND2 and OR2 (6 transistors) are as accurate as INV (2 transistors); the transistors' errors partly cancel in the sum.
- **Largest single errors** (max 3.3 %; 21 of 3,600 runs above 2 %) come from a few corners, the same ones across gates, with low vth0 (about −9 to −10 %) at a temperature extreme (−24 °C and 109 °C). The mean error is about 0.45 % below 0 °C and above 80 °C, against 0.29 % at 40–80 °C.
- **ML-solved node voltages** (`results/node_voltages.csv`) match SPICE to within 0.82 mV in every run (median difference below 0.001 mV).

**What changed from v2 to v3, and the small-Vds fix:**
- The v3 models improve every transistor metric over v2 (test mean error 0.13–0.29 % against 1.5–3.0 %).
- With v3 the gate results first came out *worse* in mean (4.97 % mean error, NAND2 00 at 27.8 %), though better in median. The cause was **not** the models' accuracy in their training range:
  - An ON transistor in a gate sits at |Vds| ≈ 1 µV, or exactly 0 because the netlists write node voltages with 6 digits.
  - The Part A feature `log_drain_factor = log(1 − exp(−|Vds|/Vt))` then falls to −10 … −27.6, far below its training minimum of about −8.4, and the MLP extrapolates. The ON PMOS gate current in NAND2 00 came out up to 27× too low at cold corners. This was found by simulating each transistor alone in ngspice at the gate's node voltages; the OFF NMOS matched to about 0.1 %.
- **Fix** (`gatelib/ml_gate.py`, `config.VDS_FLOOR = 0.5 mV`): where |Vds| < 0.5 mV, the model is evaluated at |Vds| = 0.5 mV. Ig and Ib are kept (0.5 mV moves Vgd by about 0.05 %), and the channel current is scaled linearly back to the true Vds before Id and Is are rebuilt, so KCL still holds. 0.5 mV was chosen from the training-data coverage (|Vds| 1st percentile about 5 mV), not tuned on the gate results; 0.1 mV and 2 mV give 99.49 % and 99.65 % mean accuracy. No retraining and no change to the SPICE side.
- A permanent fix in Part A would be a bounded feature (e.g. 1 − exp(−|Vds|/Vt) instead of its log) or training data at |Vds| ≈ 0.

## What each file does and why it is needed

```
part_b_gate_level/
├── README.md
├── run_part_b.py
├── step1_collect_spice.py
├── step2_ml_gates.py
├── step3_evaluate_gates.py
├── spice/                      SPICE side (ngspice code)
│   ├── inv.cir  nand2.cir  nor2.cir  and2.cir  or2.cir
│   ├── gates.lib
│   └── models/45nm_HP_param.pm
├── gatelib/                    Python side (ML and analysis), also used by Part C
│   ├── __init__.py
│   ├── config.py
│   ├── gates.py
│   ├── spice_io.py
│   ├── ml_gate.py
│   └── report.py
└── results/                    everything the steps write (created on the first run)
```

### SPICE side: `spice/`

**`inv.cir`, `nand2.cir`, `nor2.cir`, `and2.cir`, `or2.cir`**
- **What:** one ngspice netlist per gate, with the transistors written out at transistor level. Each file's `.control` script does four things:
  1. draws the 200 PVT corners;
  2. applies each corner to the model card, the temperature and VDD;
  3. sets every input combination at exactly 0 or VDD;
  4. runs an operating-point analysis and writes the corner, inputs, node voltages and leakage to `results/spice_<gate>.txt`.
- **Why:** these produce the SPICE reference that the ML prediction is checked against (spec Step 3), and they are the "one SPICE netlist per gate" of Deliverable 2. AND2 and OR2 are simulated as whole gates, as the spec requires, not as a NAND2/NOR2 result plus an INV result.

**`gates.lib`**
- **What:** the same five gates written as `.subckt` blocks, with the same transistors and widths.
- **Why:** Part C builds the 3-bit CLA by instantiating these subcircuits, so the CLA is made of exactly the gates validated here.

**`models/45nm_HP_param.pm`**
- **What:** a copy of the assigned `45nm_HP.pm` in which vth0, toxe, ndep, u0 and xj are `.param`s (`n_vth0`, `p_vth0`, …). Their defaults are the nominal values, and every other model value is unchanged.
- **Why:** a netlist can only move to a new process corner if these values can be changed from inside ngspice. With parameters, `alterparam` + `reset` can set separate NMOS and PMOS values for each corner, as the spec requires.

### Python side: the step scripts

**`run_part_b.py`**
- **What:** the one command that runs Part B end to end. It runs the five netlists with ngspice, then steps 1–3 in order. `--skip-spice` reuses the ngspice output from an earlier run.
- **Why:** it puts the whole part in a fixed order and makes it reproducible with one command (Deliverable 9).

**`step1_collect_spice.py`**
- **What:** reads the five `results/spice_<gate>.txt` files, then:
  - checks that every netlist (and `gates.lib`) contains exactly the transistors the ML side assumes (`gatelib/gates.py`);
  - checks that all five netlists drew the same corners, and that every output voltage sits at its logic level;
  - writes `results/corners.csv` and `results/spice_gates.csv`.
- **Why:**
  - The ML prediction is only a fair comparison if it models the same circuit SPICE simulated; a width or wiring mismatch would show up as "ML error".
  - It turns the raw ngspice text into one clean table for the later steps.

**`step2_ml_gates.py`**
- **What:** for every gate, input combination and corner:
  - predicts the gate leakage with the ML gate model, using the node voltages recorded by SPICE (`ml_ileak`, spec Step 4);
  - solves the gate's internal node voltages from the transistor models alone (`ml_v_*`).

  It writes `results/gate_results.csv`, which has the SPICE and ML values side by side for every run.
- **Why:** this is the ML prediction the spec asks for, and spec Section 5 asks to record "the internal node voltages from both" SPICE and ML. `--no-solve` skips the node-voltage solve.

**`step3_evaluate_gates.py`**
- **What:** compares ML with SPICE and writes:
  - the gate results table for the 18 cases (SPICE median, ML median, bias, median error, p95 error, mean accuracy), with per-gate and overall rows, and a short summary;
  - the SPICE vs ML node-voltage comparison;
  - the 15 largest errors;
  - the parity plot and the error-vs-corner plot.
- **Why:** it produces Deliverables 6 and 7 and the material for the report's gate results and error analysis sections.

### Python side: the `gatelib/` library

**`config.py`**
- **What:** paths (including which Part A models are used: `model_training/v3/saved_models`), the gate widths (`WIDTHS`), the junction geometry (`LDIFF`), the corner column names, the ngspice binary, the settings for the ML node solve, and the small-Vds floor (`VDS_FLOOR`).
- **Why:** each setting is defined in one place, so the scripts cannot disagree. The widths here must equal the widths in the netlists, and step 1 checks this.

**`gates.py`**
- **What:** the five gates as Python data: each transistor's type, drain/gate/source/body nets and width, the gate's internal nodes, and its logic function.
- **Why:** the ML side needs to know every transistor's terminals to give the transistor models the right voltages. The logic function decides whether each transistor is ON or OFF, and therefore which of the four models to use.

**`spice_io.py`**
- **What:**
  - runs a netlist with ngspice from its own folder and saves a `.log` file;
  - parses the netlists' output files;
  - compares the transistors in the `.cir` files and `gates.lib` with `gates.py`;
  - checks that two sets of corners are identical.
- **Why:** it is the bridge between the ngspice side and the Python side, and the consistency checks it provides guard against the two drifting apart.

**`ml_gate.py`**
- **What:** the ML gate model, built only from the four Part A transistor models.
  - `TransistorModels` loads the saved models.
  - `GateModel` gives each transistor its terminal voltages, picks the ON or OFF model, predicts its four terminal currents (evaluated at |Vds| ≥ `VDS_FLOOR`, see above), and adds them up into the gate leakage.
  - `solve_nodes` finds the gate's node voltages from KCL on the models alone.
- **Why:**
  - This is the core of Part B: gate leakage from transistor models.
  - It is also the middle level of the Part C hierarchy (circuit → gate → transistor).

**`report.py`**
- **What:** the metrics and the plots. The metrics are SPICE median, ML median, bias, median and p95 error, and mean accuracy (mean and max error are also computed but not shown in the Part B table). The plots are the log-axis parity plot and error vs corner parameters.
- **Why:** Part B and Part C use the same metric and plot code, so the gate-level and circuit-level numbers in the report are directly comparable.

**`__init__.py`**
- **What:** marks `gatelib` as a Python package.
- **Why:** so the scripts here, and Part C, can `import gatelib`.

### Outputs: `results/`

| File | What it is | Why it is needed |
|---|---|---|
| `spice_<gate>.txt` | raw ngspice output, one line per (corner, inputs) | the SPICE reference as simulated |
| `corners.csv` | the 200 corners as drawn by ngspice | the ML needs each corner's VDD, temperature and process values; Part C checks its corners against it |
| `spice_gates.csv` | all SPICE results in one table | input to step 2, and to Part C (pre-recorded gate node voltages, SPICE gate sum) |
| `gate_results.csv` | every run with SPICE and ML values and corner parameters | the full record behind every table and plot, and the data for error analysis |
| `gate_results_table.md` / `.csv` | SPICE median, ML median, bias, median and p95 error, and mean accuracy for the 18 cases, per gate and overall | **Deliverable 6** |
| `node_voltages.csv` | SPICE vs ML-solved internal node voltages | the spec asks for node voltages from both |
| `plots/gate_parity.png` | ML vs SPICE leakage for all runs, log axes | **Deliverable 7** |
| `plots/gate_error_vs_corner.png` | error vs temperature, VDD, vth0 and toxe | for the report's error-analysis section |
| `spice/*.log` | ngspice console output of each netlist (written next to the netlist) | for debugging a failed run |

## How the netlists work

Each netlist has one `.control` script that does the following.

1. **Draws the 200 corners** with `setseed 142` and `sunif()`, following spec Section 4:
   - VDD: 0.9 V ± 10 %.
   - Temperature: −40 to 125 °C.
   - Process: vth0, toxe, ndep, u0 and xj, each nominal ± 10 %. All NMOS share one draw and all PMOS share another.

   This block is byte-identical in all five gate netlists (and in Part C's CLA netlist), so every gate sees the same corners. Step 1 checks this.
2. **Applies each corner:**
   - process: `alterparam n_vth0=…` etc. on the parameterised model card, then `reset`;
   - temperature: `option temp=…`;
   - supply: `alter vdd dc = …`.
3. **Sweeps every input combination** at exactly 0 or VDD with `alter va dc = …`, runs `op`, and writes the corner, the inputs, the node voltages and the leakage with `echo … >> ../results/spice_<gate>.txt`.

All netlists use the required `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`. AND2 and OR2 are simulated as complete gates. Every transistor carries the junction geometry of the v3 Part A data (AD = AS = W·90 nm, PD = PS = 2(W + 90 nm)); step 1 checks it.

**Sizing** is given as W in multiples of L = 45 nm, inside the trained range of 1–8× L:

| Gate | NMOS | PMOS | Nodes recorded |
|---|---|---|---|
| INV | 2 | 4 | out |
| NAND2 | 4 (series) | 4 | out, x |
| NOR2 | 2 | 8 (series) | out, y |
| AND2 | NAND2 + INV | | out, x, n1 |
| OR2 | NOR2 + INV | | out, y, n1 |

The ML side reads the same topology from `gatelib/gates.py`. Step 1 checks that every netlist and `gates.lib` match it transistor for transistor.

## Leakage definition

Leakage is the static supply power divided by VDD.

- **SPICE** (computed in the netlist): `ileak = −I(VDD) − Σ bit·I(V_in)`, i.e. the VDD current plus the current of each input held at VDD.
- **ML** (`gatelib/ml_gate.py`): `(1/VDD) · Σ over transistors of (Vds·Id + Vgs·Ig + Vbs·Ib)`, using the predicted terminal currents.

By Tellegen's theorem the two are the same physical quantity. The ML form is used because an ON transistor sits at |Vds| ≈ µV, where the transistor models are least accurate, and here its channel current is multiplied by that tiny Vds. Adding up the currents at the VDD pins would bring that error straight into the leakage.

The definition also adds up across gates, which the hierarchical Part C relies on. The netlists also record the VDD-only current (`spice_ivdd` in `spice_gates.csv`); it is not used in the results.

## ML gate model (spec Step 4)

`gatelib.ml_gate.GateModel` is the middle level of the hierarchy that Part C uses. For each transistor of the gate it takes:
- the terminal voltages from the gate's net voltages (rails; inputs at exactly 0 or VDD; internal and output nodes);
- the corner's VDD, temperature and that device type's process draw;
- the ON or OFF model of its type, chosen by the logic value at its gate.

Where a transistor's |Vds| is below `VDS_FLOOR` (0.5 mV), the model is evaluated at |Vds| = 0.5 mV and the channel current is scaled back to the true Vds (see "What changed from v2 to v3"). It then computes the leakage as above. Node voltages come from one of two sources:
- `"spice"` (`ml_ileak`): pre-recorded by the gate-level ngspice runs, as spec Step 4 says. **All reported results use this.**
- `"ml"` (`ml_v_*`): solved by KCL on the transistor models, using vectorised bisection in Gauss–Seidel sweeps. These are the ML node voltages that spec Section 5 asks to record.

**Bias** is the mean signed relative error (ML − SPICE)/SPICE. **Mean accuracy** is 100 % minus the mean |relative error|.
