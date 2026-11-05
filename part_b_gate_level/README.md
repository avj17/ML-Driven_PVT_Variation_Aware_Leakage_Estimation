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
python step3_evaluate_gates.py  # spec Step 5: tables and plots
python step3_evaluate_gates.py --metric vdd   # scored on the VDD-source current only
```

**Quick check:** set `let nrun = 5` in the five netlists (and in `cla3.cir` for Part C). Only the first 5 corners are simulated; the corners themselves stay the same.

## Current results

These results use the **v2 Part A models**: the 5-network MLP ensemble with KCL outputs, from commit `7d3227b`. There are 3,600 runs: 18 gate/input cases × 200 corners, all with the correct output logic level in SPICE. The ML uses the node voltages recorded by SPICE (spec Step 4).

Leakage is total leakage (supply power / VDD). Mean accuracy = 100 % − mean |relative error|. Bias = mean signed relative error. The full tables are in `results/gate_results_table.md`.

| | Median error | p95 error | **Mean accuracy** |
|---|---|---|---|
| **All 3,600 runs** | **2.75 %** | **15.6 %** | **95.3 %** |
| With the earlier v1 models | 5.60 % | 17.0 % | 93.3 % |

The spec's target is "up to 99 % mean accuracy".

**Per gate and input case** (Deliverable 6):

| Gate | Inputs | SPICE median (nA) | ML median (nA) | Bias | Median error | p95 error | Mean accuracy |
|---|---|---|---|---|---|---|---|
| INV | 0 | 1.828 | 1.874 | +2.02 % | 2.01 % | 8.99 % | 96.94 % |
| INV | 1 | 0.885 | 0.819 | −4.98 % | 5.00 % | 16.46 % | 93.73 % |
| NAND2 | 00 | 0.690 | 0.774 | +16.56 % | 14.27 % | 38.35 % | 83.38 % |
| NAND2 | 01 | 3.687 | 3.664 | +0.37 % | 1.67 % | 6.59 % | 97.80 % |
| NAND2 | 10 | 1.840 | 1.886 | +2.77 % | 1.67 % | 9.83 % | 96.78 % |
| NAND2 | 11 | 1.807 | 1.799 | −4.59 % | 4.78 % | 16.20 % | 93.88 % |
| NOR2 | 00 | 4.383 | 4.586 | +4.05 % | 2.86 % | 13.55 % | 95.32 % |
| NOR2 | 01 | 2.505 | 2.522 | −0.81 % | 5.66 % | 14.39 % | 93.51 % |
| NOR2 | 10 | 0.688 | 0.687 | −0.21 % | 1.98 % | 7.86 % | 97.11 % |
| NOR2 | 11 | 0.348 | 0.369 | +7.41 % | 4.70 % | 20.63 % | 92.59 % |
| AND2 | 00 | 1.902 | 1.907 | +3.03 % | 4.00 % | 18.52 % | 94.62 % |
| AND2 | 01 | 4.947 | 4.872 | −1.59 % | 1.93 % | 6.98 % | 97.43 % |
| AND2 | 10 | 3.184 | 3.087 | −1.12 % | 2.03 % | 7.63 % | 97.36 % |
| AND2 | 11 | 4.529 | 4.373 | −2.32 % | 2.57 % | 10.39 % | 96.40 % |
| OR2 | 00 | 5.628 | 5.690 | +1.40 % | 2.00 % | 8.99 % | 97.04 % |
| OR2 | 01 | 5.102 | 4.816 | −0.28 % | 3.28 % | 10.61 % | 95.92 % |
| OR2 | 10 | 2.858 | 2.888 | +0.78 % | 1.43 % | 5.76 % | 97.98 % |
| OR2 | 11 | 2.289 | 2.333 | +2.42 % | 2.23 % | 7.70 % | 97.09 % |

**Per gate:**

| Gate | Median error | p95 error | Mean accuracy |
|---|---|---|---|
| INV | 2.98 % | 14.85 % | 95.34 % |
| NAND2 | 3.46 % | 25.46 % | 92.96 % |
| NOR2 | 3.41 % | 15.66 % | 94.63 % |
| AND2 | 2.42 % | 10.47 % | 96.45 % |
| OR2 | 2.06 % | 8.61 % | 97.01 % |

**Observations:**
- **Most and least accurate:**
  - 12 of the 18 cases have a median error of about 2 % or less.
  - **NAND2 AB=00** (83.4 %, +16.6 % bias) is the clear outlier. Both NMOS are OFF and in series, with the node between them at about 82 mV. The models over-predict the current through the upper transistor, whose source is raised slightly above ground (a slight reverse body bias). That operating point is rare in the uniformly sampled Part A training data.
  - **NOR2 AB=11** (92.6 %, +7.4 %) is the PMOS counterpart of the same situation.
- **The SPICE setup is not a source of error.** A check simulated 20 Part A dataset transistors per case the Part B way: parameterised model card, `alterparam` + `reset`, `option temp` and `alter`. Id and Is agreed with the dataset to within 0.0005 %. Every other difference was below 1e-15 A, which is simulator noise. The remaining error is therefore the ML models' own error.
- **ML-solved node voltages** (no SPICE input at all) give almost the same accuracy: 2.86 % median error and 95.2 % mean accuracy. So the node voltages are not the limiting factor; the transistor current predictions are.
- **The models are less accurate inside gates than on their own test set.** On the uniform Part A test set the median error is 0.40–0.68 %; inside gates it is 2.75 %. Gates operate their transistors at rail and stack-node voltages, which uniform sampling of Vd and Vs rarely produces. More training data, in particular in those regions, is the planned improvement.
- **VDD-only metric** (`--metric vdd`, `results/gate_results_table_vdd.md`): 94.3 % mean accuracy. This metric is ill-conditioned when inputs are held high: the VDD current is then the small difference of larger currents. For example, NOR2 AB=11 has only 0.04 nA of VDD current. That is why the total leakage is the main metric.

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
- **What:** for every gate, input combination and corner, predicts the gate leakage with the ML gate model (spec Step 4). It does this twice:
  - with the node voltages recorded by SPICE (`ml_ileak`);
  - with node voltages the ML solves for itself (`ml_ileak_mlnodes`, `ml_v_*`).

  It writes `results/gate_results.csv`, which has the SPICE and ML values side by side for every run.
- **Why:** this is the ML prediction the spec asks for. The second variant also gives the "internal node voltages from both" that spec Section 5 asks to record.

**`step3_evaluate_gates.py`**
- **What:** compares ML with SPICE and writes:
  - the gate results table for the 18 cases, with per-gate and overall rows;
  - the node-voltage comparison;
  - the 15 largest errors;
  - the parity plot and the error-vs-corner plots.

  `--metric vdd` scores on the VDD-source current instead of the total leakage.
- **Why:** it produces Deliverables 6 and 7 and the material for the report's gate results and error analysis sections.

### Python side: the `gatelib/` library

**`config.py`**
- **What:** paths, the gate widths (`WIDTHS`), the corner column names, the ngspice binary, and the settings for the ML node solve.
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
  - `GateModel` gives each transistor its terminal voltages, picks the ON or OFF model, predicts its four terminal currents, and adds them up into the gate leakage.
  - `solve_nodes` finds the gate's node voltages from KCL on the models alone.
- **Why:**
  - This is the core of Part B: gate leakage from transistor models.
  - It is also the middle level of the Part C hierarchy (circuit → gate → transistor).

**`report.py`**
- **What:** the metrics and the plots. The metrics are SPICE median, ML median, bias, median, p95, mean and max error, and mean accuracy. The plots are the log-axis parity plot and error vs corner parameters.
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
| `gate_results_table.md` / `.csv` | SPICE median, ML median, bias, median and p95 error for the 18 cases | **Deliverable 6** |
| `gate_results_table_mlnodes.csv` | the same with ML-solved node voltages | shows the accuracy with no SPICE input at all |
| `*_vdd` files | the same tables scored on the VDD-source current | an alternative leakage definition, for comparison |
| `node_voltages.csv` | SPICE vs ML-solved internal node voltages | the spec asks for node voltages from both |
| `plots/gate_parity.png` | ML vs SPICE leakage for all runs, log axes | **Deliverable 7** |
| `plots/gate_parity_mlnodes.png` | the same with ML-solved node voltages | comparison |
| `plots/gate_error_vs_corner.png` | error vs temperature, VDD, vth0 and toxe | for the report's error-analysis section |
| `spice/*.log` | ngspice console output of each netlist (written next to the netlist) | for debugging a failed run |

## How the netlists work

Each netlist has one `.control` script that does the following.

1. **Draws the 200 corners** with `setseed 142` and `sunif()`, following spec Section 4:
   - VDD: 0.9 V ± 10 %.
   - Temperature: −40 to 125 °C.
   - Process: vth0, toxe, ndep, u0 and xj, each nominal ± 10 %. All NMOS share one draw and all PMOS share another.

   This block is byte-identical in all six netlists (the five gates and the CLA), so every gate and the CLA see the same corners. Step 1 checks this.
2. **Applies each corner:**
   - process: `alterparam n_vth0=…` etc. on the parameterised model card, then `reset`;
   - temperature: `option temp=…`;
   - supply: `alter vdd dc = …`.
3. **Sweeps every input combination** at exactly 0 or VDD with `alter va dc = …`, runs `op`, and writes the corner, the inputs, the node voltages and the leakage with `echo … >> ../results/spice_<gate>.txt`.

All netlists use the required `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`. AND2 and OR2 are simulated as complete gates.

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

The definition also adds up across gates, which the hierarchical Part C relies on. The VDD-only current is recorded as well (`ivdd`, `--metric vdd`).

## ML gate model (spec Step 4)

`gatelib.ml_gate.GateModel` is the middle level of the hierarchy that Part C uses. For each transistor of the gate it takes:
- the terminal voltages from the gate's net voltages (rails; inputs at exactly 0 or VDD; internal and output nodes);
- the corner's VDD, temperature and that device type's process draw;
- the ON or OFF model of its type, chosen by the logic value at its gate.

It then computes the leakage as above. Node voltages come from one of two sources:
- `"spice"` (`ml_ileak`): pre-recorded by the gate-level ngspice runs, as spec Step 4 says.
- `"ml"` (`ml_ileak_mlnodes`, `ml_v_*`): solved by KCL on the transistor models, using vectorised bisection in Gauss–Seidel sweeps. These are the ML node voltages that spec Section 5 asks to record, and this variant needs no SPICE input at all.

**Bias** is the mean signed relative error (ML − SPICE)/SPICE. **Mean accuracy** is 100 % minus the mean |relative error|.
