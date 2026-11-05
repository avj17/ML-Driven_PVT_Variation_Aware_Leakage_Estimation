"""Part B configuration for the Python (ML and analysis) side: paths, gate sizing
and solver settings. The SPICE side lives in part_b_gate_level/spice/*.cir; the
200 corners are drawn inside those netlists (setseed 142).

Part C imports this module too.
"""
import os
import shutil
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]                 # repository root
PART_B = ROOT / "part_b_gate_level"
MODEL_TRAINING = ROOT / "model_training"                    # holds the leakage_ml package
SAVED_MODELS = MODEL_TRAINING / "saved_models"              # the 4 chosen Part A models
SPICE_DIR = PART_B / "spice"                                # gate netlists, gates.lib, model card
RESULTS_DIR = PART_B / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CORNERS_CSV = RESULTS_DIR / "corners.csv"
SPICE_CSV = RESULTS_DIR / "spice_gates.csv"
GATE_RESULTS_CSV = RESULTS_DIR / "gate_results.csv"


def spice_out(gate_name):
    """Raw output written by spice/<gate>.cir."""
    return RESULTS_DIR / f"spice_{gate_name.lower()}.txt"


# ngspice binary used by run_part_b.py / run_part_c.py. On Windows the console
# build (ngspice_con) is preferred.
NGSPICE = os.environ.get("NGSPICE") or shutil.which("ngspice_con") or shutil.which("ngspice") or "ngspice"

# ----------------------------------------------------------------------------
# Technology (team StaticSilicon: 45nm_HP, VDD 0.9 V); same values as Part A
# ----------------------------------------------------------------------------
L_MIN = 45e-9
VDD_NOM = 0.9
PROCESS_PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]
CORNER_COLS = ["corner", "vdd", "temp"] + [f"{m}_{p}" for m in ("n", "p") for p in PROCESS_PARAMS]

# ----------------------------------------------------------------------------
# Gate sizing, as multiples of L_MIN (inside the trained 1x..8x). Series stacks
# are doubled; PMOS are 2x NMOS for the lower hole mobility. AND2 = NAND2 + INV
# and OR2 = NOR2 + INV reuse these sizes. Must match spice/*.cir and
# spice/gates.lib; step1_collect_spice.py checks this.
# ----------------------------------------------------------------------------
WIDTHS = {
    "INV":   {"nmos": 2, "pmos": 4},
    "NAND2": {"nmos": 4, "pmos": 4},
    "NOR2":  {"nmos": 2, "pmos": 8},
}

# ----------------------------------------------------------------------------
# ML node-voltage solve (KCL on the transistor models): vectorised bisection
# per node over [0, VDD], repeated in Gauss-Seidel sweeps for coupled nodes.
# 30 halvings of ~1 V give ~1 nV resolution.
# ----------------------------------------------------------------------------
SOLVE_ITERS = 30
SOLVE_SWEEPS = 5
