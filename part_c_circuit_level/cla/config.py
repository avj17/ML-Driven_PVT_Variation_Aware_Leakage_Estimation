"""Part C paths. Importing this puts part_b_gate_level on sys.path, so `gatelib`
(gate definitions, gate model, transistor models, metrics) is shared with Part B."""
import sys
from pathlib import Path

PART_C = Path(__file__).resolve().parents[1]
ROOT = PART_C.parent
sys.path.insert(0, str(ROOT / "part_b_gate_level"))

SPICE_DIR = PART_C / "spice"                       # cla3.cir (ngspice)
RESULTS_DIR = PART_C / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SPICE_OUT = RESULTS_DIR / "spice_cla.txt"          # written by spice/cla3.cir
SPICE_CSV = RESULTS_DIR / "spice_cla.csv"
CLA_RESULTS_CSV = RESULTS_DIR / "cla_results.csv"
