"""Run Part B end to end:
  1. ngspice: spice/{inv,nand2,nor2,and2,or2}.cir  (the SPICE reference, ngspice code)
  2. step1_collect_spice.py   check and collect the ngspice output
  3. step2_ml_gates.py        ML gate leakage from the transistor models
  4. step3_evaluate_gates.py  results table, summary and plots

    python run_part_b.py
    python run_part_b.py --skip-spice     # reuse results/spice_*.txt from an earlier ngspice run

ngspice binary: NGSPICE env var, else ngspice_con, else ngspice.
For a quick check set `let nrun = 5` in the five netlists first.
"""
import argparse
import subprocess
import sys
import time

from gatelib import config as C
from gatelib.gates import GATES
from gatelib.spice_io import run_netlist

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--skip-spice", action="store_true", help="do not rerun the ngspice netlists")
args = ap.parse_args()

if not args.skip_spice:
    print("===== ngspice =====", flush=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in GATES:
        t0 = time.perf_counter()
        run_netlist(C.SPICE_DIR / f"{name.lower()}.cir")
        print(f"    {time.perf_counter() - t0:.0f} s", flush=True)

for step in (["step1_collect_spice.py"], ["step2_ml_gates.py"], ["step3_evaluate_gates.py"]):
    print(f"\n===== {' '.join(step)} =====", flush=True)
    subprocess.run([sys.executable, str(C.PART_B / step[0]), *step[1:]], check=True, cwd=C.PART_B)
