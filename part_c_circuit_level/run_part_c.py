"""Run Part C end to end (needs Part B's results):
  1. ngspice: spice/cla3.cir          (full-netlist SPICE reference, ngspice code)
  2. step1_collect_spice_cla.py       check and collect the ngspice output
  3. step2_ml_cla.py                  hierarchical ML: CLA -> gate models -> transistor models
  4. step3_evaluate_cla.py            table and plots

    python run_part_c.py
    python run_part_c.py --skip-spice     # reuse results/spice_cla.txt from an earlier ngspice run

ngspice binary: NGSPICE env var, else ngspice_con, else ngspice.
For a quick check set `let nrun = 5` in spice/cla3.cir (and in the Part B netlists) first.
"""
import argparse
import subprocess
import sys
import time

from cla import config as CC
from gatelib.spice_io import run_netlist

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--skip-spice", action="store_true", help="do not rerun the ngspice netlist")
args = ap.parse_args()

if not args.skip_spice:
    print("===== ngspice =====", flush=True)
    CC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run_netlist(CC.SPICE_DIR / "cla3.cir")
    print(f"    {time.perf_counter() - t0:.0f} s", flush=True)

for step in ("step1_collect_spice_cla.py", "step2_ml_cla.py", "step3_evaluate_cla.py"):
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, str(CC.PART_C / step)], check=True, cwd=CC.PART_C)
