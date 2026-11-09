"""Step 2 (spec Step 4): ML leakage for the same gates, inputs and corners as the SPICE runs.

Uses only the four Part A transistor models (model_training/saved_models),
through the gate model gatelib.ml_gate.GateModel:
  ml_ileak   ML leakage, with the node voltages pre-recorded by the gate-level
             SPICE runs (spec Step 4)
  ml_v_*     ML internal node voltages, solved by KCL on the transistor models
             (spec Section 5 asks to record the node voltages from both SPICE and ML)

    python step2_ml_gates.py
    python step2_ml_gates.py --no-solve     # skip the ML node-voltage solve (faster)

Output
    results/gate_results.csv   one row per (gate, inputs, corner): corner
                               parameters, SPICE and ML node voltages and leakage
"""
import argparse
import time

import numpy as np
import pandas as pd

from gatelib import config as C
from gatelib.gates import ALL_NODES, GATES, bits_str
from gatelib.ml_gate import GateModel, TransistorModels
from gatelib.report import leak_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-solve", action="store_true", help="skip the ML node-voltage solve")
    args = ap.parse_args()

    if not C.SPICE_CSV.exists():
        raise SystemExit(f"{C.SPICE_CSV} not found; run step1_collect_spice.py first.")
    spice = pd.read_csv(C.SPICE_CSV, dtype={"inputs": str})
    corners = pd.read_csv(C.CORNERS_CSV).set_index("corner")
    transistors = TransistorModels()
    gate_spice_nodes = GateModel("spice", transistors, spice)
    gate_ml_nodes = GateModel("ml", transistors)

    parts = []
    for gate in GATES.values():
        for bits in gate.combos():
            s = spice[(spice["gate"] == gate.name) & (spice["inputs"] == bits_str(bits))]
            s = s.dropna(subset=["spice_ileak"]).sort_values("corner").reset_index(drop=True)
            if s.empty:
                continue
            cs = corners.loc[s["corner"]].reset_index()
            t0 = time.perf_counter()

            s["ml_ileak"], _, _ = gate_spice_nodes.predict(gate.name, bits, cs)
            m = leak_metrics(s["spice_ileak"], s["ml_ileak"])
            msg = f"median err {m['median_err_pct']:.2f} %, p95 {m['p95_err_pct']:.2f} %"

            if not args.no_solve:
                _, _, V = gate_ml_nodes.predict(gate.name, bits, cs)
                for n in ALL_NODES:
                    s[f"ml_v_{n}"] = V[n] if n in gate.nodes else np.nan
            print(f"{gate.name:5s} {bits_str(bits):2s} ({len(s)} corners, {time.perf_counter() - t0:.1f} s)  {msg}",
                  flush=True)
            parts.append(s)

    df = pd.concat(parts, ignore_index=True).merge(corners.reset_index(), on="corner", how="left")
    df.to_csv(C.GATE_RESULTS_CSV, index=False)
    print(f"-> {C.GATE_RESULTS_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
