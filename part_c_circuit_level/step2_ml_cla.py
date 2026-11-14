"""Step 2 (spec Step 7): hierarchical ML leakage of the CLA.

    CLA model (cla/hierarchy.py)  ->  Part B gate model  ->  Part A transistor models

For each input vector the gate network is logic-simulated to find the input
combination at every gate; the Part B gate model of that gate type is called at
that combination and corner (it evaluates the gate's transistors with the four
transistor models, using the gate node voltages pre-recorded by Part B's
gate-level SPICE, spec Step 4), and the gate leakages are summed. No
circuit-level SPICE data is used.

Columns added to the SPICE table:
  ml_ileak           the hierarchical ML leakage (spec Step 7)
  spice_gatesum      sum of the gate-level SPICE leakages of Part B, for the error
                     analysis: it separates the error of composing gates (loading,
                     inputs not exactly at 0 or VDD) from the ML error
  gate_<TYPE>_share  fraction of ml_ileak contributed by each gate type

    python step2_ml_cla.py
"""
import time

import pandas as pd

from cla import config as CC
from cla.circuit import N_VECTORS, simulate, vector
from cla.hierarchy import CLAModel
from gatelib import config as C
from gatelib.gates import bits_str
from gatelib.ml_gate import GateModel, TransistorModels
from gatelib.report import leak_metrics


def main():
    for p, step in ((CC.SPICE_CSV, "step1_collect_spice_cla.py"),
                    (C.SPICE_CSV, "part_b_gate_level/step1_collect_spice.py")):
        if not p.exists():
            raise SystemExit(f"{p} not found; run {step} first.")

    spice = pd.read_csv(CC.SPICE_CSV, dtype={"inputs": str})
    spice_gates = pd.read_csv(C.SPICE_CSV, dtype={"inputs": str})
    corners = pd.read_csv(C.CORNERS_CSV)
    corners = corners[corners["corner"].isin(spice["corner"].unique())].reset_index(drop=True)

    model = CLAModel(GateModel("spice", TransistorModels(), spice_gates))
    gate_spice = {k: g.set_index("corner")["spice_ileak"] for k, g in spice_gates.groupby(["gate", "inputs"])}

    rows = []
    t0 = time.perf_counter()
    for v in range(N_VECTORS):
        out = {"corner": corners["corner"].to_numpy(), "vec": v}
        out["ml_ileak"], by_type = model.predict(v, corners)
        for t, leak in by_type.items():
            out[f"gate_{t}_share"] = leak / out["ml_ileak"]
        _, seen = simulate(vector(v))
        out["spice_gatesum"] = sum(gate_spice[(i.gate, bits_str(b))].reindex(corners["corner"]).to_numpy()
                                   for i, b in seen)
        rows.append(pd.DataFrame(out))
        if (v + 1) % 16 == 0:
            print(f"  {v + 1}/{N_VECTORS} vectors ({time.perf_counter() - t0:.0f} s)", flush=True)

    ml = pd.concat(rows, ignore_index=True)
    df = spice.merge(ml, on=["corner", "vec"], how="left").merge(corners, on="corner", how="left")
    df.to_csv(CC.CLA_RESULTS_CSV, index=False)
    print(f"-> {CC.CLA_RESULTS_CSV} ({len(df)} rows)")
    m = leak_metrics(df["spice_ileak"], df["ml_ileak"])
    print(f"  ML vs SPICE: median err {m['median_err_pct']:.2f} %, p95 {m['p95_err_pct']:.2f} %, "
          f"mean accuracy {m['mean_accuracy_pct']:.2f} %")


if __name__ == "__main__":
    main()
