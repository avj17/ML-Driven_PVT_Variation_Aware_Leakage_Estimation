"""Step 1 (spec Step 6): collect the full-netlist CLA SPICE results written by ngspice.

The simulation itself is ngspice code: spice/cla3.cir (run it with run_part_c.py,
or by hand from the spice/ folder with `ngspice_con -b cla3.cir`). This script
only reads its output, checks it and puts it in one table.

Checks
    - cla3.cir's gate instances match cla/circuit.py (the hierarchical ML side),
      and gates.lib matches gatelib/gates.py
    - the gate network adds correctly (logic), and every SPICE output sits at
      its expected logic level
    - the corners are the same 200 as in Part B

    python step1_collect_spice_cla.py

Output
    results/spice_cla.csv   one row per (corner, vector): leakage, VDD current, outputs
"""
import numpy as np
import pandas as pd

from cla import config as CC
from cla.circuit import INPUTS, OUTPUTS, check_adder, check_netlist, n_transistors, simulate, vector_number
from gatelib import config as C
from gatelib.gates import GATES
from gatelib.spice_io import check_gate_library, check_same_corners, corners_from, read_output


def main():
    check_adder()
    check_netlist()
    check_gate_library(GATES, C.SPICE_DIR / "gates.lib")
    print(f"cla3.cir matches cla/circuit.py ({n_transistors()} transistors); gates.lib matches gatelib/gates.py")

    raw = read_output(CC.SPICE_OUT, ["corner", *INPUTS])
    if C.CORNERS_CSV.exists():
        check_same_corners(pd.read_csv(C.CORNERS_CSV), corners_from(raw), "CLA vs Part B")
        print("corners match Part B")
    else:
        print("WARNING: Part B corners.csv not found; corners not cross-checked")

    rows = []
    for r in raw.itertuples(index=False):
        vec = {n: getattr(r, n) for n in INPUTS}
        nets, _ = simulate(vec)
        v = vector_number(vec)
        ok = all(abs(getattr(r, f"v_{o}") - nets[o] * r.vdd) < 0.1 * r.vdd for o in OUTPUTS)
        rows.append({"corner": r.corner, "vec": v, "a": v & 7, "b": (v >> 3) & 7, "cin": v >> 6,
                     "inputs": "".join(str(vec[n]) for n in INPUTS),
                     "spice_ileak": r.ileak, "spice_ivdd": r.ivdd,
                     **{f"spice_v_{o}": getattr(r, f"v_{o}") for o in OUTPUTS}, "logic_ok": ok})
    df = pd.DataFrame(rows).sort_values(["corner", "vec"])
    CC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CC.SPICE_CSV, index=False)

    print(f"-> {CC.SPICE_CSV} ({len(df)} rows: {df['corner'].nunique()} corners x {df['vec'].nunique()} vectors, "
          f"{int((~df['logic_ok']).sum())} with a wrong output level)")
    print(f"SPICE leakage: median {np.nanmedian(df['spice_ileak']):.4g} A, "
          f"range {np.nanmin(df['spice_ileak']):.3g} .. {np.nanmax(df['spice_ileak']):.3g} A")


if __name__ == "__main__":
    main()
