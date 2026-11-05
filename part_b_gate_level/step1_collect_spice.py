"""Step 1 (spec Step 3): collect the gate-level SPICE results written by the ngspice netlists.

The simulations themselves are ngspice code: spice/{inv,nand2,nor2,and2,or2}.cir
(run them with run_part_b.py, or by hand from the spice/ folder with
`ngspice_con -b <gate>.cir`). This script only reads their output, checks it and
puts it in one table. These results are for evaluation only; nothing is trained on them.

Checks
    - each netlist's transistors (and spice/gates.lib) match gatelib/gates.py,
      which the ML side uses
    - all five netlists drew the same 200 corners
    - every output sits at its logic level

    python step1_collect_spice.py

Outputs
    results/corners.csv       the corners as drawn by ngspice (Part C checks against it)
    results/spice_gates.csv   one row per (gate, inputs, corner): node voltages and leakage
"""
import numpy as np
import pandas as pd

from gatelib import config as C
from gatelib.gates import ALL_NODES, GATES
from gatelib.spice_io import (check_gate_library, check_gate_netlist, check_same_corners, corners_from,
                              read_output)


def main():
    for name, gate in GATES.items():
        check_gate_netlist(gate, C.SPICE_DIR / f"{name.lower()}.cir")
    check_gate_library(GATES, C.SPICE_DIR / "gates.lib")
    print("netlists and gates.lib match gatelib/gates.py")

    parts, corners = [], None
    for name, gate in GATES.items():
        raw = read_output(C.spice_out(name), ["corner", *gate.inputs])
        c = corners_from(raw)
        if corners is None:
            corners = c
        else:
            check_same_corners(corners, c, f"{name} vs INV")
            if len(c) > len(corners):
                corners = c
        df = pd.DataFrame({
            "gate": name,
            "inputs": raw[list(gate.inputs)].astype(str).agg("".join, axis=1),
            "corner": raw["corner"],
        })
        for n in ALL_NODES:
            df[f"spice_v_{n}"] = raw[f"v_{n}"] if f"v_{n}" in raw else np.nan
        df["spice_ileak"] = raw["ileak"]
        df["spice_ivdd"] = raw["ivdd"]
        expected = np.array([gate.logic(*bits)["out"] for bits in raw[list(gate.inputs)].itertuples(index=False)])
        df["expected_out"] = expected
        df["logic_ok"] = (df["spice_v_out"] - expected * raw["vdd"]).abs() < 0.1 * raw["vdd"]
        n_cases = df.groupby("inputs").ngroups
        print(f"  {name:5s}: {len(df)} rows, {df['corner'].nunique()} corners x {n_cases} input cases, "
              f"{int((~df['logic_ok']).sum())} with a wrong output level")
        parts.append(df)

    spice = pd.concat(parts, ignore_index=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    corners.to_csv(C.CORNERS_CSV, index=False)
    spice.to_csv(C.SPICE_CSV, index=False)
    print(f"-> {C.CORNERS_CSV} ({len(corners)} corners), {C.SPICE_CSV} ({len(spice)} rows)")
    print(spice.groupby(["gate", "inputs"], sort=False)["spice_ileak"].median()
          .rename("median SPICE leakage (A)").to_string())


if __name__ == "__main__":
    main()
