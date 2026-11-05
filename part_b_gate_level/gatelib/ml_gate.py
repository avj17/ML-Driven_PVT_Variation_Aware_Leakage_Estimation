"""ML gate model: gate leakage from the four Part A transistor models only.

For one gate, input combination and a batch of corners:
  1. every net gets a voltage: rails, inputs at exactly 0 or VDD, and the gate's
     own nodes (internal stack node, NAND/NOR output inside AND/OR, output),
     either recorded by SPICE (spec Step 4) or solved here by KCL on the models;
  2. each transistor's (Vd, Vg, Vs, Vb), width, VDD, temperature and its device
     type's process draw go to the ON or OFF model of that type, chosen by the
     logic value at its gate;
  3. leakage = static power / VDD, from the predicted terminal currents:

        I_leak = (1/VDD) * sum over transistors of (Vds*Id + Vgs*Ig + Vbs*Ib)

     This is the same quantity as the SPICE leakage the netlists compute,
     -I(VDD) - sum(bit * I(V_in)) (Tellegen's theorem). It is used instead of summing the currents at the VDD
     terminals because there an ON transistor with |Vds| of microvolts would
     contribute its predicted channel current directly, and the models are least
     accurate at Vds ~ 0. Here that current is multiplied by the tiny Vds, so the
     leakage comes from the OFF transistors (large Vds) and the gate currents.
     Writing it relative to the source assumes Is = -(Id + Ig + Ib) (KCL).
"""
import sys
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .gates import GATES, bits_str

sys.path.insert(0, str(C.MODEL_TRAINING))
from leakage_ml.predictor import LeakagePredictor  # noqa: E402

# The saved scalers were fitted on DataFrames and are called with arrays.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

TARGETS = ["id", "ig", "is", "ib"]
TERMINAL = {"d": 0, "g": 1, "s": 2}       # column of each terminal current


class TransistorModels:
    """The four chosen Part A models (saved_models/<case>.joblib)."""

    def __init__(self, models_dir=C.SAVED_MODELS):
        self.models = {case: LeakagePredictor.load(models_dir / f"{case}.joblib") for case in C.CASES}

    def predict(self, case, df) -> np.ndarray:
        return self.models[case].predict(df)[TARGETS].to_numpy(dtype=float)


def rail_voltages(gate, bits, corners):
    """Rails and inputs; the gate's own nodes are added by the caller."""
    vdd = corners["vdd"].to_numpy(dtype=float)
    V = {"vdd": vdd, "0": np.zeros_like(vdd)}
    for net, bit in zip(gate.inputs, bits):
        V[net] = bit * vdd
    return V


def _fet_frame(fet, corners, V):
    pre = "n_" if fet.mos == "nmos" else "p_"
    df = pd.DataFrame({
        "vdd": corners["vdd"].to_numpy(dtype=float),
        "temp": corners["temp"].to_numpy(dtype=float),
        "w": fet.w,
        "vd": V[fet.d], "vs": V[fet.s], "vg": V[fet.g], "vb": V[fet.b],
    })
    for p in C.PROCESS_PARAMS:
        df[p] = corners[pre + p].to_numpy(dtype=float)
    return df


def fet_currents(models, gate, bits, corners, V, fets=None):
    """{fet name: (n, 4) array [id, ig, is, ib]}, one model call per transistor case."""
    fets = gate.fets if fets is None else fets
    n = len(corners)
    by_case = {}
    for f in fets:
        by_case.setdefault(gate.fet_case(f, bits), []).append(f)
    out = {}
    for case, group in by_case.items():
        I = models.predict(case, pd.concat([_fet_frame(f, corners, V) for f in group], ignore_index=True))
        for k, f in enumerate(group):
            out[f.name] = I[k * n:(k + 1) * n]
    return out


def leakage(gate, bits, V, I):
    """(total leakage, VDD-supply current) in amperes, arrays over corners.
    The VDD-supply current is the total minus the gate currents drawn from the
    inputs held at VDD (those are delivered by the input sources)."""
    p = 0.0
    for f in gate.fets:
        i_d, i_g, _, i_b = I[f.name].T
        vs = V[f.s]
        p = p + (V[f.d] - vs) * i_d + (V[f.g] - vs) * i_g + (V[f.b] - vs) * i_b
    total = p / V["vdd"]
    high = {net for net, bit in zip(gate.inputs, bits) if bit}
    i_inputs = sum((I[f.name][:, 1] for f in gate.fets if f.g in high), np.zeros_like(total))
    return total, total - i_inputs


def solve_nodes(models, gate, bits, corners, iters=C.SOLVE_ITERS, sweeps=C.SOLVE_SWEEPS):
    """ML estimate of the gate's own node voltages: at each node the predicted
    currents into the attached transistor terminals must sum to zero (KCL).
    That sum rises with the node voltage, so each node is bisected over
    [0, VDD] (vectorised over corners) with the other nodes held, in
    Gauss-Seidel sweeps until the coupled nodes settle."""
    V = rail_voltages(gate, bits, corners)
    logic = gate.net_logic(bits)
    for node in gate.nodes:                      # start from the logic levels
        V[node] = logic[node] * V["vdd"] if node in logic else 0.5 * V["vdd"]
    attached = {node: [(f, TERMINAL[t]) for f in gate.fets for t in TERMINAL if getattr(f, t) == node]
                for node in gate.nodes}
    for _ in range(sweeps if len(gate.nodes) > 1 else 1):
        for node in gate.nodes:
            fets = list({f.name: f for f, _ in attached[node]}.values())
            lo, hi = np.zeros_like(V["vdd"]), V["vdd"].copy()
            for _ in range(iters):
                V[node] = 0.5 * (lo + hi)
                I = fet_currents(models, gate, bits, corners, V, fets)
                into_fets = sum(I[f.name][:, k] for f, k in attached[node])
                too_high = into_fets > 0               # devices drain the node: lower it
                hi = np.where(too_high, V[node], hi)
                lo = np.where(too_high, lo, V[node])
            V[node] = 0.5 * (lo + hi)
    return V


def predict_gate(models, gate, bits, corners, node_voltages=None):
    """Gate leakage for one input combination over a batch of corners.

    node_voltages: {node: array} recorded by SPICE (spec Step 4). If None the
    nodes are solved with the models (`solve_nodes`), which needs no SPICE.
    Returns (total leakage, VDD current, V) with V the net voltages used."""
    if node_voltages is None:
        V = solve_nodes(models, gate, bits, corners)
    else:
        V = rail_voltages(gate, bits, corners)
        V.update({n: np.asarray(v, dtype=float) for n, v in node_voltages.items()})
    I = fet_currents(models, gate, bits, corners, V)
    total, ivdd = leakage(gate, bits, V, I)
    return total, ivdd, V


class GateModel:
    """The Part B gate model, the middle level of the hierarchy
    (circuit -> gate model -> transistor models).

    predict(gate name, input bits, corners) -> (total leakage, VDD current, V).
    Node voltages:
      "spice"  the internal/output node voltages pre-recorded by the gate-level
               SPICE runs (spec Step 4), looked up by gate, inputs and corner;
      "ml"     solved by KCL on the transistor models (no SPICE input at all).
    The corners must carry the columns of config.CORNER_COLS."""

    def __init__(self, nodes="spice", transistor_models=None, spice_gates=None):
        self.nodes = nodes
        self.models = transistor_models or TransistorModels()
        if nodes == "spice":
            if spice_gates is None:
                spice_gates = pd.read_csv(C.SPICE_CSV, dtype={"inputs": str})
            self.recorded = {k: g.set_index("corner") for k, g in spice_gates.groupby(["gate", "inputs"])}
        elif nodes != "ml":
            raise ValueError(f"nodes must be 'spice' or 'ml', not {nodes!r}")

    def predict(self, gate_name, bits, corners):
        gate = GATES[gate_name]
        corners = corners.reset_index(drop=True)
        node_voltages = None
        if self.nodes == "spice":
            rec = self.recorded[(gate_name, bits_str(bits))].reindex(corners["corner"])
            node_voltages = {n: rec[f"spice_v_{n}"].to_numpy() for n in gate.nodes}
        return predict_gate(self.models, gate, bits, corners, node_voltages)
