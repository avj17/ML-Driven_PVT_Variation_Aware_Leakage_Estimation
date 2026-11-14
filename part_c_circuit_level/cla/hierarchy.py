"""Hierarchical ML leakage model of the CLA (spec Step 7).

    circuit level   CLAModel          logic-simulates the input vector to find the
                                      input combination at every gate, sums gates
    gate level      GateModel (Part B)  gate leakage from its transistors
    device level    TransistorModels    the four Part A models (NMOS/PMOS, ON/OFF)

Every gate input is taken at exactly 0 or VDD and loading between gates is
ignored, as the spec states. Gate leakages are supply power / VDD, which is
additive: the current one gate's input draws is supplied, in the circuit, by
the gate that drives it.
"""
import numpy as np

from .circuit import simulate, vector


class CLAModel:
    def __init__(self, gate_model):
        self.gate_model = gate_model
        self._cache = {}            # (gate, bits) -> leakage over the current corners
        self._corner_ids = None

    def _gate_leakage(self, gate, bits, corners):
        key = (gate, bits)
        if key not in self._cache:
            self._cache[key] = self.gate_model.predict(gate, bits, corners)[0]
        return self._cache[key]

    def predict(self, v, corners):
        """Circuit leakage for input vector number v over `corners` (a corners
        table); also returns the per-gate-type breakdown. Identical gate states
        repeat across instances and vectors, so each (gate, inputs) is
        evaluated once per corner set."""
        ids = tuple(corners["corner"])
        if self._corner_ids != ids:
            self._cache, self._corner_ids = {}, ids
        _, seen = simulate(vector(v))
        total = np.zeros(len(corners))
        by_type = {}
        for inst, bits in seen:
            leak = self._gate_leakage(inst.gate, bits, corners)
            total = total + leak
            by_type[inst.gate] = by_type.get(inst.gate, 0.0) + leak
        return total, by_type
