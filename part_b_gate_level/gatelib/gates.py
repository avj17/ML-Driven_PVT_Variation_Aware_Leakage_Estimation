"""Transistor-level definitions of the five static CMOS gates.

One definition drives both the SPICE netlists and the ML prediction, so the
two always see the same transistors, terminals and widths.

Nets: "vdd" and "0" are the rails, "a"/"b" the inputs, "out" the output,
"x" the node between the series NMOS of a NAND2, "y" the node between the
series PMOS of a NOR2, and "n1" the NAND2/NOR2 output inside AND2/OR2.
Drains face the output, sources face the rail.
"""
import itertools
from dataclasses import dataclass
from typing import Callable

from . import config as C


@dataclass(frozen=True)
class Fet:
    name: str
    mos: str          # "nmos" or "pmos"
    d: str
    g: str
    s: str
    w_mult: float     # width in multiples of L_MIN

    @property
    def b(self):
        return "vdd" if self.mos == "pmos" else "0"

    @property
    def w(self):
        return self.w_mult * C.L_MIN


@dataclass(frozen=True)
class Gate:
    name: str
    inputs: tuple
    fets: tuple
    nodes: tuple          # nets the gate itself sets (solved in ML mode), in solve order
    logic: Callable       # bits -> {net: logic value} for driven nets ("out", "n1")

    def combos(self):
        return list(itertools.product((0, 1), repeat=len(self.inputs)))

    def net_logic(self, bits):
        """Logic value of every net that can drive a transistor gate."""
        vals = dict(zip(self.inputs, bits))
        vals.update(self.logic(*bits))
        return vals

    def fet_case(self, fet, bits):
        """Which of the four transistor models applies (gate at 0 or VDD)."""
        g = self.net_logic(bits)[fet.g]
        on = g == 1 if fet.mos == "nmos" else g == 0
        return f"{fet.mos.upper()}_{'ON' if on else 'OFF'}"


def bits_str(bits):
    return "".join(str(b) for b in bits)


def _inv(a, out, prefix=""):
    w = C.WIDTHS["INV"]
    return (Fet(f"{prefix}p", "pmos", out, a, "vdd", w["pmos"]),
            Fet(f"{prefix}n", "nmos", out, a, "0", w["nmos"]))


def _nand2(a, b, out, x, prefix=""):
    w = C.WIDTHS["NAND2"]
    return (Fet(f"{prefix}pa", "pmos", out, a, "vdd", w["pmos"]),
            Fet(f"{prefix}pb", "pmos", out, b, "vdd", w["pmos"]),
            Fet(f"{prefix}na", "nmos", out, a, x, w["nmos"]),       # top of the stack
            Fet(f"{prefix}nb", "nmos", x, b, "0", w["nmos"]))       # bottom of the stack


def _nor2(a, b, out, y, prefix=""):
    w = C.WIDTHS["NOR2"]
    return (Fet(f"{prefix}pa", "pmos", y, a, "vdd", w["pmos"]),     # top of the stack
            Fet(f"{prefix}pb", "pmos", out, b, y, w["pmos"]),       # bottom of the stack
            Fet(f"{prefix}na", "nmos", out, a, "0", w["nmos"]),
            Fet(f"{prefix}nb", "nmos", out, b, "0", w["nmos"]))


GATES = {
    "INV": Gate("INV", ("a",), _inv("a", "out"), ("out",),
                lambda a: {"out": 1 - a}),
    "NAND2": Gate("NAND2", ("a", "b"), _nand2("a", "b", "out", "x"), ("x", "out"),
                  lambda a, b: {"out": 1 - (a & b)}),
    "NOR2": Gate("NOR2", ("a", "b"), _nor2("a", "b", "out", "y"), ("y", "out"),
                 lambda a, b: {"out": 1 - (a | b)}),
    "AND2": Gate("AND2", ("a", "b"), _nand2("a", "b", "n1", "x", "nand_") + _inv("n1", "out", "inv_"),
                 ("x", "n1", "out"), lambda a, b: {"n1": 1 - (a & b), "out": a & b}),
    "OR2": Gate("OR2", ("a", "b"), _nor2("a", "b", "n1", "y", "nor_") + _inv("n1", "out", "inv_"),
                ("y", "n1", "out"), lambda a, b: {"n1": 1 - (a | b), "out": a | b}),
}

# Union of node columns recorded for any gate (NaN where a gate has no such node).
ALL_NODES = ["out", "x", "y", "n1"]
