"""3-bit carry-lookahead adder as a list of Part B gates: the structural
description the hierarchical ML side works from.

The SPICE side is the ngspice netlist spice/cla3.cir, built from the same gates
(part_b_gate_level/spice/gates.lib); `check_netlist` verifies that its X cards
match INSTANCES one for one.

    P_i = A_i XOR B_i,  G_i = A_i AND B_i
    C1 = G0 + P0 C0
    C2 = G1 + P1 G0 + P1 P0 C0
    C3 = G2 + P2 G1 + P2 P1 G0 + P2 P1 P0 C0
    S_i = P_i XOR C_i
XOR(a, b) = AND2(NAND2(a, b), OR2(a, b)); G_i = INV(NAND2(A_i, B_i)) shares that NAND2.
"""
from dataclasses import dataclass

from . import config as CC
from gatelib.gates import GATES

N_BITS = 3
INPUTS = [f"a{i}" for i in range(N_BITS)] + [f"b{i}" for i in range(N_BITS)] + ["c0"]
OUTPUTS = [f"s{i}" for i in range(N_BITS)] + [f"c{N_BITS}"]


@dataclass(frozen=True)
class Inst:
    name: str
    gate: str        # key of gatelib.gates.GATES
    ins: tuple
    out: str


def _build():
    g = []
    for i in range(N_BITS):                                  # propagate and generate
        g += [Inst(f"nab{i}", "NAND2", (f"a{i}", f"b{i}"), f"nab{i}"),
              Inst(f"gen{i}", "INV", (f"nab{i}",), f"g{i}"),
              Inst(f"oab{i}", "OR2", (f"a{i}", f"b{i}"), f"oab{i}"),
              Inst(f"prp{i}", "AND2", (f"nab{i}", f"oab{i}"), f"p{i}")]
    g += [Inst("t_p0c0", "AND2", ("p0", "c0"), "t_p0c0"),     # C1
          Inst("c1", "OR2", ("g0", "t_p0c0"), "c1"),
          Inst("t_p1g0", "AND2", ("p1", "g0"), "t_p1g0"),     # C2
          Inst("t_p1p0", "AND2", ("p1", "p0"), "t_p1p0"),
          Inst("t_p1p0c0", "AND2", ("t_p1p0", "c0"), "t_p1p0c0"),
          Inst("t_c2a", "OR2", ("g1", "t_p1g0"), "t_c2a"),
          Inst("c2", "OR2", ("t_c2a", "t_p1p0c0"), "c2"),
          Inst("t_p2g1", "AND2", ("p2", "g1"), "t_p2g1"),     # C3
          Inst("t_p2p1", "AND2", ("p2", "p1"), "t_p2p1"),
          Inst("t_p2p1g0", "AND2", ("t_p2p1", "g0"), "t_p2p1g0"),
          Inst("t_p2p1p0c0", "AND2", ("p2", "t_p1p0c0"), "t_p2p1p0c0"),
          Inst("t_c3a", "OR2", ("g2", "t_p2g1"), "t_c3a"),
          Inst("t_c3b", "OR2", ("t_p2p1g0", "t_p2p1p0c0"), "t_c3b"),
          Inst("c3", "OR2", ("t_c3a", "t_c3b"), "c3")]
    for i in range(N_BITS):                                  # sums
        g += [Inst(f"sn{i}", "NAND2", (f"p{i}", f"c{i}"), f"sn{i}"),
              Inst(f"so{i}", "OR2", (f"p{i}", f"c{i}"), f"so{i}"),
              Inst(f"sum{i}", "AND2", (f"sn{i}", f"so{i}"), f"s{i}")]
    return g


INSTANCES = _build()
N_VECTORS = 2 ** len(INPUTS)


def vector(v: int) -> dict:
    """Input vector number v (0..127): A = bits 0-2, B = bits 3-5, Cin = bit 6."""
    a, b, cin = v & 7, (v >> 3) & 7, (v >> 6) & 1
    vec = {f"a{i}": (a >> i) & 1 for i in range(N_BITS)}
    vec.update({f"b{i}": (b >> i) & 1 for i in range(N_BITS)})
    vec["c0"] = cin
    return vec


def vector_number(vec: dict) -> int:
    return sum(vec[f"a{i}"] << i for i in range(N_BITS)) + sum(vec[f"b{i}"] << (N_BITS + i)
                                                              for i in range(N_BITS)) + (vec["c0"] << 2 * N_BITS)


def simulate(vec: dict):
    """Logic values of every net, and the input bits each gate instance sees."""
    nets = dict(vec)
    seen = []
    for inst in INSTANCES:                       # instances are in topological order
        bits = tuple(nets[n] for n in inst.ins)
        nets[inst.out] = GATES[inst.gate].logic(*bits)["out"]
        seen.append((inst, bits))
    return nets, seen


def check_adder():
    """The gate network must add: S + 8*C3 == A + B + Cin for every vector."""
    for v in range(N_VECTORS):
        nets, _ = simulate(vector(v))
        got = sum(nets[f"s{i}"] << i for i in range(N_BITS)) + (nets[f"c{N_BITS}"] << N_BITS)
        assert got == (v & 7) + ((v >> 3) & 7) + (v >> 6), f"adder logic wrong for vector {v}"


def check_netlist(path=CC.SPICE_DIR / "cla3.cir"):
    """The X cards of the ngspice netlist must be exactly INSTANCES
    (X<name> inputs... out vdd <GATE>)."""
    got = set()
    for ln in path.read_text().splitlines():
        t = ln.split()
        if t and t[0][0].lower() == "x":
            got.add((t[0][1:].lower(), t[-1].upper(), tuple(x.lower() for x in t[1:-3]), t[-3].lower()))
    want = {(i.name, i.gate, i.ins, i.out) for i in INSTANCES}
    if got != want:
        raise SystemExit(f"{path.name} does not match cla/circuit.py:\n"
                         f"  only in netlist: {sorted(got - want)}\n  only in circuit.py: {sorted(want - got)}")


def n_transistors():
    return sum(len(GATES[i.gate].fets) for i in INSTANCES)
