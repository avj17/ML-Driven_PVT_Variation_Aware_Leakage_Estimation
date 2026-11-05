"""Reading the ngspice results and checking the netlists against gates.py.

The netlists write one whitespace-separated line per operating point with
`echo ... >> file`; the first line is the header. Values substituted with $&
may be printed as floats, so integer columns are converted back here.
"""
import re
import subprocess
from collections import Counter

import numpy as np
import pandas as pd

from . import config as C

SUFFIX = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3}


def run_netlist(path, ngspice=C.NGSPICE):
    """Run one netlist in batch mode from its own folder (relative .include and
    output paths in the netlists are resolved from there)."""
    print(f"  {ngspice} -b {path.name}   (in {path.parent})", flush=True)
    res = subprocess.run([ngspice, "-b", path.name], cwd=path.parent, capture_output=True, text=True)
    log = path.with_suffix(".log")
    log.write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        raise SystemExit(f"ngspice failed on {path.name} (exit {res.returncode}); see {log}")


def read_output(path, int_cols) -> pd.DataFrame:
    """Parse a netlist's output file; rows that do not parse (e.g. an echo of a
    vector that failed to converge) are dropped and reported."""
    if not path.exists():
        raise SystemExit(f"{path} not found; run the ngspice netlist first.")
    lines = path.read_text().splitlines()
    header = lines[0].split()
    rows, bad = [], 0
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != len(header):
            bad += 1
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            bad += 1
    if bad:
        print(f"  WARNING {path.name}: {bad} malformed lines skipped")
    df = pd.DataFrame(rows, columns=header)
    for c in int_cols:
        df[c] = df[c].round().astype(int)
    return df


def corners_from(df) -> pd.DataFrame:
    return df[C.CORNER_COLS].drop_duplicates("corner").sort_values("corner").reset_index(drop=True)


def check_same_corners(a, b, what):
    """Every netlist draws its corners with the same seed; make sure they agree."""
    m = a.merge(b, on="corner", suffixes=("_a", "_b"))
    for c in C.CORNER_COLS[1:]:
        if not np.allclose(m[f"{c}_a"], m[f"{c}_b"], rtol=1e-6, atol=0):
            raise SystemExit(f"corner mismatch in {c} ({what}): the corner blocks of the netlists differ")


# ----------------------------------------------------------------------------
# Netlist <-> gates.py consistency
# ----------------------------------------------------------------------------
def _value(tok):
    m = re.fullmatch(r"([-+]?[\d.]+(?:e[-+]?\d+)?)([fpnumk]?)", tok.lower())
    return float(m.group(1)) * SUFFIX.get(m.group(2), 1.0)


def mosfets(lines, rename=None):
    """Multiset of (type, d, g, s, b, W/L) from MOSFET cards."""
    r = rename or {}
    out = Counter()
    for ln in lines:
        t = ln.split()
        if not t or t[0][0].lower() != "m":
            continue
        d, g, s, b, mos = (r.get(x.lower(), x.lower()) for x in t[1:6])
        w = next(_value(x.split("=")[1]) for x in t[6:] if x.lower().startswith("w="))
        out[(mos, d, g, s, b, round(w / C.L_MIN, 3))] += 1
    return out


def expected_mosfets(gate):
    return Counter((f.mos, f.d, f.g, f.s, f.b, round(f.w_mult, 3)) for f in gate.fets)


def check_gate_netlist(gate, path):
    """The transistors of a flat gate netlist must equal gates.py's."""
    got = mosfets(path.read_text().splitlines())
    if got != expected_mosfets(gate):
        raise SystemExit(f"{path.name} does not match gatelib/gates.py for {gate.name}:\n"
                         f"  netlist: {sorted(got.items())}\n  gates.py: {sorted(expected_mosfets(gate).items())}")


def check_gate_library(gates, path):
    """Each .subckt in gates.lib must equal the gates.py definition; ports are
    (inputs..., out, vdd) as in gates.py."""
    text = path.read_text().splitlines()
    found = set()
    for m in re.finditer(r"^\.subckt\s+(\w+)\s+([^\n]+)$(.*?)^\.ends", "\n".join(text),
                         re.IGNORECASE | re.MULTILINE | re.DOTALL):
        name, ports, body = m.group(1).upper(), m.group(2).split(), m.group(3).splitlines()
        gate = gates[name]
        if [p.lower() for p in ports] != list(gate.inputs) + ["out", "vdd"]:
            raise SystemExit(f"gates.lib {name}: ports {ports} != {list(gate.inputs) + ['out', 'vdd']}")
        if mosfets(body) != expected_mosfets(gate):
            raise SystemExit(f"gates.lib {name} does not match gatelib/gates.py")
        found.add(name)
    missing = set(gates) - found
    if missing:
        raise SystemExit(f"gates.lib is missing {sorted(missing)}")
