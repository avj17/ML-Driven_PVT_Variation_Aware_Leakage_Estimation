"""Single-transistor SPICE datasets (Part A, step 1).

    python generate_data.py                 # 4 x 10,000 rows -> <CASE>_new.csv + example netlists
    python generate_data.py --extra 20000   # also 20,000 extra training rows per case -> <CASE>_new_extra.csv

The main 10,000 rows per case reuse the original random draws (seed 42, same
order), so their inputs are identical to the first datasets (<CASE>.csv) and
the train/val/test split selects the same rows. The extra rows come from the
same distributions with a different seed and are used for training only.

Every transistor now has drain/source junction geometry, with diffusion
length LDIFF = 2 * Lmin: AD = AS = W * LDIFF, PD = PS = 2 * (W + LDIFF).
(<CASE>.csv were simulated without it, so they have no junction-area leakage.)
The netlists use the body node as ground (see simulate_single_sample).
"""
import argparse
import os
import re
import csv
import shutil
import tempfile
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Fixed random seeds for reproducibility (Deliverable 9)
SEED = 42
SEED_EXTRA = 4242            # extra training rows: independent of the main draws
NUM_SAMPLES = 10000
L_MIN = 45e-9
LDIFF = 2 * L_MIN            # drain/source diffusion length
VDD_NOM = 0.9
SUFFIX = "_new"              # output names: <CASE>_new.csv, <case>_new.net
# On Windows, ngspice.exe is the GUI build and prints nothing to stdout; use the console build.
NGSPICE = os.environ.get("NGSPICE") or shutil.which("ngspice_con") or "ngspice"


def junction_geometry(w):
    """AD/AS/PD/PS instance parameters for a transistor of width w (metres)."""
    area, perim = w * LDIFF, 2 * (w + LDIFF)
    return f"AD={area:.8e} AS={area:.8e} PD={perim:.8e} PS={perim:.8e}"

# Nominal parameters from 45nm_HP.pm
NOMINAL_PARAMS = {
    'nmos': {'vth0': 0.46893,  'toxe': 1.25e-9, 'ndep': 3.24e18, 'u0': 0.054, 'xj': 1.4e-8},
    'pmos': {'vth0': -0.49158, 'toxe': 1.30e-9, 'ndep': 2.44e18, 'u0': 0.020, 'xj': 1.4e-8}
}

with open("45nm_HP.pm", "r") as f:
    RAW_MODEL_TEXT = f.read()

def replace_model_params(model_text, mos_type, sampled_params):
    """Replaces vth0, toxe, ndep, u0, xj inside the specified .model block."""
    parts = re.split(r'(\.model\s+pmos\s+pmos)', model_text, flags=re.IGNORECASE)
    nmos_block = parts[0]
    pmos_block = parts[1] + parts[2]

    target_block = nmos_block if mos_type == 'nmos' else pmos_block
    for param, val in sampled_params.items():
        pattern = rf'(\b{param}\s*=\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        target_block = re.sub(pattern, rf'\g<1>{val:.8e}', target_block, count=1)

    return (target_block + pmos_block) if mos_type == 'nmos' else (nmos_block + target_block)

def save_example_netlists():
    """Saves the 4 clean example single-transistor netlists for Deliverable 2."""
    examples = [
        (f"nmos_on{SUFFIX}.net",  "NMOS_ON",  "nmos", 0.9, 0.0, 0.9, 0.0),
        (f"nmos_off{SUFFIX}.net", "NMOS_OFF", "nmos", 0.9, 0.0, 0.0, 0.0),
        (f"pmos_on{SUFFIX}.net",  "PMOS_ON",  "pmos", 0.0, 0.9, 0.0, 0.9),
        (f"pmos_off{SUFFIX}.net", "PMOS_OFF", "pmos", 0.0, 0.9, 0.9, 0.9),
    ]
    w_example = 180e-9
    for fname, label, mos, vd, vs, vg, vb in examples:
        content = f"""* Deliverable 2 Example Single-Transistor Netlist: {label}
* Absolute bias: Vd={vd} Vg={vg} Vs={vs} Vb={vb}. Ground is the body node, so every
* source is shifted by -Vb (same device biases; avoids round-off in the body current).
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9 numdgt=12
.temp 27.0

.include "45nm_HP.pm"

Vd d 0 DC {vd - vb:g}
Vg g 0 DC {vg - vb:g}
Vs s 0 DC {vs - vb:g}
Vb b 0 DC {vb - vb:g}

* Junction geometry: Ldiff = 2*Lmin = 90n, AD = AS = W*Ldiff, PD = PS = 2*(W+Ldiff)
M1 d g s b {mos} W=180n L=45n {junction_geometry(w_example)}

.control
op
print -i(Vd)
print -i(Vg)
print -i(Vs)
print -i(Vb)
quit
.endc
.end
"""
        with open(fname, "w") as f:
            f.write(content)
    print(f"Saved 4 Deliverable 2 example netlists: {', '.join(e[0] for e in examples)}")

def simulate_single_sample(args):
    """Runs a single Ngspice simulation in batch mode and extracts Id, Ig, Is, Ib."""
    case_name, mos_type, row_dict = args

    sampled_p = {
        'vth0': row_dict['vth0'],
        'toxe': row_dict['toxe'],
        'ndep': row_dict['ndep'],
        'u0':   row_dict['u0'],
        'xj':   row_dict['xj']
    }
    modified_model = replace_model_params(RAW_MODEL_TEXT, mos_type, sampled_p)

    # The netlist ground is the body node: every source is shifted by -vb. The
    # device sees the same terminal voltage differences (all BSIM4 depends on),
    # but a branch current solved next to a 0.9 V node carries ~1e-17 A of
    # floating-point round-off, which swamped the tiny PMOS body current
    # (body at VDD). For NMOS vb = 0, so nothing changes. The CSV keeps the
    # absolute voltages of spec Section 4.
    ref = row_dict['vb']
    netlist = f"""* {case_name} Sample Simulation (body-referenced: all sources shifted by -vb = {-ref:.8e} V)
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9 numdgt=12
.temp {row_dict['temp']:.6f}

{modified_model}

Vd d 0 DC {row_dict['vd'] - ref:.8e}
Vg g 0 DC {row_dict['vg'] - ref:.8e}
Vs s 0 DC {row_dict['vs'] - ref:.8e}
Vb b 0 DC {row_dict['vb'] - ref:.8e}

M1 d g s b {mos_type} W={row_dict['w']:.8e} L={L_MIN:.8e} {junction_geometry(row_dict['w'])}

.control
op
print -i(Vd)
print -i(Vg)
print -i(Vs)
print -i(Vb)
quit
.endc
.end
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.net', delete=False) as tmp:
        tmp.write(netlist)
        tmp_path = tmp.name

    try:
        res = subprocess.run([NGSPICE, '-b', tmp_path], capture_output=True, text=True, check=True)
        out = res.stdout
        id_val = float(re.search(r'-i\(vd\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', out, re.I).group(1))
        ig_val = float(re.search(r'-i\(vg\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', out, re.I).group(1))
        is_val = float(re.search(r'-i\(vs\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', out, re.I).group(1))
        ib_val = float(re.search(r'-i\(vb\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', out, re.I).group(1))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    row_dict['id'] = id_val
    row_dict['ig'] = ig_val
    row_dict['is'] = is_val
    row_dict['ib'] = ib_val
    return row_dict

def draw_case_inputs(mos_type, is_on, rng, n):
    """n random input rows for one case (spec Section 4 distributions)."""
    nom = NOMINAL_PARAMS[mos_type]
    rows = []
    for _ in range(n):
        vdd = rng.uniform(0.9 * VDD_NOM, 1.1 * VDD_NOM)
        temp = rng.uniform(-40.0, 125.0)
        w = rng.uniform(1.0 * L_MIN, 8.0 * L_MIN)
        vd = rng.uniform(0.0, vdd)
        vs = rng.uniform(0.0, vdd)

        if mos_type == 'nmos':
            vg = vdd if is_on else 0.0
            vb = 0.0
        else:
            vg = 0.0 if is_on else vdd
            vb = vdd

        def draw_pm10(val):
            low, high = sorted([0.9 * val, 1.1 * val])
            return rng.uniform(low, high)

        row = {
            'vdd': vdd, 'temp': temp, 'w': w,
            'vd': vd, 'vs': vs, 'vg': vg, 'vb': vb,
            'vth0': draw_pm10(nom['vth0']),
            'toxe': draw_pm10(nom['toxe']),
            'ndep': draw_pm10(nom['ndep']),
            'u0':   draw_pm10(nom['u0']),
            'xj':   draw_pm10(nom['xj'])
        }
        rows.append(row)
    return rows


def simulate_rows(case_name, mos_type, rows, csv_filename, workers=None):
    """Simulate every row with ngspice and write the CSV."""
    print(f"Simulating {len(rows)} samples for {case_name} -> {csv_filename}")
    tasks = [(case_name, mos_type, row) for row in rows]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, res in enumerate(executor.map(simulate_single_sample, tasks, chunksize=50), 1):
            results.append(res)
            if idx % 2000 == 0 or idx == len(rows):
                print(f"  [{case_name}] Completed {idx}/{len(rows)} simulations...", flush=True)

    fieldnames = ['vdd', 'temp', 'w', 'vd', 'vs', 'vg', 'vb',
                  'vth0', 'toxe', 'ndep', 'u0', 'xj', 'id', 'ig', 'is', 'ib']
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    kcl_errors = [abs(r['id'] + r['ig'] + r['is'] + r['ib']) for r in results]
    print(f"Done -> {csv_filename} ({len(results)} rows) | Max KCL error: {max(kcl_errors):.2e} A\n")


CASES = [
    ("NMOS_ON",  "nmos", True),
    ("NMOS_OFF", "nmos", False),
    ("PMOS_ON",  "pmos", True),
    ("PMOS_OFF", "pmos", False)
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", type=int, default=0, help="extra training rows per case (seed SEED_EXTRA)")
    ap.add_argument("--cases", nargs="+", default=[c[0] for c in CASES])
    ap.add_argument("--skip-main", action="store_true", help="only generate the extra rows")
    ap.add_argument("--limit", type=int, default=None, help="simulate only the first N rows (timing test)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    save_example_netlists()
    # Draw every case in the original order so each case gets the same inputs as before.
    rng = np.random.default_rng(SEED)
    main_rows = {c: draw_case_inputs(m, on, rng, NUM_SAMPLES) for c, m, on in CASES}
    rng_extra = np.random.default_rng(SEED_EXTRA)
    extra_rows = {c: draw_case_inputs(m, on, rng_extra, args.extra) for c, m, on in CASES}

    for case_name, mos_type, _ in CASES:
        if case_name not in args.cases:
            continue
        if not args.skip_main:
            simulate_rows(case_name, mos_type, main_rows[case_name][:args.limit],
                          f"{case_name}{SUFFIX}.csv", args.workers)
        if args.extra:
            simulate_rows(case_name, mos_type, extra_rows[case_name][:args.limit],
                          f"{case_name}{SUFFIX}_extra.csv", args.workers)