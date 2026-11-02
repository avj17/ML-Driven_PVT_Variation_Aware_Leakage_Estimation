import os
import re
import csv
import tempfile
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Fixed random seed for reproducibility (Deliverable 9)
SEED = 42
NUM_SAMPLES = 10000
L_MIN = 45e-9
VDD_NOM = 0.9

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
        ("nmos_on.net",  "NMOS_ON",  "nmos", 0.9, 0.0, 0.9, 0.0),
        ("nmos_off.net", "NMOS_OFF", "nmos", 0.9, 0.0, 0.0, 0.0),
        ("pmos_on.net",  "PMOS_ON",  "pmos", 0.0, 0.9, 0.0, 0.9),
        ("pmos_off.net", "PMOS_OFF", "pmos", 0.0, 0.9, 0.9, 0.9),
    ]
    for fname, label, mos, vd, vs, vg, vb in examples:
        content = f"""* Deliverable 2 Example Single-Transistor Netlist: {label}
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9 numdgt=12
.temp 27.0

.include "45nm_HP.pm"

Vd d 0 DC {vd}
Vg g 0 DC {vg}
Vs s 0 DC {vs}
Vb b 0 DC {vb}

M1 d g s b {mos} W=180n L=45n

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
    print("Saved 4 Deliverable 2 example netlists: nmos_on.net, nmos_off.net, pmos_on.net, pmos_off.net")

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

    netlist = f"""* {case_name} Sample Simulation
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9 numdgt=12
.temp {row_dict['temp']:.6f}

{modified_model}

Vd d 0 DC {row_dict['vd']:.8e}
Vg g 0 DC {row_dict['vg']:.8e}
Vs s 0 DC {row_dict['vs']:.8e}
Vb b 0 DC {row_dict['vb']:.8e}

M1 d g s b {mos_type} W={row_dict['w']:.8e} L={L_MIN:.8e}

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
        res = subprocess.run(['ngspice', '-b', tmp_path], capture_output=True, text=True, check=True)
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

def generate_case_dataset(case_name, mos_type, is_on, rng):
    print(f"Generating {NUM_SAMPLES} samples for {case_name}...")
    nom = NOMINAL_PARAMS[mos_type]
    tasks = []

    for _ in range(NUM_SAMPLES):
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
        tasks.append((case_name, mos_type, row))

    results = []
    with ProcessPoolExecutor() as executor:
        for idx, res in enumerate(executor.map(simulate_single_sample, tasks, chunksize=50), 1):
            results.append(res)
            if idx % 2000 == 0 or idx == NUM_SAMPLES:
                print(f"  [{case_name}] Completed {idx}/{NUM_SAMPLES} simulations...")

    fieldnames = ['vdd', 'temp', 'w', 'vd', 'vs', 'vg', 'vb',
                  'vth0', 'toxe', 'ndep', 'u0', 'xj', 'id', 'ig', 'is', 'ib']
    csv_filename = f"{case_name}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    kcl_errors = [abs(r['id'] + r['ig'] + r['is'] + r['ib']) for r in results]
    print(f"Done -> {csv_filename} ({len(results)} rows) | Max KCL error: {max(kcl_errors):.2e} A\n")

if __name__ == "__main__":
    save_example_netlists()
    rng = np.random.default_rng(SEED)
    cases = [
        ("NMOS_ON",  "nmos", True),
        ("NMOS_OFF", "nmos", False),
        ("PMOS_ON",  "pmos", True),
        ("PMOS_OFF", "pmos", False)
    ]
    for case_name, mos_type, is_on in cases:
        generate_case_dataset(case_name, mos_type, is_on, rng)