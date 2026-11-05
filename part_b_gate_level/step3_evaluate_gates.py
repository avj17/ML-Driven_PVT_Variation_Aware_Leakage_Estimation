"""Step 3 (spec Step 5): accuracy of the ML gate leakage against SPICE.

    python step3_evaluate_gates.py                 # total leakage (supply power / VDD)
    python step3_evaluate_gates.py --metric vdd    # current from the VDD source only

Outputs (results/)
    gate_results_table.md / .csv          Deliverable 6: per gate and input case (SPICE nodes)
    gate_results_table_mlnodes.csv        the same with ML-solved node voltages
    node_voltages.csv                     SPICE vs ML-solved node voltages
    plots/gate_parity.png                 Deliverable 7: ML vs SPICE, all runs, log axes
    plots/gate_parity_mlnodes.png         the same with ML-solved node voltages
    plots/gate_error_vs_corner.png        |error| vs temperature, VDD, vth0, toxe
"""
import argparse

import pandas as pd

from gatelib import config as C
from gatelib.gates import GATES
from gatelib.report import (error_vs_corner_plot, leak_metrics, metrics_table, parity_plot,
                            signed_err_pct, to_md)

METRICS = {"total": ("spice_ileak", "ml_ileak"), "vdd": ("spice_ivdd", "ml_ivdd")}
COLS = ["gate", "inputs", "n", "spice_median_A", "ml_median_A", "bias_pct",
        "median_err_pct", "p95_err_pct", "mean_err_pct", "max_err_pct", "mean_accuracy_pct"]


def results_table(df, spice_col, ml_col):
    """Per (gate, inputs) rows, then one row per gate and an overall row."""
    case = metrics_table(df, ["gate", "inputs"], spice_col, ml_col)
    per_gate = metrics_table(df, ["gate"], spice_col, ml_col).assign(inputs="all")
    overall = pd.DataFrame([{"gate": "ALL", "inputs": "all", **leak_metrics(df[spice_col], df[ml_col])}])
    return pd.concat([case, per_gate, overall], ignore_index=True)[COLS]


def node_table(df):
    rows = []
    for (gate, inputs), g in df.groupby(["gate", "inputs"], sort=False):
        for n in GATES[gate].nodes:
            d = 1e3 * (g[f"ml_v_{n}"] - g[f"spice_v_{n}"]).abs()
            rows.append({"gate": gate, "inputs": inputs, "node": n,
                         "spice_median_V": g[f"spice_v_{n}"].median(), "ml_median_V": g[f"ml_v_{n}"].median(),
                         "median_abs_diff_mV": d.median(), "max_abs_diff_mV": d.max()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=list(METRICS), default="total")
    args = ap.parse_args()

    if not C.GATE_RESULTS_CSV.exists():
        raise SystemExit(f"{C.GATE_RESULTS_CSV} not found; run step2_ml_gates.py first.")
    df = pd.read_csv(C.GATE_RESULTS_CSV, dtype={"inputs": str})
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    spice_col, ml_col = METRICS[args.metric]
    suffix = "" if args.metric == "total" else "_vdd"
    has_solve = f"{ml_col}_mlnodes" in df.columns
    label = {"total": "total leakage (supply power / VDD)", "vdd": "VDD supply current"}[args.metric]

    df["case"] = df["gate"] + " " + df["inputs"]
    df["err_pct"] = signed_err_pct(df[spice_col], df[ml_col])

    table = results_table(df, spice_col, ml_col)
    table.to_csv(C.RESULTS_DIR / f"gate_results_table{suffix}.csv", index=False)
    md = [f"# Gate results: {label}, ML vs SPICE\n",
          "Bias = mean signed relative error (ML - SPICE) / SPICE. Errors are |relative errors| in %.\n",
          "## ML with SPICE-recorded node voltages (spec Step 4)\n", to_md(table), ""]

    parity_plot(df[spice_col], df[ml_col], df["gate"],
                f"Gate leakage, ML vs SPICE ({len(df)} runs)", C.PLOTS_DIR / f"gate_parity{suffix}.png")
    error_vs_corner_plot(df, "err_pct", df["gate"], "Gate leakage |error| vs corner",
                         C.PLOTS_DIR / f"gate_error_vs_corner{suffix}.png")

    if has_solve:
        t2 = results_table(df, spice_col, f"{ml_col}_mlnodes")
        t2.to_csv(C.RESULTS_DIR / f"gate_results_table_mlnodes{suffix}.csv", index=False)
        nodes = node_table(df)
        nodes.to_csv(C.RESULTS_DIR / "node_voltages.csv", index=False)
        md += ["## ML with ML-solved node voltages (no SPICE input)\n", to_md(t2), "",
               "## Node voltages: SPICE vs ML-solved\n", to_md(nodes), ""]
        parity_plot(df[spice_col], df[f"{ml_col}_mlnodes"], df["gate"],
                    "Gate leakage, ML (ML-solved nodes) vs SPICE",
                    C.PLOTS_DIR / f"gate_parity_mlnodes{suffix}.png")

    worst = df.reindex(df["err_pct"].abs().sort_values(ascending=False).index).head(15)
    md += ["## 15 largest errors (SPICE nodes)\n",
           to_md(worst[["gate", "inputs", "corner", "vdd", "temp", "n_vth0", "p_vth0",
                        spice_col, ml_col, "err_pct"]]), ""]
    (C.RESULTS_DIR / f"gate_results_table{suffix}.md").write_text("\n".join(md), encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n-> {C.RESULTS_DIR / f'gate_results_table{suffix}.md'} and plots in {C.PLOTS_DIR}")


if __name__ == "__main__":
    main()
