"""Step 3 (spec Step 5): accuracy of the ML gate leakage against SPICE.

    python step3_evaluate_gates.py

Outputs (results/)
    gate_results_table.md / .csv     Deliverable 6: per gate and input case, the SPICE
                                     median, ML median, bias, median error, p95 error,
                                     plus mean accuracy (the spec's 99 % target)
    node_voltages.csv                SPICE vs ML internal node voltages (spec Section 5:
                                     "the internal node voltages from both")
    plots/gate_parity.png            Deliverable 7: ML vs SPICE, all runs, log axes
    plots/gate_error_vs_corner.png   |error| vs temperature, VDD, vth0, toxe (report: error analysis)
"""
import pandas as pd

from gatelib import config as C
from gatelib.gates import GATES
from gatelib.report import error_vs_corner_plot, leak_metrics, metrics_table, parity_plot, signed_err_pct, to_md

SPICE_COL, ML_COL = "spice_ileak", "ml_ileak"
# Deliverable 6 columns, plus mean accuracy for the spec's "99 % mean accuracy" target
COLS = ["gate", "inputs", "spice_median_A", "ml_median_A", "bias_pct",
        "median_err_pct", "p95_err_pct", "mean_accuracy_pct"]
SUMMARY_COLS = ["gate", "median_err_pct", "p95_err_pct", "mean_accuracy_pct"]


def results_table(df):
    """Per (gate, inputs) rows, then one row per gate and an overall row."""
    case = metrics_table(df, ["gate", "inputs"], SPICE_COL, ML_COL)
    per_gate = metrics_table(df, ["gate"], SPICE_COL, ML_COL).assign(inputs="all")
    overall = pd.DataFrame([{"gate": "ALL", "inputs": "all", **leak_metrics(df[SPICE_COL], df[ML_COL])}])
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
    if not C.GATE_RESULTS_CSV.exists():
        raise SystemExit(f"{C.GATE_RESULTS_CSV} not found; run step2_ml_gates.py first.")
    df = pd.read_csv(C.GATE_RESULTS_CSV, dtype={"inputs": str})
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df["err_pct"] = signed_err_pct(df[SPICE_COL], df[ML_COL])

    table = results_table(df)
    table.to_csv(C.RESULTS_DIR / "gate_results_table.csv", index=False)
    summary = table[table["inputs"] == "all"][SUMMARY_COLS]
    md = ["# Gate results: total leakage (supply power / VDD), ML vs SPICE\n",
          "ML uses the internal node voltages pre-recorded by the gate-level SPICE runs (spec Step 4).",
          "Bias = mean signed relative error (ML - SPICE) / SPICE. Errors are |relative errors| in %.",
          "Mean accuracy = 100 - mean |relative error|.\n",
          to_md(table), "",
          "## Summary\n", to_md(summary), ""]

    parity_plot(df[SPICE_COL], df[ML_COL], df["gate"],
                f"Gate leakage, ML vs SPICE ({len(df)} runs)", C.PLOTS_DIR / "gate_parity.png")
    error_vs_corner_plot(df, "err_pct", df["gate"], "Gate leakage |error| vs corner",
                         C.PLOTS_DIR / "gate_error_vs_corner.png")

    if any(c.startswith("ml_v_") for c in df.columns):
        nodes = node_table(df)
        nodes.to_csv(C.RESULTS_DIR / "node_voltages.csv", index=False)
        md += ["## Internal node voltages: SPICE vs ML\n", to_md(nodes), ""]

    worst = df.reindex(df["err_pct"].abs().sort_values(ascending=False).index).head(15)
    md += ["## 15 largest errors\n",
           to_md(worst[["gate", "inputs", "corner", "vdd", "temp", "n_vth0", "p_vth0",
                        SPICE_COL, ML_COL, "err_pct"]]), ""]
    (C.RESULTS_DIR / "gate_results_table.md").write_text("\n".join(md), encoding="utf-8")

    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=fmt))
        print("\nSummary")
        print(summary.to_string(index=False, float_format=fmt))
    print(f"\n-> {C.RESULTS_DIR / 'gate_results_table.md'} and plots in {C.PLOTS_DIR}")


if __name__ == "__main__":
    main()
