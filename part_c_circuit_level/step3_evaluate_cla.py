"""Step 3 (spec Step 8): accuracy of the hierarchical ML leakage against the CLA SPICE netlist.

    python step3_evaluate_cla.py

Outputs (results/)
    cla_results_table.md / .csv     Deliverable 8: the same metrics as the gate results
                                    table (SPICE median, ML median, bias, median error,
                                    p95 error, mean accuracy), with the Part B gate-level
                                    accuracy beside it for comparison
    cla_per_vector.csv              the same metrics per input vector (over the corners)
    plots/cla_parity.png            Deliverable 8: ML vs SPICE leakage, log axes
    plots/cla_error_vs_corner.png   |error| vs temperature, VDD, vth0, toxe (report: error analysis)
"""
import pandas as pd

from cla import config as CC
from gatelib import config as C
from gatelib.report import error_vs_corner_plot, leak_metrics, metrics_table, parity_plot, signed_err_pct, to_md

# the same columns as Part B's gate results table
COLS = ["spice_median_A", "ml_median_A", "bias_pct", "median_err_pct", "p95_err_pct", "mean_accuracy_pct"]


def main():
    if not CC.CLA_RESULTS_CSV.exists():
        raise SystemExit(f"{CC.CLA_RESULTS_CSV} not found; run step2_ml_cla.py first.")
    df = pd.read_csv(CC.CLA_RESULTS_CSV, dtype={"inputs": str})
    CC.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Deliverable 8, with the gate-level accuracy beside it (report: compare with gate level)
    rows = [{"level": f"3-bit CLA, hierarchical ML ({len(df)} runs)", **leak_metrics(df["spice_ileak"], df["ml_ileak"])}]
    gate_tab = C.RESULTS_DIR / "gate_results_table.csv"
    if gate_tab.exists():
        g = pd.read_csv(gate_tab, dtype={"inputs": str})
        g = g[g["gate"] == "ALL"].iloc[0]
        rows.append({"level": "Gate level (Part B, all 18 cases)", **{c: g[c] for c in COLS if c in g}})
    table = pd.DataFrame(rows)[["level", *COLS]]
    table.to_csv(CC.RESULTS_DIR / "cla_results_table.csv", index=False)

    df["err_pct"] = signed_err_pct(df["spice_ileak"], df["ml_ileak"])
    per_vec = metrics_table(df, ["vec", "a", "b", "cin", "inputs"], "spice_ileak", "ml_ileak")
    per_vec.to_csv(CC.RESULTS_DIR / "cla_per_vector.csv", index=False)
    per_cin = metrics_table(df, ["cin"], "spice_ileak", "ml_ileak")[["cin", *COLS]]
    composition = pd.DataFrame([leak_metrics(df["spice_ileak"], df["spice_gatesum"])])[COLS]
    composition = composition.rename(columns={"ml_median_A": "gatesum_median_A"})
    shares = df.filter(like="gate_").filter(like="_share").mean().rename("mean share of ML leakage")
    shares.index = [s[len("gate_"):-len("_share")] for s in shares.index]

    parity_plot(df["spice_ileak"], df["ml_ileak"], ["CLA"] * len(df),
                f"3-bit CLA leakage, ML vs full-netlist SPICE ({len(df)} runs)", CC.PLOTS_DIR / "cla_parity.png")
    error_vs_corner_plot(df, "err_pct", ["CLA"] * len(df), "3-bit CLA leakage |error| vs corner",
                         CC.PLOTS_DIR / "cla_error_vs_corner.png")

    cols = ["vec", "a", "b", "cin", "spice_median_A", "ml_median_A", "bias_pct", "median_err_pct", "p95_err_pct",
            "mean_accuracy_pct"]
    worst = per_vec.sort_values("mean_accuracy_pct")
    md = ["# 3-bit CLA results: total leakage (supply power / VDD), ML vs full-netlist SPICE\n",
          "Bias = mean signed relative error (ML - SPICE) / SPICE. Errors are |relative errors| in %. "
          "Mean accuracy = 100 - mean |relative error|.\n",
          to_md(table), "",
          "## Error analysis\n",
          "### Composition error: sum of the gate-level SPICE leakages vs the full-netlist SPICE\n",
          "This separates the error of building the circuit from gates (loading, gate inputs not exactly "
          "at 0 or VDD) from the ML error.\n",
          to_md(composition), "",
          "### By carry-in\n", to_md(per_cin), "",
          "### Share of the ML leakage by gate type\n",
          to_md(shares.reset_index().rename(columns={"index": "gate"})), "",
          "### 10 input vectors with the lowest mean accuracy\n", to_md(worst.head(10)[cols]), "",
          "### 10 input vectors with the highest mean accuracy\n", to_md(worst.tail(10)[cols]), ""]
    (CC.RESULTS_DIR / "cla_results_table.md").write_text("\n".join(md), encoding="utf-8")

    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=fmt))
        print("\nComposition check (SPICE gate sum vs full netlist):")
        print(composition.to_string(index=False, float_format=fmt))
    print(f"\n-> {CC.RESULTS_DIR / 'cla_results_table.md'} and plots in {CC.PLOTS_DIR}")


if __name__ == "__main__":
    main()
