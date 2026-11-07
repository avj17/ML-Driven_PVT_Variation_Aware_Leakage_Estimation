"""Where does the chosen model's error come from? (TRAIN and VAL only; test unused.)

    python analyze_errors.py            # after evaluate.py

For every case and terminal current it breaks the relative error down by
operating region and checks how clean the SPICE labels are. Output:
results/error_analysis.md
"""
import numpy as np
import pandas as pd

from leakage_ml import config as C
from leakage_ml.data import build_features
from leakage_ml.metrics import relative_error
from leakage_ml.predictor import LeakagePredictor
from train import prepare_case


def binned(err, key, edges, labels):
    """Median / p95 error and sample share per bin of `key`."""
    b = pd.cut(key, edges, labels=labels, include_lowest=True)
    g = pd.DataFrame({"bin": b, "err": err}).groupby("bin", observed=False)["err"]
    out = pd.DataFrame({"share_%": 100 * g.size() / len(err), "median_%": g.median(),
                        "p95_%": g.quantile(0.95)})
    return out.round(2)


def analyse(case):
    pred = LeakagePredictor.load(C.MODELS_DIR / f"{case}.joblib")
    d = prepare_case(case, pred.feature_set)
    F = build_features(d["df"], pred.mos, "physics")          # every feature, for binning
    X = d["X"][pred.feature_cols]
    lines = [f"\n## {case} ({pred.family})\n"]

    parts = {}
    for split in ("tr", "va"):
        idx = d[split]
        I = d["I"][idx]
        Y = d["Y"][idx]
        Y_hat = pred.model.predict(X.iloc[idx])
        I_hat = d["tf"].inverse_transform(Y_hat)
        parts[split] = (idx, I, Y, Y_hat, I_hat)

    idx, I, Y, Y_hat, I_hat = parts["va"]
    f = F.iloc[idx].reset_index(drop=True)
    kcl = np.abs(I.sum(axis=1))
    ch = (I[:, 0] - I[:, 2]) / 2            # antisymmetric (channel) part of Id/Is
    lk = (I[:, 0] + I[:, 2]) / 2            # symmetric part = -(Ig + Ib)/2 by KCL
    dom = np.abs(ch) / (np.abs(ch) + np.abs(lk))

    # 1. Per-target overview: fit (train) vs generalisation (val), target range, label noise.
    rows = []
    for k, t in enumerate(C.TARGETS):
        e_va = relative_error(I[:, k], I_hat[:, k])
        e_tr = relative_error(parts["tr"][1][:, k], parts["tr"][4][:, k])
        sq = (Y[:, k] - Y_hat[:, k]) ** 2
        top = np.sort(sq)[::-1]
        rows.append({
            "target": t, "train_median_%": np.median(e_tr), "val_median_%": np.median(e_va),
            "val_p95_%": np.percentile(e_va, 95), "sign_err_%": 100 * np.mean(np.sign(I[:, k]) != np.sign(I_hat[:, k])),
            "decades_p1_p99": np.log10(np.percentile(np.abs(I[:, k]), 99) / np.percentile(np.abs(I[I[:, k] != 0, k]), 1)),
            "kcl/|I|_median_%": 100 * np.median(kcl / np.abs(I[:, k])),
            "kcl/|I|>1%_share": 100 * np.mean(kcl / np.abs(I[:, k]) > 0.01),
            "MSE_share_top1%": 100 * top[:max(1, len(top) // 100)].sum() / top.sum(),
        })
    lines.append("### Per current (VAL)\n\n" + pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")

    # 2. Region breakdown for Id and Is (the weak currents) and Ig for reference.
    for k, t in [(0, "id"), (2, "is"), (1, "ig")]:
        err = relative_error(I[:, k], I_hat[:, k])
        lines.append(f"### {t}: error by region (VAL)\n")
        lines.append("Channel dominance |Ich|/(|Ich|+|Ileak|), Ich=(Id-Is)/2, Ileak=(Id+Is)/2:\n\n"
                     + binned(err, dom, [0, 0.5, 0.9, 0.99, 0.999, 1], ["<0.5", "0.5-0.9", "0.9-0.99", "0.99-0.999", ">0.999"]).to_markdown() + "\n")
        lines.append("|Vds| (V):\n\n" + binned(err, f["vds_abs"], [0, 0.005, 0.026, 0.1, 0.3, 1.0],
                                               ["<5mV", "5-26mV", "26-100mV", "0.1-0.3", ">0.3"]).to_markdown() + "\n")
        lines.append("(Vgs_eff - Vth0)/V_T (subthreshold depth):\n\n"
                     + binned(err, f["vov_over_vt"], [-60, -15, -10, -5, 0, 5, 60],
                              ["<-15", "-15..-10", "-10..-5", "-5..0", "0..5", ">5"]).to_markdown() + "\n")
        lines.append("Temperature (C):\n\n" + binned(err, f["temp_c"], [-41, 0, 40, 85, 126],
                                                  ["-40..0", "0..40", "40..85", "85..125"]).to_markdown() + "\n")
        q = np.percentile(np.abs(I[:, k]), [0, 10, 50, 90, 100])
        lines.append("|I| quantile:\n\n" + binned(err, np.abs(I[:, k]), q, ["<p10", "p10-p50", "p50-p90", ">p90"]).to_markdown() + "\n")
    return "\n".join(lines)


def main():
    out = ["# Error analysis of the chosen models (TRAIN/VAL only)\n"]
    for case in C.CASES:
        if (C.MODELS_DIR / f"{case}.joblib").exists():
            txt = analyse(case)
            print(txt)
            out.append(txt)
    (C.RESULTS_DIR / "error_analysis.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
