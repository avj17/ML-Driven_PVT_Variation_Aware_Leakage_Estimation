"""Step 2: compare candidates on VALIDATION, pick one family per case, then
evaluate the chosen models ONCE on the TEST set.

    python evaluate.py

Outputs (results/)
    model_comparison_val.csv / .md      Deliverable 4: 4 families x 4 cases (+ mean-predictor baseline)
    model_comparison_val_per_target.csv every terminal, every family
    feature_ablation_val.md             physics vs raw features (if train.py --feature-set raw was run)
    chosen_models_test.csv / .md        Deliverable 5: test median / p95 error of the chosen models
    plots/<case>_parity_test.png        Deliverable 5: predicted vs SPICE per case
    plots/<case>_error_hist_test.png, <case>_importance.png, <case>_mlp_learning_curve.png
saved_models/<case>.joblib            Deliverable 5: the four final model files
"""
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from leakage_ml import config as C
from leakage_ml import plots
from leakage_ml.metrics import summarize
from leakage_ml.models import mlp_members
from leakage_ml.predictor import LeakagePredictor
from train import prepare_case


def load_candidates():
    rows = []
    for p in sorted((C.RESULTS_DIR / "candidates").glob("*.json")):
        r = json.loads(p.read_text())
        for t, m in r["val"].items():
            rows.append({"case": r["case"], "family": r["family"], "feature_set": r["feature_set"],
                         "quick": r["quick"], "target": t, **m, "fit_seconds": r["fit_seconds"],
                         "train_r2_log": r["train"][t]["r2_log"],
                         "train_median_err_pct": r["train"][t]["median_err_pct"]})
    return pd.DataFrame(rows)


def baseline_rows(d, case):
    """Trivial baseline: predict the training mean of the signed-log target."""
    Y_hat = np.tile(d["Y"][d["tr"]].mean(axis=0), (len(d["va"]), 1))
    I_hat = d["tf"].inverse_transform(Y_hat)
    I_va, Y_va = d["I"][d["va"]], d["Y"][d["va"]]
    return [{"case": case, "family": "baseline_mean", "feature_set": "-", "target": t,
             **summarize(I_va[:, k], I_hat[:, k], Y_va[:, k], Y_hat[:, k])}
            for k, t in enumerate(C.TARGETS)]


def summary_table(per_target):
    """Collapse the four terminals: mean R^2, mean median error, mean p95 error."""
    g = per_target.groupby(["case", "family"], sort=False)
    out = g.agg(r2_log=("r2_log", "mean"), median_err_pct=("median_err_pct", "mean"),
                p95_err_pct=("p95_err_pct", "mean"), worst_p95_pct=("p95_err_pct", "max"),
                train_median_err_pct=("train_median_err_pct", "mean"))
    return out.reset_index()


class _Fitted:
    """permutation_importance requires a fit() method but never calls it on an
    already-fitted model; this adapter gives our wrappers one."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.model.predict(X)


def feature_importance(pred, X, Y, n_repeats=5):
    """sklearn permutation importance on VALIDATION: mean increase in signed-log
    MSE (averaged over the four terminals) when one feature is shuffled. Unlike
    impurity importance it is not biased toward high-cardinality features."""
    scorer = lambda est, X_, Y_: -float(np.mean((est.predict(X_) - Y_) ** 2))
    res = permutation_importance(_Fitted(pred.model), X, Y, scoring=scorer, n_repeats=n_repeats,
                                 random_state=C.SEED_CV)
    return pd.Series(res.importances_mean, index=X.columns)


def main():
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cands = load_candidates()
    if cands.empty:
        raise SystemExit("No candidates found; run train.py first.")
    if cands["quick"].any():
        print("WARNING: some candidates were trained with --quick budgets.")

    main_c = cands[cands["feature_set"] == "physics"]
    if main_c.empty:
        raise SystemExit("No physics-feature candidates found; check the train_<case>.log files.")
    cases = [c for c in C.CASES if c in set(main_c["case"])]
    prepared = {c: prepare_case(c, "physics") for c in cases}
    base = pd.DataFrame([r for c in cases for r in baseline_rows(prepared[c], c)])

    per_target = pd.concat([main_c, base], ignore_index=True)
    per_target.to_csv(C.RESULTS_DIR / "model_comparison_val_per_target.csv", index=False)
    summ = summary_table(per_target)
    summ.to_csv(C.RESULTS_DIR / "model_comparison_val.csv", index=False)
    (C.RESULTS_DIR / "model_comparison_val.md").write_text(
        "# Model comparison (validation set, mean over Id/Ig/Is/Ib)\n\n"
        + summ.to_markdown(index=False, floatfmt=".4g") + "\n\n## Per terminal\n\n"
        + per_target[["case", "family", "target", "r2_log", "median_err_pct", "p95_err_pct",
                      "sign_acc_pct", "train_r2_log", "train_median_err_pct"]].to_markdown(index=False, floatfmt=".4g") + "\n")
    print(summ.to_string(index=False))

    # Feature-engineering ablation (A2-style): same family, physics vs raw features.
    raw = cands[cands["feature_set"] == "raw"]
    if not raw.empty:
        abl = summary_table(pd.concat([main_c.assign(family=main_c["family"] + " (physics)"),
                                       raw.assign(family=raw["family"] + " (raw)")]))
        (C.RESULTS_DIR / "feature_ablation_val.md").write_text(
            "# Feature ablation (validation)\n\n" + abl.to_markdown(index=False, floatfmt=".4g") + "\n")

    # Selection on VALIDATION: lowest mean median error, tie-break on mean p95.
    chosen_rows = []
    for case in cases:
        s = summ[(summ["case"] == case) & (summ["family"] != "baseline_mean")]
        best = s.sort_values(["median_err_pct", "p95_err_pct"]).iloc[0]
        fam = best["family"]
        src = C.MODELS_DIR / "candidates" / f"{case}__{fam}.joblib"
        shutil.copy(src, C.MODELS_DIR / f"{case}.joblib")
        pred = LeakagePredictor.load(src)
        d = prepared[case]

        # ---- the only place the TEST split is used ----
        X_te = d["X"].iloc[d["te"]][pred.feature_cols]
        Y_hat = pred.model.predict(X_te)
        I_hat = pred.target_tf.inverse_transform(Y_hat)
        I_te, Y_te = d["I"][d["te"]], d["Y"][d["te"]]
        errs = {}
        for k, t in enumerate(C.TARGETS):
            m = summarize(I_te[:, k], I_hat[:, k], Y_te[:, k], Y_hat[:, k])
            errs[t] = 100 * np.abs(I_hat[:, k] - I_te[:, k]) / np.abs(I_te[:, k])
            chosen_rows.append({"case": case, "chosen_family": fam, "target": t, **m})
        plots.parity_plot(I_te, I_hat, case, fam, "test", C.PLOTS_DIR / f"{case}_parity_test.png")
        plots.error_hist(errs, case, "test", C.PLOTS_DIR / f"{case}_error_hist_test.png")

        X_va = d["X"].iloc[d["va"]][pred.feature_cols]
        imp = feature_importance(pred, X_va, d["Y"][d["va"]])
        imp.sort_values(ascending=False).to_csv(C.RESULTS_DIR / f"{case}_permutation_importance.csv")
        plots.importance_bar(imp, case, fam, C.PLOTS_DIR / f"{case}_importance.png")

        mlp_path = C.MODELS_DIR / "candidates" / f"{case}__mlp.joblib"
        if mlp_path.exists():
            members = mlp_members(LeakagePredictor.load(mlp_path).model)
            if members:      # first ensemble member's curve
                plots.learning_curve(members[0].history, case, C.PLOTS_DIR / f"{case}_mlp_learning_curve.png",
                                     loss_name="Huber" if getattr(members[0], "loss", "mse") == "huber" else "MSE")
        print(f"{case}: chosen {fam} (val median {best['median_err_pct']:.3f}%, p95 {best['p95_err_pct']:.2f}%)")

    test = pd.DataFrame(chosen_rows)
    test.to_csv(C.RESULTS_DIR / "chosen_models_test.csv", index=False)
    (C.RESULTS_DIR / "chosen_models_test.md").write_text(
        "# Chosen models on the TEST set\n\n"
        + test[["case", "chosen_family", "target", "r2_log", "median_err_pct", "p95_err_pct",
                "mean_err_pct", "sign_acc_pct"]].to_markdown(index=False, floatfmt=".4g") + "\n")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
