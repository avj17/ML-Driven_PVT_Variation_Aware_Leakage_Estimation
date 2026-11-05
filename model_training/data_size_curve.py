"""Step 3: dataset-size learning curve for the chosen model of each case.

The spec asks for high accuracy "preferably with the smallest dataset
possible". The chosen family is refit with its already-tuned hyperparameters
on growing random fractions of TRAIN and scored on VALIDATION (test unused).

    python data_size_curve.py          # after evaluate.py

Outputs: results/data_size_curve.csv, results/plots/data_size_curve.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from leakage_ml import config as C
from leakage_ml.metrics import summarize
from leakage_ml.models import PerTargetRegressor, TargetAdapter
from leakage_ml.predictor import LeakagePredictor
from train import prepare_case


def refit(model, X, Y):
    if isinstance(model, TargetAdapter):          # MLP: same recipe, ensemble size and target transform
        return model.refit(X, Y)
    return PerTargetRegressor([clone(est).fit(X, Y[:, k]) for k, est in enumerate(model.estimators)])


def case_curve(case):
    """All training-set sizes for one case. At 100 % the final model itself is
    scored (it was trained on all of TRAIN with the same recipe)."""
    import torch
    torch.set_num_threads(2)          # four cases share the node's CPUs
    rows = []
    pred = LeakagePredictor.load(C.MODELS_DIR / f"{case}.joblib")
    d = prepare_case(case, pred.feature_set)
    X = d["X"][pred.feature_cols].to_numpy()
    X_va, I_va, Y_va = X[d["va"]], d["I"][d["va"]], d["Y"][d["va"]]
    for frac in C.SIZE_FRACTIONS:
        if frac == 1.0:
            sub, model = d["tr"], pred.model
        else:
            sub = train_test_split(d["tr"], train_size=frac, random_state=C.SEED_SPLIT, shuffle=True)[0]
            model = refit(pred.model, X[sub], d["Y"][sub])
        Y_hat = model.predict(X_va)
        I_hat = pred.target_tf.inverse_transform(Y_hat)
        for k, t in enumerate(C.TARGETS):
            m = summarize(I_va[:, k], I_hat[:, k], Y_va[:, k], Y_hat[:, k])
            rows.append({"case": case, "family": pred.family, "train_frac": frac,
                         "n_train": len(sub), "target": t, **m})
        last = pd.DataFrame(rows[-4:])
        print(f"{case} {pred.family} n_train={len(sub)}: mean median err "
              f"{last['median_err_pct'].mean():.3f}%  mean p95 {last['p95_err_pct'].mean():.2f}%", flush=True)
    return rows


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    cases = [c for c in C.CASES if (C.MODELS_DIR / f"{c}.joblib").exists()]
    # One process per case on the shared GPU ("spawn": CUDA cannot be forked).
    with ProcessPoolExecutor(len(cases), mp_context=mp.get_context("spawn")) as ex:
        rows = [r for case_rows in ex.map(case_curve, cases) for r in case_rows]

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "data_size_curve.csv", index=False)
    agg = df.groupby(["case", "n_train"], as_index=False)[["median_err_pct", "p95_err_pct"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for case, g in agg.groupby("case"):
        axes[0].plot(g["n_train"], g["median_err_pct"], marker="o", label=case)
        axes[1].plot(g["n_train"], g["p95_err_pct"], marker="o", label=case)
    for ax, name in zip(axes, ["median", "95th-percentile"]):
        ax.set_xlabel("training samples"); ax.set_ylabel(f"{name} relative error (%), mean over terminals")
        ax.set_yscale("log"); ax.set_title(f"Validation {name} error vs training-set size")
        ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "data_size_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
