"""MLP recipe studies: which training recipe gives the most accurate currents?

    python mlp_experiments.py --cases NMOS_OFF        # one case (run cases in parallel)
    python mlp_experiments.py --quick                 # smoke test

One architecture (config 1: 256-256-256 SiLU) is trained with each recipe of a
study; the study is picked by the dataset (LEAKAGE_DATASET):

  v2 study ("orig" data): every combination of
      target   direct (Id, Ig, Is, Ib) | kcl (A, Ig, Ib) | kcl_sign (|A| + imposed sign)
      loss     mse | huber
      monitor  early stopping on the hold-out loss | on the spec metric (median)
  v3 study ("new" data, target = MEAN relative error < 1 %): the v2 recipe, then
      early stopping on the mean error, kappa-weighted loss and a learned sign of A.

Each run is scored on the inner hold-out (part of TRAIN, used to choose the
recipe) and on VAL (reported only). The test set is not used.

Output: results/experiments/mlp_<case>.json and .md
"""
import argparse
import copy
import itertools
import json
import time

import numpy as np
import pandas as pd

from leakage_ml import config as C
from leakage_ml.metrics import relative_error, summarize
from leakage_ml.models import MLP_TARGETS, TargetAdapter, _train_member, mlp_config
from train import prepare_case

V2_GRID = [dict(zip(("target", "loss", "monitor"), r)) for r in
           itertools.product(["direct", "kcl", "kcl_sign"], ["mse", "huber"], ["loss", "metric"])]
V3_RECIPES = [
    {"target": "kcl_sign", "loss": "huber", "monitor": "median", "weight": "none"},   # v2 recipe
    {"target": "kcl_sign", "loss": "huber", "monitor": "mean", "weight": "none"},
    {"target": "kcl_sign", "loss": "huber", "monitor": "mean", "weight": "kappa"},
    {"target": "kcl", "loss": "huber", "monitor": "mean", "weight": "none"},
    {"target": "kcl", "loss": "huber", "monitor": "mean", "weight": "kappa"},
]
STUDIES = {"orig": V2_GRID, "new": V3_RECIPES}


def run_case(case, space, recipes, log=print):
    d = prepare_case(case, "physics")
    X_df = d["X"].iloc[d["tr"]]
    cols, X = list(X_df.columns), X_df.to_numpy(dtype=float)
    I = d["I"][d["tr"]]
    X_va, I_va, Y_va = d["X"].iloc[d["va"]].to_numpy(dtype=float), d["I"][d["va"]], d["Y"][d["va"]]
    log(f"{case}: train {len(X)} rows, val {len(X_va)}")
    rows = []
    for recipe in recipes:
        cfg = mlp_config({**space["configs"][0], **recipe}, space, cols)
        inner_tf = MLP_TARGETS[cfg["target"]]().fit(I)
        t0 = time.perf_counter()
        m, inner = _train_member(X, I, inner_tf.transform(I), cfg, space, inner_tf, 0)
        model = TargetAdapter(m, inner_tf, d["tf"], cfg, space)
        Y_hat = model.predict(X_va)
        I_hat = d["tf"].inverse_transform(Y_hat)
        val = {t: summarize(I_va[:, k], I_hat[:, k], Y_va[:, k], Y_hat[:, k]) for k, t in enumerate(C.TARGETS)}
        Y_hat_tr = model.predict(X)
        I_hat_tr = d["tf"].inverse_transform(Y_hat_tr)
        tr_med = [float(np.nanmedian(relative_error(I[:, k], I_hat_tr[:, k]))) for k in range(4)]
        row = {"case": case, **recipe,
               "inner_median": inner["median"], "inner_p95": inner["p95"], "inner_mean": inner["mean"],
               "val_median": float(np.mean([v["median_err_pct"] for v in val.values()])),
               "val_p95": float(np.mean([v["p95_err_pct"] for v in val.values()])),
               "val_mean": float(np.mean([v["mean_err_pct"] for v in val.values()])),
               "val_worst_p95": float(max(v["p95_err_pct"] for v in val.values())),
               **{f"val_med_{t}": val[t]["median_err_pct"] for t in C.TARGETS},
               **{f"val_p95_{t}": val[t]["p95_err_pct"] for t in C.TARGETS},
               **{f"val_mean_{t}": val[t]["mean_err_pct"] for t in C.TARGETS},
               **{f"val_sign_{t}": val[t]["sign_acc_pct"] for t in C.TARGETS},
               "train_median": float(np.mean(tr_med)),
               "epochs": m.epochs_, "best_epoch": m.best_epoch_, "seconds": round(time.perf_counter() - t0, 1)}
        log(f"{case} {' '.join(str(v) for v in recipe.values()):32s} | inner median {inner['median']:.3f}% "
            f"mean {inner['mean']:.3f}% | val median {row['val_median']:.3f}% mean {row['val_mean']:.3f}% "
            f"p95 {row['val_p95']:.2f}% | {row['epochs']} ep {row['seconds']:.0f}s")
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(C.CASES))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    space = copy.deepcopy(C.SEARCH_SPACES["mlp"])
    if args.quick:
        space.update(max_epochs=20, patience=5)
    recipes = STUDIES[C.DATASET]
    out = C.RESULTS_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        rows = run_case(case, space, recipes)
        (out / f"mlp_{case}.json").write_text(json.dumps(rows, indent=2))
        df = pd.DataFrame(rows)
        cols = (list(recipes[0]) + ["inner_median", "inner_mean", "val_median", "val_mean", "val_p95",
                                    "val_worst_p95"] + [f"val_mean_{t}" for t in C.TARGETS]
                + ["train_median", "epochs"])
        (out / f"mlp_{case}.md").write_text(f"# MLP recipe study: {case} ({C.DATASET} data)\n\n"
                                            + df[cols].to_markdown(index=False, floatfmt=".3f") + "\n")


if __name__ == "__main__":
    main()
