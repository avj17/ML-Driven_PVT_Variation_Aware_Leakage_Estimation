"""MLP improvement study: which training recipe fixes the weak Id/Is predictions?

    python mlp_experiments.py --cases NMOS_OFF        # one case (run cases in parallel)
    python mlp_experiments.py --quick                 # smoke test

One architecture (config 1: 256-256-256 SiLU) is trained with every
combination of
    target   direct (Id, Ig, Is, Ib) | kcl (A, Ig, Ib) | kcl_sign (|A| + imposed sign)
    loss     mse | huber
    monitor  early stopping on the hold-out loss | on the spec metric
Row "direct / mse / loss" is the current method (the baseline).
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
from leakage_ml.metrics import summarize
from leakage_ml.models import MLP_TARGETS, TargetAdapter, _train_member, mlp_config
from train import prepare_case

GRID = {"target": ["direct", "kcl", "kcl_sign"], "loss": ["mse", "huber"], "monitor": ["loss", "metric"]}


def run_case(case, space, log=print):
    d = prepare_case(case, "physics")
    X_df = d["X"].iloc[d["tr"]]
    cols, X = list(X_df.columns), X_df.to_numpy(dtype=float)
    I = d["I"][d["tr"]]
    X_va, I_va, Y_va = d["X"].iloc[d["va"]].to_numpy(dtype=float), d["I"][d["va"]], d["Y"][d["va"]]
    rows = []
    for target, loss, monitor in itertools.product(*GRID.values()):
        cfg = mlp_config({**space["configs"][0], "target": target, "loss": loss, "monitor": monitor}, space, cols)
        inner_tf = MLP_TARGETS[target]().fit(I)
        t0 = time.perf_counter()
        m, inner = _train_member(X, I, inner_tf.transform(I), cfg, space, inner_tf, 0)
        model = TargetAdapter(m, inner_tf, d["tf"], cfg, space)
        Y_hat = model.predict(X_va)
        I_hat = d["tf"].inverse_transform(Y_hat)
        val = {t: summarize(I_va[:, k], I_hat[:, k], Y_va[:, k], Y_hat[:, k]) for k, t in enumerate(C.TARGETS)}
        Y_hat_tr = model.predict(X)
        tr_med = [float(np.median(100 * np.abs(d["tf"].inverse_transform(Y_hat_tr)[:, k] - I[:, k]) / np.abs(I[:, k])))
                  for k in range(4)]
        row = {"case": case, "target": target, "loss": loss, "monitor": monitor,
               "inner_median": inner["median"], "inner_p95": inner["p95"],
               "val_median": float(np.mean([v["median_err_pct"] for v in val.values()])),
               "val_p95": float(np.mean([v["p95_err_pct"] for v in val.values()])),
               "val_worst_p95": float(max(v["p95_err_pct"] for v in val.values())),
               **{f"val_med_{t}": val[t]["median_err_pct"] for t in C.TARGETS},
               **{f"val_p95_{t}": val[t]["p95_err_pct"] for t in C.TARGETS},
               **{f"val_sign_{t}": val[t]["sign_acc_pct"] for t in C.TARGETS},
               "train_median": float(np.mean(tr_med)),
               "epochs": m.epochs_, "best_epoch": m.best_epoch_, "seconds": round(time.perf_counter() - t0, 1)}
        log(f"{case} {target:8s} {loss:5s} {monitor:6s} | inner {inner['median']:.3f}% | val median "
            f"{row['val_median']:.3f}% p95 {row['val_p95']:.2f}% | id {row['val_med_id']:.2f} is "
            f"{row['val_med_is']:.2f} | {row['epochs']} ep {row['seconds']:.0f}s")
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
    out = C.RESULTS_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        rows = run_case(case, space)
        (out / f"mlp_{case}.json").write_text(json.dumps(rows, indent=2))
        df = pd.DataFrame(rows)
        cols = ["target", "loss", "monitor", "inner_median", "val_median", "val_p95", "val_worst_p95",
                "val_med_id", "val_med_ig", "val_med_is", "val_med_ib", "train_median", "epochs"]
        (out / f"mlp_{case}.md").write_text(f"# MLP recipe study: {case}\n\n"
                                            + df[cols].to_markdown(index=False, floatfmt=".3f") + "\n")


if __name__ == "__main__":
    main()
