"""Step 1: tune and fit every (case, model family) candidate on the TRAINING set
and score it on the VALIDATION set. The test set is not touched here.

    python train.py                                   # all 4 cases x 4 families
    python train.py --cases NMOS_OFF --models xgboost mlp
    python train.py --feature-set raw --models xgboost mlp   # ablation run
    python train.py --quick                           # tiny budgets, smoke test

Outputs
    results/candidates/<case>__<family>[__raw].json   val metrics + chosen hyperparameters
    saved_models/candidates/<case>__<family>[__raw].joblib
"""
import argparse
import copy
import json
import platform
import time

import numpy as np

from leakage_ml import config as C
from leakage_ml.data import build_features, drop_uninformative, load_case, split_indices
from leakage_ml.metrics import summarize
from leakage_ml.models import FITTERS
from leakage_ml.predictor import LeakagePredictor
from leakage_ml.transforms import SignedLogTransform


def prepare_case(case, feature_set):
    """Features, split and fitted target transform for one case (all fitted on TRAIN)."""
    df = load_case(case)
    X = build_features(df, C.CASES[case]["mos"], feature_set)
    I = df[C.TARGETS].to_numpy(dtype=float)
    tr, va, te = split_indices(len(df), case)
    drop = drop_uninformative(X.iloc[tr])
    X = X.drop(columns=drop)
    tf = SignedLogTransform().fit(I[tr])
    return {"df": df, "X": X, "I": I, "Y": tf.transform(I), "tf": tf, "drop": drop,
            "tr": tr, "va": va, "te": te}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(C.CASES))
    ap.add_argument("--models", nargs="+", default=C.MODEL_FAMILIES, choices=C.MODEL_FAMILIES)
    ap.add_argument("--feature-set", default="physics", choices=["physics", "raw"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="tiny search budgets for a smoke test")
    args = ap.parse_args()

    spaces = copy.deepcopy(C.SEARCH_SPACES)
    if args.quick:
        for fam, over in C.QUICK_OVERRIDES.items():
            spaces[fam].update(over)
    suffix = "" if args.feature_set == "physics" else f"__{args.feature_set}"
    cand_res = C.RESULTS_DIR / "candidates"
    cand_mod = C.MODELS_DIR / "candidates"
    cand_res.mkdir(parents=True, exist_ok=True)
    cand_mod.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        d = prepare_case(case, args.feature_set)
        X_tr, Y_tr = d["X"].iloc[d["tr"]], d["Y"][d["tr"]]
        X_va = d["X"].iloc[d["va"]]
        print(f"\n=== {case}: train={len(d['tr'])} val={len(d['va'])} test={len(d['te'])} "
              f"features={d['X'].shape[1]} dropped={d['drop']} I0={np.array2string(d['tf'].i0_, precision=2)}")

        for fam in args.models:
            print(f"  -> {fam}")
            t0 = time.perf_counter()
            model, tuning = FITTERS[fam](X_tr, Y_tr, spaces[fam], n_jobs=args.n_jobs)
            fit_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            Y_hat = model.predict(X_va)
            pred_ms = 1e3 * (time.perf_counter() - t1) / len(X_va)
            I_hat = d["tf"].inverse_transform(Y_hat)
            I_va, Y_va = d["I"][d["va"]], d["Y"][d["va"]]
            per_target = {t: summarize(I_va[:, k], I_hat[:, k], Y_va[:, k], Y_hat[:, k])
                          for k, t in enumerate(C.TARGETS)}
            # Same metrics on TRAIN: the train/val gap is the bias-variance diagnostic.
            Y_hat_tr = model.predict(X_tr)
            I_hat_tr = d["tf"].inverse_transform(Y_hat_tr)
            train_m = {t: summarize(d["I"][d["tr"]][:, k], I_hat_tr[:, k], Y_tr[:, k], Y_hat_tr[:, k])
                       for k, t in enumerate(C.TARGETS)}

            for t, m in per_target.items():
                print(f"     {t}: R2(log)={m['r2_log']:.5f} median={m['median_err_pct']:.3f}% "
                      f"p95={m['p95_err_pct']:.2f}% sign={m['sign_acc_pct']:.1f}%  "
                      f"(train: R2 {train_m[t]['r2_log']:.5f}, median {train_m[t]['median_err_pct']:.3f}%)")
            print(f"     fit {fit_s:.0f}s, predict {pred_ms:.3f} ms/sample")

            name = f"{case}__{fam}{suffix}"
            pred = LeakagePredictor(case, fam, model, d["tf"], list(d["X"].columns), args.feature_set,
                                    meta={"tuning": tuning, "quick": args.quick})
            pred.save(cand_mod / f"{name}.joblib")
            (cand_res / f"{name}.json").write_text(json.dumps({
                "case": case, "family": fam, "feature_set": args.feature_set, "quick": args.quick,
                "val": per_target, "train": train_m, "tuning": tuning,
                "fit_seconds": fit_s, "predict_ms_per_sample": pred_ms,
                "n_features": d["X"].shape[1], "dropped_features": d["drop"],
                "i0": d["tf"].i0_.tolist(), "host": platform.node(),
            }, indent=2, default=str))


if __name__ == "__main__":
    main()
