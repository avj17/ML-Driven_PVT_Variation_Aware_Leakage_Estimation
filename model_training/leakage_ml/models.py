"""The four model families and their hyperparameter tuning (library models only:
scikit-learn, XGBoost, PyTorch).

All tuning uses the TRAINING set only (K-fold CV, or an inner hold-out carved
from TRAIN for the MLP's early stopping). The official validation set is
reserved for comparing the four families; the test set is touched once, in
evaluate.py. GPU is used wherever the library supports it (XGBoost, XGBoost
random forest, PyTorch MLP).

Every fitted model exposes predict(X) -> array (n, 4) in signed-log space.
Every fitter returns (model, info) where info holds, per target, the chosen
hyperparameters and the full search trace (for the report's tuning tables).
"""
import copy
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, ParameterSampler, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import config as C


def gpu_device():
    # A broken torch install must fail loudly, not silently fall back to CPU.
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("REQUIRE_GPU") == "1":
        raise RuntimeError("REQUIRE_GPU=1 but CUDA is not available")
    return "cpu"


def _kfold():
    return KFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED_CV)


class PerTargetRegressor:
    """One independently tuned estimator per terminal current."""

    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X):
        return np.column_stack([est.predict(np.asarray(X)) for est in self.estimators])


# ----------------------------------------------------------------------------
# 1. Polynomial Elastic Net (scikit-learn, CPU)
# ----------------------------------------------------------------------------
def fit_poly_elasticnet(X, Y, space, n_jobs=-1, log=print):
    """Degree chosen by CV MSE; alpha and l1_ratio chosen by ElasticNetCV's own
    K-fold path search. Both scalers sit inside the pipeline (fit on TRAIN)."""
    estimators, info = [], []
    for t in range(Y.shape[1]):
        best, trace = None, []
        for deg in space["degree"]:
            t0 = time.perf_counter()
            pipe = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=deg, include_bias=False),
                StandardScaler(),
                ElasticNetCV(l1_ratio=space["l1_ratio"], alphas=space["n_alphas"], cv=_kfold(),
                             max_iter=space["max_iter"], tol=1e-4, selection="random",
                             random_state=C.SEED_MODEL, n_jobs=n_jobs),
            )
            pipe.fit(X, Y[:, t])
            enet = pipe[-1]
            row = {"degree": deg, "alpha": float(enet.alpha_), "l1_ratio": float(enet.l1_ratio_),
                   "cv_mse": float(np.min(np.mean(enet.mse_path_, axis=-1))),
                   "nonzero": int(np.sum(enet.coef_ != 0)), "n_terms": int(enet.coef_.size),
                   "seconds": round(time.perf_counter() - t0, 1)}
            trace.append(row)
            log(f"      {C.TARGETS[t]} {row}")
            if best is None or row["cv_mse"] < best[0]["cv_mse"]:
                best = (row, pipe)
        estimators.append(best[1])
        info.append({"best": best[0], "trace": trace})
    return PerTargetRegressor(estimators), info


# ----------------------------------------------------------------------------
# 2. Random Forest (XGBoost RF mode on GPU, or scikit-learn on CPU)
# ----------------------------------------------------------------------------
def _cv_trace(gs):
    keep = [k for k in gs.cv_results_ if k.startswith("param_")] + ["mean_test_score", "std_test_score",
                                                                    "mean_fit_time"]
    rows = [{k.replace("param_", ""): (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in
             ((k, gs.cv_results_[k]) for k in keep)} for i in range(len(gs.cv_results_["params"]))]
    for r in rows:
        r["cv_mse"] = -r.pop("mean_test_score")
    return rows


def fit_random_forest(X, Y, space, n_jobs=-1, log=print):
    if C.RF_BACKEND == "sklearn":
        return _fit_random_forest_sklearn(X, Y, space["sklearn_grid"], n_jobs, log)
    import xgboost as xgb

    # XGBoost's documented random-forest recipe: ONE boosting round of
    # num_parallel_tree bagged trees, learning rate 1 (no shrinkage), row
    # subsampling + per-node feature subsampling, negligible L2.
    device = gpu_device()
    estimators, info = [], []
    for t in range(Y.shape[1]):
        base = xgb.XGBRegressor(n_estimators=1, learning_rate=1.0, reg_lambda=1e-5,
                                tree_method="hist", device=device, random_state=C.SEED_MODEL,
                                n_jobs=n_jobs)
        gs = GridSearchCV(base, space["grid"], cv=_kfold(), scoring="neg_mean_squared_error",
                          n_jobs=1, refit=True)
        gs.fit(np.asarray(X), Y[:, t])
        best = dict(gs.best_params_, cv_mse=float(-gs.best_score_))
        log(f"      {C.TARGETS[t]} best {best} ({device})")
        estimators.append(gs.best_estimator_)
        info.append({"best": best, "trace": _cv_trace(gs), "device": device})
    return PerTargetRegressor(estimators), info


def _fit_random_forest_sklearn(X, Y, grid, n_jobs, log):
    """CPU fallback: grid scored by out-of-bag R^2 (no extra CV fits needed)."""
    estimators, info = [], []
    for t in range(Y.shape[1]):
        best, trace = None, []
        for params in ParameterGrid(grid):
            rf = RandomForestRegressor(oob_score=True, n_jobs=n_jobs, random_state=C.SEED_MODEL, **params)
            rf.fit(X, Y[:, t])
            trace.append(dict(params, oob_r2=float(rf.oob_score_)))
            if best is None or rf.oob_score_ > best[1]["oob_r2"]:
                best = (rf, trace[-1])
        log(f"      {C.TARGETS[t]} best {best[1]}")
        estimators.append(best[0])
        info.append({"best": best[1], "trace": trace, "device": "cpu"})
    return PerTargetRegressor(estimators), info


# ----------------------------------------------------------------------------
# 3. XGBoost (gradient-boosted trees, GPU)
# ----------------------------------------------------------------------------
def fit_xgboost(X, Y, space, n_jobs=-1, log=print):
    """Randomised search; each sampled config is scored with K-fold CV on TRAIN
    via xgb.cv (early stopping on each fold's held-out part). The final model is
    refit on all of TRAIN with the CV-chosen number of trees."""
    import xgboost as xgb

    device = gpu_device()
    sampler = list(ParameterSampler(space["param_distributions"], n_iter=space["n_iter"],
                                    random_state=C.SEED_CV))
    estimators, info = [], []
    for t in range(Y.shape[1]):
        dtrain = xgb.DMatrix(np.asarray(X), label=Y[:, t])
        best, trace = None, []
        for params in sampler:
            p = {"objective": "reg:squarederror", "tree_method": "hist", "device": device,
                 "eta": params["learning_rate"], "max_depth": params["max_depth"],
                 "min_child_weight": params["min_child_weight"], "subsample": params["subsample"],
                 "colsample_bytree": params["colsample_bytree"], "lambda": params["reg_lambda"],
                 "alpha": params["reg_alpha"], "seed": C.SEED_MODEL}
            if n_jobs > 0:
                p["nthread"] = n_jobs
            res = xgb.cv(p, dtrain, num_boost_round=params["n_estimators"], nfold=C.CV_FOLDS,
                         early_stopping_rounds=100, seed=C.SEED_CV, shuffle=True)
            row = dict(params, best_rounds=len(res),
                       cv_mse=float(res["test-rmse-mean"].iloc[-1]) ** 2,
                       train_mse=float(res["train-rmse-mean"].iloc[-1]) ** 2)
            trace.append(row)
            if best is None or row["cv_mse"] < best["cv_mse"]:
                best = row
        params = {k: best[k] for k in space["param_distributions"]}
        params["n_estimators"] = best["best_rounds"]
        est = xgb.XGBRegressor(tree_method="hist", device=device, n_jobs=n_jobs,
                               random_state=C.SEED_MODEL, **params)
        est.fit(np.asarray(X), Y[:, t])
        log(f"      {C.TARGETS[t]} cv_mse={best['cv_mse']:.4e} {params} ({device})")
        estimators.append(est)
        info.append({"best": dict(params, cv_mse=best["cv_mse"]), "trace": trace, "device": device})
    return PerTargetRegressor(estimators), info


# ----------------------------------------------------------------------------
# 4. MLP (PyTorch, GPU), multi-output: one network predicts all four currents
# ----------------------------------------------------------------------------
class TorchMLPRegressor:
    """torch.nn MLP with standardised inputs/targets (scalers fit on the
    training portion), AdamW, ReduceLROnPlateau and early stopping with
    best-weight restore. All tensors live on the GPU during training; the
    fitted net is moved to CPU so saved bundles load anywhere."""

    def __init__(self, hidden, activation="silu", dropout=0.0, weight_decay=0.0,
                 lr=1e-3, batch_size=256, max_epochs=1000, patience=100, seed=C.SEED_TORCH):
        self.hidden, self.activation, self.dropout = hidden, activation, dropout
        self.weight_decay, self.lr, self.batch_size = weight_decay, lr, batch_size
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed
        self.net, self.history = None, {"train": [], "val": []}

    def get_params(self):
        return {k: getattr(self, k) for k in ("hidden", "activation", "dropout", "weight_decay", "lr",
                                              "batch_size", "max_epochs", "patience", "seed")}

    def _build(self, n_in, n_out):
        import torch.nn as nn
        act = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[self.activation]
        layers, prev = [], n_in
        for h in self.hidden:
            lin = nn.Linear(prev, h)
            if self.activation == "tanh":
                nn.init.xavier_normal_(lin.weight)          # Xavier for tanh
            else:
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")   # He for ReLU-like
            nn.init.zeros_(lin.bias)
            layers += [lin, act()]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, n_out))       # identity output for regression
        return nn.Sequential(*layers)

    def fit(self, X, Y, X_val, Y_val):
        import torch

        torch.manual_seed(self.seed)
        dev = gpu_device()
        self.x_scaler = StandardScaler().fit(X)
        self.y_scaler = StandardScaler().fit(Y)
        to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        Xt, Yt = to_t(self.x_scaler.transform(X)), to_t(self.y_scaler.transform(Y))
        Xv, Yv = to_t(self.x_scaler.transform(X_val)), to_t(self.y_scaler.transform(Y_val))

        self.net = self._build(X.shape[1], Y.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=self.patience // 3,
                                                           min_lr=1e-6)
        loss_fn = torch.nn.MSELoss()
        gen = torch.Generator(device=dev).manual_seed(self.seed)
        best_val, best_state, bad = np.inf, None, 0
        n = Xt.shape[0]
        for epoch in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen, device=dev)
            tot = torch.zeros((), device=dev)
            for i in range(0, n, self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.net(Xt[b]), Yt[b])
                loss.backward()
                opt.step()
                tot += loss.detach() * len(b)
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(Xv), Yv).item()
            self.history["train"].append(tot.item() / n)
            self.history["val"].append(val)
            sched.step(val)
            if val < best_val * (1 - 1e-4):
                best_val, bad = val, 0
                best_state = copy.deepcopy(self.net.state_dict())   # deep copy: true snapshot
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val_mse_, self.epochs_ = best_val, epoch + 1
        self.net.cpu().eval()
        return self

    def predict(self, X):
        import torch
        with torch.no_grad():
            z = self.net(torch.tensor(self.x_scaler.transform(np.asarray(X)), dtype=torch.float32))
        return self.y_scaler.inverse_transform(z.numpy().astype(float))


def inner_split(n):
    """Inner hold-out carved from TRAIN for MLP early stopping / config choice."""
    idx = np.arange(n)
    it, iv = train_test_split(idx, test_size=C.SEARCH_SPACES["mlp"]["inner_val_frac"],
                              random_state=C.SEED_CV, shuffle=True)
    return it, iv


def fit_mlp(X, Y, space, n_jobs=-1, log=print):
    """Each config: train on 90 % of TRAIN, early-stop on the other 10 %; keep
    the config with the lowest inner-validation MSE (standardised targets)."""
    import torch
    if n_jobs > 0:
        torch.set_num_threads(n_jobs)
    X, Y = np.asarray(X), np.asarray(Y)
    it, iv = inner_split(len(X))
    best, trace = None, []
    for cfg in space["configs"]:
        t0 = time.perf_counter()
        m = TorchMLPRegressor(max_epochs=space["max_epochs"], patience=space["patience"], **cfg)
        m.fit(X[it], Y[it], X[iv], Y[iv])
        row = dict(cfg, inner_val_mse=m.best_val_mse_, final_train_mse=m.history["train"][-1],
                   epochs=m.epochs_, seconds=round(time.perf_counter() - t0, 1), device=gpu_device())
        log(f"      {row}")
        trace.append(row)
        if best is None or m.best_val_mse_ < best.best_val_mse_:
            best = m
    chosen = next(r for r in trace if r["inner_val_mse"] == best.best_val_mse_)
    return best, [{"best": chosen, "trace": trace}]


FITTERS = {
    "poly_elasticnet": fit_poly_elasticnet,
    "random_forest": fit_random_forest,
    "xgboost": fit_xgboost,
    "mlp": fit_mlp,
}
