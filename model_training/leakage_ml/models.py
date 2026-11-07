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
from .metrics import relative_error
from .transforms import KCLTransform, SignedLogTransform


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
def fit_poly_elasticnet(X, Y, space, n_jobs=-1, log=print, tf=None):
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


def fit_random_forest(X, Y, space, n_jobs=-1, log=print, tf=None):
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
def fit_xgboost(X, Y, space, n_jobs=-1, log=print, tf=None):
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
    fitted net is moved to CPU so saved bundles load anywhere.

    loss: "mse" or "huber" (robust: errors above huber_delta, in standardised
    units, count linearly, so a few sign-crossing samples cannot dominate).
    Early stopping monitors the inner hold-out loss, or score_fn (lower is
    better) when one is passed to fit().
    sign_cols=(i_vd, i_vs) with sign_outputs: those outputs are learned as
    magnitudes and get the sign sign_flip_ * sign(vd - vs) at prediction."""

    def __init__(self, hidden, activation="silu", dropout=0.0, weight_decay=0.0,
                 lr=1e-3, batch_size=256, max_epochs=1000, patience=100, seed=C.SEED_TORCH,
                 loss="mse", huber_delta=0.1, sign_cols=None, sign_outputs=()):
        self.hidden, self.activation, self.dropout = hidden, activation, dropout
        self.weight_decay, self.lr, self.batch_size = weight_decay, lr, batch_size
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed
        self.loss, self.huber_delta = loss, huber_delta
        self.sign_cols, self.sign_outputs = sign_cols, tuple(sign_outputs)
        self.sign_flip_ = 1.0
        self.net, self.history = None, {"train": [], "val": []}

    def get_params(self):
        return {k: getattr(self, k) for k in ("hidden", "activation", "dropout", "weight_decay", "lr",
                                              "batch_size", "max_epochs", "patience", "seed",
                                              "loss", "huber_delta", "sign_cols", "sign_outputs")}

    def _signs(self, X):
        return self.sign_flip_ * np.sign(X[:, self.sign_cols[0]] - X[:, self.sign_cols[1]])

    def _post(self, Y, X):
        """Impose the known signs on the magnitude outputs (unscaled target space)."""
        if getattr(self, "sign_outputs", ()):      # attribute absent in pre-KCL pickles
            s = self._signs(X)
            for k in self.sign_outputs:
                Y[:, k] = s * np.maximum(Y[:, k], 0.0)
        return Y

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

    def fit(self, X, Y, X_val, Y_val, score_fn=None, sample_weight=None, val_weight=None):
        """sample_weight / val_weight: optional per-row loss weights (mean 1)."""
        import torch

        X, X_val = np.asarray(X, dtype=float), np.asarray(X_val, dtype=float)
        Y, Y_val = np.array(Y, dtype=float), np.array(Y_val, dtype=float)
        if self.sign_outputs:
            k = list(self.sign_outputs)
            agree = np.sum(np.sign(Y[:, k[0]]) * np.sign(X[:, self.sign_cols[0]] - X[:, self.sign_cols[1]]))
            self.sign_flip_ = 1.0 if agree >= 0 else -1.0
            Y[:, k], Y_val[:, k] = np.abs(Y[:, k]), np.abs(Y_val[:, k])

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
        elem = (torch.nn.HuberLoss(delta=self.huber_delta, reduction="none") if self.loss == "huber"
                else torch.nn.MSELoss(reduction="none"))
        Wt = to_t(np.ones(len(X)) if sample_weight is None else sample_weight)
        Wv = to_t(np.ones(len(X_val)) if val_weight is None else val_weight)
        loss_fn = lambda out, y, w: (elem(out, y).mean(dim=1) * w).sum() / w.sum()
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
                loss = loss_fn(self.net(Xt[b]), Yt[b], Wt[b])
                loss.backward()
                opt.step()
                tot += loss.detach() * len(b)
            self.net.eval()
            with torch.no_grad():
                out_v = self.net(Xv)
                val = loss_fn(out_v, Yv, Wv).item()
            monitor = val
            if score_fn is not None:
                Zp = self.y_scaler.inverse_transform(out_v.cpu().numpy().astype(float))
                monitor = score_fn(self._post(Zp, X_val))
                self.history.setdefault("score", []).append(monitor)
            self.history["train"].append(tot.item() / n)
            self.history["val"].append(val)
            sched.step(val)
            if monitor < best_val * (1 - 1e-4):
                best_val, bad, self.best_epoch_ = monitor, 0, epoch + 1
                best_state = copy.deepcopy(self.net.state_dict())   # deep copy: true snapshot
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_monitor_, self.epochs_ = best_val, epoch + 1
        self.net.cpu().eval()
        return self

    def predict(self, X):
        import torch
        X = np.asarray(X, dtype=float)
        with torch.no_grad():
            z = self.net(torch.tensor(self.x_scaler.transform(X), dtype=torch.float32))
        return self._post(self.y_scaler.inverse_transform(z.numpy().astype(float)), X)


class EnsembleRegressor:
    """Mean of independently trained members, in their (shared) target space."""

    def __init__(self, members):
        self.members = members

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members], axis=0)


class TargetAdapter:
    """Wraps a model trained on another target parameterisation (inner_tf,
    e.g. KCLTransform) so predict() returns the standard signed-log targets
    (outer_tf) that the rest of the pipeline and LeakagePredictor expect."""

    def __init__(self, model, inner_tf, outer_tf, cfg, space):
        self.model, self.inner_tf, self.outer_tf = model, inner_tf, outer_tf
        self.cfg, self.space = cfg, space

    def predict(self, X):
        return self.outer_tf.transform(self.inner_tf.inverse_transform(self.model.predict(X)))

    def refit(self, X, Y):
        """Same recipe and target transform, new training data (size curve)."""
        X = np.asarray(X, dtype=float)
        I = self.outer_tf.inverse_transform(Y)
        Z = self.inner_tf.transform(I)
        n = len(mlp_members(self))
        members = [_train_member(X, I, Z, self.cfg, self.space, self.inner_tf, k)[0] for k in range(n)]
        return TargetAdapter(members[0] if n == 1 else EnsembleRegressor(members),
                             self.inner_tf, self.outer_tf, self.cfg, self.space)


def mlp_members(model):
    """The TorchMLPRegressor(s) inside a fitted MLP model, for plots and refits."""
    if isinstance(model, TargetAdapter):
        model = model.model
    if isinstance(model, EnsembleRegressor):
        return list(model.members)
    return [model] if isinstance(model, TorchMLPRegressor) else []


MLP_TARGETS = {"direct": SignedLogTransform, "kcl": KCLTransform, "kcl_sign": KCLTransform}


def inner_split(n, member=0):
    """Inner hold-out carved from TRAIN for MLP early stopping / config choice.
    Ensemble member k uses its own split, so together the members see all of TRAIN."""
    idx = np.arange(n)
    it, iv = train_test_split(idx, test_size=C.SEARCH_SPACES["mlp"]["inner_val_frac"],
                              random_state=C.SEED_CV + member, shuffle=True)
    return it, iv


def spec_score(I_true, I_pred):
    """The spec metrics in amperes: mean over the four currents of the median,
    95th-percentile and mean relative error, in percent."""
    err = relative_error(I_true, I_pred)          # NaN where I_true == 0
    return {"median": float(np.mean(np.nanmedian(err, axis=0))),
            "p95": float(np.mean(np.nanpercentile(err, 95, axis=0))),
            "mean": float(np.mean(np.nanmean(err, axis=0)))}


def kappa_weights(I, w_max):
    """Loss weight per row: how strongly KCL amplifies errors in A and S into
    Id and Is, kappa = (|A| + |S|) / |I| averaged over Id and Is, clipped to
    [1, w_max] and normalised to mean 1. ~1 in the ON state; large in the OFF
    rows where Id or Is is a near-cancellation."""
    a, s = np.abs(I[:, 0] - I[:, 2]) / 2, np.abs(I[:, 0] + I[:, 2]) / 2
    k = 0.5 * ((a + s) / np.maximum(np.abs(I[:, 0]), 1e-30) + (a + s) / np.maximum(np.abs(I[:, 2]), 1e-30))
    w = np.clip(k, 1.0, w_max)
    return w / w.mean()


def _train_member(X, I, Z, cfg, space, inner_tf, member):
    """Train one network of config cfg (target transform already applied: Z) on
    TRAIN minus an inner hold-out, early-stopping on that hold-out."""
    it, iv = inner_split(len(X), member)
    arch = {k: cfg[k] for k in ("hidden", "activation", "dropout", "weight_decay", "lr", "batch_size")}
    sign = ({"sign_cols": cfg["sign_cols"], "sign_outputs": (0,)} if cfg["target"] == "kcl_sign" else {})
    m = TorchMLPRegressor(max_epochs=space["max_epochs"], patience=space["patience"],
                          seed=C.SEED_TORCH + member, loss=cfg["loss"],
                          huber_delta=cfg.get("huber_delta", 0.1), **arch, **sign)
    score_fn = None
    if cfg["monitor"] != "loss":            # "metric" (= median, the v2 name), "median" or "mean"
        key = "median" if cfg["monitor"] == "metric" else cfg["monitor"]
        I_iv = I[iv]
        score_fn = lambda Zp: spec_score(I_iv, inner_tf.inverse_transform(Zp))[key]
    w_tr = w_iv = None
    if cfg.get("weight", "none") == "kappa":
        w_tr, w_iv = kappa_weights(I[it], cfg["weight_max"]), kappa_weights(I[iv], cfg["weight_max"])
    m.fit(X[it], Z[it], X[iv], Z[iv], score_fn=score_fn, sample_weight=w_tr, val_weight=w_iv)
    return m, spec_score(I[iv], inner_tf.inverse_transform(m.predict(X[iv])))


def mlp_config(cfg, space, cols):
    """Fill a config with the space-level recipe defaults and resolve column indices."""
    cfg = {"target": space["target"], "loss": space["loss"], "huber_delta": space["huber_delta"],
           "monitor": space["monitor"], "weight": space.get("weight", "none"),
           "weight_max": space.get("weight_max", 50.0), **cfg}
    if cfg["target"] == "kcl_sign":
        cfg["sign_cols"] = (cols.index("vd"), cols.index("vs"))
    return cfg


def fit_mlp(X, Y, space, n_jobs=-1, log=print, tf=None):
    """Each config (architecture, target parameterisation, loss, early-stopping
    monitor) is trained on 90 % of TRAIN and scored on the other 10 % with the
    spec metric in amperes (mean median relative error, ties by p95), so configs
    with different target spaces are comparable. The best config is retrained
    n_ensemble times (different seeds and inner hold-outs) and averaged."""
    import torch
    if n_jobs > 0:
        torch.set_num_threads(n_jobs)
    cols = list(X.columns)
    X = np.asarray(X, dtype=float)
    I = tf.inverse_transform(np.asarray(Y))
    best, trace = None, []
    for cfg in space["configs"]:
        cfg = mlp_config(cfg, space, cols)
        inner_tf = MLP_TARGETS[cfg["target"]]().fit(I)
        t0 = time.perf_counter()
        m, inner = _train_member(X, I, inner_tf.transform(I), cfg, space, inner_tf, 0)
        row = dict(cfg, inner_median_pct=inner["median"], inner_p95_pct=inner["p95"],
                   inner_mean_pct=inner["mean"], epochs=m.epochs_, best_epoch=m.best_epoch_,
                   seconds=round(time.perf_counter() - t0, 1), device=gpu_device())
        log(f"      {row}")
        trace.append(row)
        key = (inner[space.get("select", "median")], inner["p95"])
        if best is None or key < best[0]:
            best = (key, cfg, m, inner_tf, row)
    _, cfg, m0, inner_tf, row = best
    Z = inner_tf.transform(I)
    members = [m0]
    for k in range(1, space["n_ensemble"]):
        t0 = time.perf_counter()
        mk, inner = _train_member(X, I, Z, cfg, space, inner_tf, k)
        log(f"      ensemble member {k}: inner median {inner['median']:.3f}% mean {inner['mean']:.3f}% "
            f"p95 {inner['p95']:.2f}% epochs {mk.epochs_} ({time.perf_counter() - t0:.0f}s)")
        members.append(mk)
    model = TargetAdapter(m0 if len(members) == 1 else EnsembleRegressor(members), inner_tf, tf, cfg, space)
    return model, [{"best": row, "trace": trace, "n_ensemble": len(members),
                    "member_epochs": [mm.epochs_ for mm in members]}]


FITTERS = {
    "poly_elasticnet": fit_poly_elasticnet,
    "random_forest": fit_random_forest,
    "xgboost": fit_xgboost,
    "mlp": fit_mlp,
}
