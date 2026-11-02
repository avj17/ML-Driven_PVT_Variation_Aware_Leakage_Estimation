"""Loading, cleaning, physics-informed feature engineering and the train/val/test split."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_case(case: str) -> pd.DataFrame:
    """Read one case CSV, normalise column names and run basic sanity checks."""
    df = pd.read_csv(C.DATA_DIR / f"{case}.csv")
    df = df.rename(columns={"vtho": "vth0"})
    missing = set(C.RAW_FEATURES + C.TARGETS) - set(df.columns)
    if missing:
        raise ValueError(f"{case}: missing columns {sorted(missing)}")
    if df[C.RAW_FEATURES + C.TARGETS].isna().any().any():
        raise ValueError(f"{case}: NaNs present")
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  [{case}] dropped {n_dup} duplicate rows")
    return df


def physics_features(df: pd.DataFrame, mos: str) -> pd.DataFrame:
    """Raw inputs (in O(1) engineering units) plus device-physics features.

    The transistor is symmetric, so the terminal at the lower potential acts as
    the source for an NMOS (higher potential for a PMOS). Subthreshold current
    goes as W * u0 * exp((Vgs - Vth) / (n*Vt)) * (1 - exp(-|Vds| / Vt)), so the
    log-current is close to linear in the features built below; this helps the
    linear and neural models most and costs the tree models nothing.
    """
    p = 1.0 if mos == "nmos" else -1.0
    out = pd.DataFrame(index=df.index)

    # Raw inputs rescaled to engineering units (a scaler is still applied later;
    # this only keeps polynomial expansions numerically well conditioned).
    out["vdd"] = df["vdd"]
    out["temp_c"] = df["temp"]
    out["w_over_l"] = df["w"] / C.L_MIN
    out["vd"] = df["vd"]
    out["vs"] = df["vs"]
    out["vg"] = df["vg"]
    out["vb"] = df["vb"]
    out["vth0_abs"] = df["vth0"].abs()
    out["toxe_nm"] = df["toxe"] * 1e9
    out["ndep_1e18"] = df["ndep"] / 1e18
    out["u0"] = df["u0"]
    out["xj_nm"] = df["xj"] * 1e9

    # Terminal voltage differences.
    out["vgs"] = df["vg"] - df["vs"]
    out["vds"] = df["vd"] - df["vs"]
    out["vbs"] = df["vb"] - df["vs"]
    out["vgd"] = df["vg"] - df["vd"]
    out["vbd"] = df["vb"] - df["vd"]

    # Effective source/drain (polarity-aware) and channel direction.
    v_src = np.minimum(df["vd"], df["vs"]) if mos == "nmos" else np.maximum(df["vd"], df["vs"])
    out["vds_abs"] = (df["vd"] - df["vs"]).abs()
    out["dir"] = np.sign(p * (df["vd"] - df["vs"]))
    out["vgs_eff"] = p * (df["vg"] - v_src)
    out["vbs_eff"] = p * (df["vb"] - v_src)

    # Thermal voltage and subthreshold exponents.
    t_k = df["temp"] + 273.15
    vt = C.K_BOLTZMANN * t_k / C.Q_ELECTRON
    out["inv_vt"] = 1.0 / vt
    out["log_tk"] = np.log(t_k)
    out["vov_over_vt"] = (out["vgs_eff"] - out["vth0_abs"]) / vt
    out["vds_over_vt"] = out["vds_abs"] / vt
    out["log_drain_factor"] = np.log1p(-np.exp(-out["vds_over_vt"]) + 1e-12)

    # Width, mobility and oxide terms in log form (current ~ W*u0, Ig ~ exp(-B*tox)).
    out["log_w"] = np.log(df["w"])
    out["log_u0"] = np.log(df["u0"])
    out["inv_toxe"] = 1.0 / out["toxe_nm"]
    return out


RAW_ONLY = ["vdd", "temp_c", "w_over_l", "vd", "vs", "vg", "vb",
            "vth0_abs", "toxe_nm", "ndep_1e18", "u0", "xj_nm"]


def build_features(df: pd.DataFrame, mos: str, feature_set: str = "physics") -> pd.DataFrame:
    feats = physics_features(df, mos)
    return feats[RAW_ONLY] if feature_set == "raw" else feats


def drop_uninformative(X_train: pd.DataFrame, tol: float = 1e-12, corr_tol: float = 1 - 1e-9):
    """Columns to drop, decided on the TRAIN set only: constant columns and exact
    duplicates (|r| ~ 1, e.g. vg == vdd in the ON case)."""
    drop = [c for c in X_train.columns if X_train[c].std() < tol]
    keep = [c for c in X_train.columns if c not in drop]
    corr = X_train[keep].corr().abs().values
    for i in range(len(keep)):
        if keep[i] in drop:
            continue
        for j in range(i + 1, len(keep)):
            if keep[j] not in drop and corr[i, j] > corr_tol:
                drop.append(keep[j])
    return drop


def split_indices(n: int, case: str):
    """Seeded random 70/15/15 split; indices are written to disk so every model
    family (and later Part B analysis) uses exactly the same rows."""
    path = C.SPLITS_DIR / f"{case}_split.json"
    if path.exists():
        idx = json.loads(path.read_text())
        if idx["n"] == n and idx["seed"] == C.SEED_SPLIT:
            return np.array(idx["train"]), np.array(idx["val"]), np.array(idx["test"])
    idx = np.arange(n)
    tr, rest = train_test_split(idx, train_size=C.TRAIN_FRAC, random_state=C.SEED_SPLIT, shuffle=True)
    va, te = train_test_split(rest, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
                              random_state=C.SEED_SPLIT, shuffle=True)
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n": n, "seed": C.SEED_SPLIT,
                                "train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}))
    return tr, va, te
