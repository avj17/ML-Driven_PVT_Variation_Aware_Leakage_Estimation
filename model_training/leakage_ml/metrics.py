"""Metrics required by the spec (Deliverables 4-5): R^2, median and 95th-percentile error."""
import numpy as np
from sklearn.metrics import r2_score


def relative_error(I_true: np.ndarray, I_pred: np.ndarray) -> np.ndarray:
    """|I_pred - I_true| / |I_true| in percent."""
    return 100.0 * np.abs(I_pred - I_true) / np.maximum(np.abs(I_true), 1e-30)


def summarize(I_true: np.ndarray, I_pred: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-target metrics. R^2 is computed in the signed-log space the models are
    trained in (on raw amperes it would be dominated by the largest currents);
    R^2 on raw current is reported too for completeness."""
    err = relative_error(I_true, I_pred)
    return {
        "r2_log": float(r2_score(y_true, y_pred)),
        "r2_raw": float(r2_score(I_true, I_pred)),
        "median_err_pct": float(np.median(err)),
        "p95_err_pct": float(np.percentile(err, 95)),
        "mean_err_pct": float(np.mean(err)),
        "sign_acc_pct": float(100.0 * np.mean(np.sign(I_true) == np.sign(I_pred))),
    }
