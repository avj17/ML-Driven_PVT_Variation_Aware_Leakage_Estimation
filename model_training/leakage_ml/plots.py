"""Figures for the report. Every plot has a title, axis labels and a legend."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C


def parity_plot(I_true, I_pred, case, family, split, path):
    """Predicted vs SPICE |I| on log axes, one panel per terminal (Deliverable 5)."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for k, (ax, t) in enumerate(zip(axes, C.TARGETS)):
        a, b = np.abs(I_true[:, k]), np.abs(I_pred[:, k])
        same = np.sign(I_true[:, k]) == np.sign(I_pred[:, k])
        ax.scatter(a[same], b[same], s=4, alpha=0.4, label="sign correct")
        if (~same).any():
            ax.scatter(a[~same], b[~same], s=10, color="tab:red", label=f"sign wrong ({(~same).sum()})")
        lo, hi = min(a.min(), b[b > 0].min()), max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"SPICE |{t}| (A)"); ax.set_ylabel(f"ML |{t}| (A)")
        ax.set_title(f"{t}"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"{case}: {family} predicted vs SPICE ({split} set)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def error_hist(errs: dict, case, split, path):
    """Distribution of relative error per terminal (log x-axis)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(-4, 3, 60)
    for t, e in errs.items():
        ax.hist(np.clip(e, 1e-4, 1e3), bins=bins, histtype="step", lw=1.5, label=t)
    ax.set_xscale("log"); ax.set_xlabel("relative error (%)"); ax.set_ylabel("count")
    ax.set_title(f"{case}: relative error distribution ({split} set)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def learning_curve(history, case, path):
    """MLP train vs inner-validation loss per epoch."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="inner validation")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardised signed-log target)")
    ax.set_title(f"{case}: MLP learning curve"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def importance_bar(imp, case, family, path):
    """Permutation importance on the validation set (mean increase in MSE)."""
    imp = imp.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, label="mean over terminals")
    ax.set_xlabel("increase in signed-log MSE when shuffled"); ax.set_ylabel("feature")
    ax.set_title(f"{case}: permutation importance ({family}, validation)")
    ax.grid(True, axis="x", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
