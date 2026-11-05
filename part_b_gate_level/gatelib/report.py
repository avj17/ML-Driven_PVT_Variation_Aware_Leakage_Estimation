"""Leakage metrics (spec Deliverables 6 and 8) and the plots shared by Part B and Part C."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def signed_err_pct(spice, ml):
    spice, ml = np.asarray(spice, float), np.asarray(ml, float)
    return 100.0 * (ml - spice) / np.abs(spice)


def leak_metrics(spice, ml) -> dict:
    """SPICE median, ML median, bias (mean signed relative error), median, 95th
    percentile and mean |relative error|, and mean accuracy = 100 - mean error."""
    spice, ml = np.asarray(spice, float), np.asarray(ml, float)
    ok = np.isfinite(spice) & np.isfinite(ml) & (spice != 0)
    e = signed_err_pct(spice[ok], ml[ok])
    a = np.abs(e)
    return {
        "n": int(ok.sum()),
        "spice_median_A": float(np.median(spice[ok])),
        "ml_median_A": float(np.median(ml[ok])),
        "bias_pct": float(np.mean(e)),
        "median_err_pct": float(np.median(a)),
        "p95_err_pct": float(np.percentile(a, 95)),
        "mean_err_pct": float(np.mean(a)),
        "max_err_pct": float(np.max(a)),
        "mean_accuracy_pct": float(100.0 - np.mean(a)),
    }


def metrics_table(df, group_cols, spice_col, ml_col) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)), **leak_metrics(g[spice_col], g[ml_col])})
    return pd.DataFrame(rows)


def to_md(df) -> str:
    return df.to_markdown(index=False, floatfmt=".4g")


def parity_plot(spice, ml, groups, title, path):
    """ML vs SPICE leakage on log axes, coloured by group, with +-5 % / +-10 % bands."""
    spice, ml = np.abs(np.asarray(spice, float)), np.abs(np.asarray(ml, float))
    groups = np.asarray(groups)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for g in pd.unique(groups):
        m = groups == g
        ax.scatter(spice[m], ml[m], s=6, alpha=0.5, label=str(g))
    lo = np.nanmin(np.r_[spice, ml]) / 1.5
    hi = np.nanmax(np.r_[spice, ml]) * 1.5
    x = np.array([lo, hi])
    ax.plot(x, x, "k-", lw=1, label="ideal")
    for f, ls in ((0.05, "--"), (0.10, ":")):
        ax.plot(x, x * (1 + f), "k" + ls, lw=0.7)
        ax.plot(x, x * (1 - f), "k" + ls, lw=0.7, label=f"±{int(f * 100)} %")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("SPICE leakage (A)")
    ax.set_ylabel("ML leakage (A)")
    ax.set_title(title)
    ax.legend(fontsize=7, markerscale=2, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_vs_corner_plot(df, err_col, groups, title, path,
                         params=("temp", "vdd", "n_vth0", "p_vth0", "n_toxe", "p_toxe")):
    """|relative error| against the corner parameters, to locate where errors occur."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    groups = np.asarray(groups)
    for ax, p in zip(axes.ravel(), params):
        for g in pd.unique(groups):
            m = groups == g
            ax.scatter(df[p].to_numpy()[m], df[err_col].abs().to_numpy()[m], s=4, alpha=0.4, label=str(g))
        ax.set_yscale("log")
        ax.set_xlabel(p)
        ax.set_ylabel("|error| (%)")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=7, markerscale=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
