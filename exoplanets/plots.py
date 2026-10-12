"""Figures for the report, drawn with matplotlib and returned as PNG bytes.

Each figure is rendered for a light and a dark theme. Colors come from a
validated categorical palette assigned in fixed order, and a discovery method
keeps the same color in every figure.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "other": "#a8a69f",
        "bar": "#3987e5",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "other": "#6b6a64",
        "bar": "#3987e5",
    },
}

SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#0d366b"]
KNOWN_METHOD_SLOT = {"Transit": 0, "Radial Velocity": 1, "Microlensing": 2}


def method_color_map(groups: list[str], theme: str) -> dict[str, str]:
    """Color follows the entity: a known method always gets the same slot."""
    t = THEMES[theme]
    colors: dict[str, str] = {}
    used = set()
    for g in groups:
        if g in KNOWN_METHOD_SLOT:
            used.add(KNOWN_METHOD_SLOT[g])
    next_slot = 0
    for g in groups:
        if g == "Other":
            colors[g] = t["other"]
        elif g in KNOWN_METHOD_SLOT:
            colors[g] = t["series"][KNOWN_METHOD_SLOT[g]]
        else:
            while next_slot in used:
                next_slot += 1
            colors[g] = t["series"][next_slot] if next_slot < len(t["series"]) else t["other"]
            used.add(next_slot)
    return colors


def _seq_cmap(theme: str) -> LinearSegmentedColormap:
    ramp = SEQ_RAMP if theme == "light" else SEQ_RAMP[::-1]
    return LinearSegmentedColormap.from_list(f"seq_{theme}", ramp)


def _new_fig(theme: str, size=(8.0, 4.4)):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])
    return fig, ax


def _png(fig, theme: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor=THEMES[theme]["surface"], bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plain(v, _pos=None) -> str:
    return f"{v:g}"


def _spread(ys, min_gap: float) -> np.ndarray:
    """Nudge label positions apart so direct labels do not overlap."""
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(ys)
    out = ys[order].copy()
    for i in range(1, len(out)):
        if out[i] - out[i - 1] < min_gap:
            out[i] = out[i - 1] + min_gap
    res = np.empty_like(out)
    res[order] = out
    return res


def fig_cumulative(wide: pd.DataFrame, theme: str) -> bytes:
    """Cumulative confirmed planets by discovery method."""
    t = THEMES[theme]
    fig, ax = _new_fig(theme)
    cmap = method_color_map(list(wide.columns), theme)
    cum = wide.cumsum()
    x_last = cum.index[-1]
    ends = []
    for g in cum.columns:
        ax.plot(cum.index, cum[g], color=cmap[g], linewidth=2, label=g, solid_capstyle="round")
        ax.plot(
            [x_last], [cum[g].iloc[-1]], "o", color=cmap[g], markersize=6,
            markeredgecolor=t["surface"], markeredgewidth=1.5,
        )
        ends.append(cum[g].iloc[-1])
    ymax = max(ends)
    label_y = _spread(ends, 0.07 * ymax)
    for g, end, y in zip(cum.columns, ends, label_y):
        if abs(y - end) > 0.02 * ymax:
            # Short leader line from the line end to a label that was nudged apart.
            ax.plot([x_last + 0.3, x_last + 0.9], [end, y], color=t["muted"], linewidth=0.8)
        ax.text(x_last + 1.1, y, g, color=t["ink2"], va="center", fontsize=9)
    ax.set_xlim(cum.index[0] - 0.5, x_last + 8.5)
    ax.set_xticks([v for v in ax.get_xticks() if cum.index[0] <= v <= x_last])
    ax.set_ylim(0, ymax * 1.08)
    ax.set_ylabel("Cumulative confirmed planets")
    ax.set_xlabel("Discovery year")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    return _png(fig, theme)


def fig_period_radius(df: pd.DataFrame, top_methods: list[str], theme: str, period_max: float, r_min: float, r_max: float) -> bytes:
    """Orbital period against planet radius, colored by discovery method."""
    t = THEMES[theme]
    fig, ax = _new_fig(theme, size=(8.0, 4.8))
    data = df.dropna(subset=["pl_orbper", "pl_rade"]).copy()
    data["group"] = data["discoverymethod"].where(data["discoverymethod"].isin(top_methods), "Other")
    groups = [g for g in top_methods if g in set(data["group"])]
    if (data["group"] == "Other").any():
        groups.append("Other")
    cmap = method_color_map(groups, theme)
    handles = {}
    # Draw the gray group first so the colored methods sit on top of it.
    for g in ["Other"] + [g for g in groups if g != "Other"]:
        sub = data[data["group"] == g]
        if sub.empty:
            continue
        handles[g] = ax.scatter(
            sub["pl_orbper"], sub["pl_rade"], s=12, color=cmap[g], alpha=0.7,
            edgecolors=t["surface"], linewidths=0.3, label=f"{g} ({len(sub):,})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticks([1, 10, 100, 1000])
    xmin = data["pl_orbper"].min() * 0.8
    ax.add_patch(
        Rectangle((xmin, r_min), period_max - xmin, r_max - r_min, fill=False,
                  edgecolor=t["ink2"], linestyle="--", linewidth=1.2)
    )
    ax.text(xmin * 1.1, r_max * 1.12, "Region used for the gap analysis", color=t["ink2"], fontsize=8.5)
    ax.xaxis.set_major_formatter(FuncFormatter(_plain))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain))
    ax.minorticks_off()
    ax.set_xlabel("Orbital period (days)")
    ax.set_ylabel("Planet radius (Earth radii)")
    ordered = [handles[g] for g in groups if g in handles]
    leg = ax.legend(handles=ordered, frameon=False, loc="lower right", fontsize=9, markerscale=1.6)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    return _png(fig, theme)


def fig_radius_gap(gap, theme: str) -> bytes:
    """Histogram of small-planet radii with the density estimate and any detected gap."""
    t = THEMES[theme]
    fig, ax = _new_fig(theme)
    n_bins = 32
    edges = np.logspace(np.log10(gap.r_min), np.log10(gap.r_max), n_bins + 1)
    ax.hist(gap.radii, bins=edges, color=t["bar"], edgecolor=t["surface"], linewidth=1.2, label="Planets per size bin")
    if gap.grid_re is not None:
        bin_dex = (np.log10(gap.r_max) - np.log10(gap.r_min)) / n_bins
        ax.plot(gap.grid_re, gap.density_per_dex * len(gap.radii) * bin_dex,
                color=t["series"][1], linewidth=2, label="Smoothed density")
    if gap.found:
        if gap.ci_low is not None:
            ax.axvspan(gap.ci_low, gap.ci_high, color=t["ink2"], alpha=0.15, linewidth=0)
        ax.axvline(gap.gap_re, color=t["ink2"], linestyle="--", linewidth=1.2)
        ax.text(gap.gap_re * 1.03, ax.get_ylim()[1] * 0.97, f"dip near {gap.gap_re:.2f}",
                color=t["ink2"], fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 4])
    ax.xaxis.set_major_formatter(FuncFormatter(_plain))
    ax.minorticks_off()
    ax.set_xlabel("Planet radius (Earth radii)")
    ax.set_ylabel("Number of planets")
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    return _png(fig, theme)


def fig_importance(importance: pd.DataFrame, theme: str) -> bytes:
    """Permutation importance of each feature on held-out systems."""
    t = THEMES[theme]
    fig, ax = _new_fig(theme, size=(8.0, 3.2))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    data = importance.iloc[::-1]
    ax.barh(data["feature"], data["mean_drop"], xerr=data["std_drop"], color=t["bar"],
            edgecolor=t["surface"], linewidth=1.2, height=0.62,
            error_kw={"ecolor": t["muted"], "elinewidth": 1.2, "capsize": 3})
    ax.axvline(0, color=t["axis"], linewidth=1)
    ax.set_xlabel("Drop in R squared when the feature is shuffled (average over folds)")
    ax.tick_params(axis="y", colors=t["ink2"], labelsize=10)
    return _png(fig, theme)


def fig_pred_obs(oof: pd.DataFrame, theme: str) -> bytes:
    """Held-out predictions against observed radius, binned because points overlap."""
    t = THEMES[theme]
    fig, ax = _new_fig(theme, size=(6.4, 5.2))
    ax.grid(axis="both", color=t["grid"], linewidth=0.8)
    hb = ax.hexbin(
        oof["observed_re"], oof["predicted_re"], gridsize=38, xscale="log", yscale="log",
        bins="log", cmap=_seq_cmap(theme), mincnt=1, linewidths=0.3, edgecolors=t["surface"],
    )
    lo = float(min(oof["observed_re"].min(), oof["predicted_re"].min()))
    hi = float(max(oof["observed_re"].max(), oof["predicted_re"].max()))
    ax.plot([lo, hi], [lo, hi], color=t["ink2"], linestyle="--", linewidth=1.2)
    ax.text(lo * 1.15, lo * 1.45, "perfect prediction", color=t["ink2"], fontsize=8.5, rotation=0)
    ticks = [v for v in (0.5, 1, 2, 5, 10, 20) if lo <= v <= hi]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_plain))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain))
    ax.minorticks_off()
    ax.set_xlabel("Observed radius (Earth radii)")
    ax.set_ylabel("Predicted radius on held-out systems (Earth radii)")
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("Planets per cell (log scale)", color=t["muted"], fontsize=9)
    cb.ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    cb.outline.set_visible(False)
    return _png(fig, theme)
