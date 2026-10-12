"""Analyses: discovery trends (SQL), the small-planet radius gap, and a grouped
cross-validated model of planet radius."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Discovery trends
# ---------------------------------------------------------------------------

DISCOVERIES_SQL = """
WITH ranked AS (
    SELECT discoverymethod,
           ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC, discoverymethod) AS rk
    FROM planets
    WHERE discoverymethod IS NOT NULL
    GROUP BY discoverymethod
)
SELECT p.disc_year,
       CASE WHEN r.rk <= ? THEN p.discoverymethod ELSE 'Other' END AS method_group,
       COUNT(*) AS n
FROM planets p
JOIN ranked r USING (discoverymethod)
WHERE p.disc_year IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2
"""

METHOD_TOTALS_SQL = """
SELECT discoverymethod AS method,
       COUNT(*) AS planets,
       MIN(disc_year) AS first_year,
       MAX(disc_year) AS last_year
FROM planets
WHERE discoverymethod IS NOT NULL
GROUP BY 1
ORDER BY planets DESC, method
"""


def discoveries_by_year(con: duckdb.DuckDBPyConnection, top_n: int = 3) -> pd.DataFrame:
    """Planets discovered per year, with all but the top_n methods folded into 'Other'."""
    return con.execute(DISCOVERIES_SQL, [top_n]).df()


def discoveries_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one column per method group and one row per year (no gaps in years).
    Columns are ordered by total, with 'Other' last."""
    wide = long.pivot_table(
        index="disc_year", columns="method_group", values="n", aggfunc="sum", fill_value=0
    )
    years = np.arange(int(wide.index.min()), int(wide.index.max()) + 1)
    wide = wide.reindex(years, fill_value=0)
    order = [c for c in wide.sum().sort_values(ascending=False).index if c != "Other"]
    if "Other" in wide.columns:
        order.append("Other")
    return wide[order].astype(int)


def method_totals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(METHOD_TOTALS_SQL).df()


def assign_method_group(df: pd.DataFrame, top_methods: list[str]) -> pd.Series:
    return df["discoverymethod"].where(df["discoverymethod"].isin(top_methods), "Other")


def summarize(df: pd.DataFrame) -> dict:
    per_host = df.groupby("hostname").size()
    return {
        "n_planets": int(len(df)),
        "n_hosts": int(df["hostname"].nunique()),
        "n_multi": int((per_host >= 2).sum()),
        "n_radius": int(df["pl_rade"].notna().sum()),
        "n_mass": int(df["pl_bmasse"].notna().sum()),
        "year_min": int(df["disc_year"].min()),
        "year_max": int(df["disc_year"].max()),
    }


# ---------------------------------------------------------------------------
# Radius gap
# ---------------------------------------------------------------------------

PROMINENCE_FRAC = 0.05   # a peak must stand this far above its surroundings (share of max density)
MIN_DEPTH_RATIO = 0.90   # dip must fall to at most this fraction of the lower peak to count as a gap


@dataclass
class GapResult:
    found: bool
    reason: str = ""
    n: int = 0
    period_max: float = 100.0
    max_rel_err: float | None = None
    r_min: float = 1.0
    r_max: float = 4.0
    peaks_re: tuple = ()
    gap_re: float | None = None
    depth_ratio: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    detect_rate: float | None = None
    grid_re: np.ndarray | None = None
    density_per_dex: np.ndarray | None = None
    radii: np.ndarray | None = None


def _locate_gap(x_log: np.ndarray, grid_log: np.ndarray):
    """Find the two most prominent density peaks and the valley between them."""
    dens = gaussian_kde(x_log)(grid_log)
    peaks, props = find_peaks(dens, prominence=PROMINENCE_FRAC * dens.max())
    if len(peaks) < 2:
        return None, dens
    top_two = peaks[np.argsort(props["prominences"])[-2:]]
    p1, p2 = sorted(int(p) for p in top_two)
    valley = p1 + int(np.argmin(dens[p1 : p2 + 1]))
    depth = float(dens[valley] / min(dens[p1], dens[p2]))
    return (p1, valley, p2, depth), dens


def radius_gap(
    df: pd.DataFrame,
    period_max: float = 100.0,
    r_min: float = 1.0,
    r_max: float = 4.0,
    n_boot: int = 300,
    seed: int = 0,
    min_n: int = 50,
    max_rel_err: float | None = None,
) -> GapResult:
    """Look for a dip in the radius distribution of short-period small planets.

    A kernel density estimate of log10(radius) is searched for two peaks. The valley
    between the two most prominent peaks is the candidate gap. Resampling planets with
    replacement gives an interval for the gap location and the share of resamples in
    which a gap is found at all.
    """
    sub = df[(df["pl_orbper"] < period_max) & df["pl_rade"].between(r_min, r_max)]
    if max_rel_err is not None:
        # Keep only planets whose stated radius uncertainty is small (NaN uncertainty fails the test).
        sub = sub[sub["pl_rade_relerr"] <= max_rel_err]
    radii = sub["pl_rade"].to_numpy(dtype=float)
    result = GapResult(found=False, n=len(radii), period_max=period_max, max_rel_err=max_rel_err, r_min=r_min, r_max=r_max, radii=radii)
    if len(radii) < min_n:
        result.reason = f"only {len(radii)} planets in the region (need at least {min_n})"
        return result

    x = np.log10(radii)
    grid = np.linspace(np.log10(r_min), np.log10(r_max), 400)
    located, dens = _locate_gap(x, grid)
    result.grid_re = 10**grid
    result.density_per_dex = dens

    if located is None:
        result.reason = "the density has fewer than two distinct peaks"
        return result
    p1, valley, p2, depth = located
    result.peaks_re = (float(10 ** grid[p1]), float(10 ** grid[p2]))
    result.gap_re = float(10 ** grid[valley])
    result.depth_ratio = depth
    if depth > MIN_DEPTH_RATIO:
        result.reason = f"the dip is shallow (valley at {depth:.1%} of the lower peak, above the {MIN_DEPTH_RATIO:.0%} cutoff)"
        return result
    result.found = True

    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        loc, _ = _locate_gap(xb, grid)
        if loc is not None and loc[3] <= MIN_DEPTH_RATIO:
            gaps.append(10 ** grid[loc[1]])
    result.detect_rate = len(gaps) / n_boot
    if len(gaps) >= 20:
        result.ci_low, result.ci_high = (float(v) for v in np.percentile(gaps, [2.5, 97.5]))
    return result


# ---------------------------------------------------------------------------
# Radius model
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "pl_orbper": "Orbital period",
    "st_teff": "Star temperature",
    "st_rad": "Star radius",
    "st_mass": "Star mass",
    "sy_dist": "Distance from Earth",
}
LOG_FEATURES = {"pl_orbper", "sy_dist"}


@dataclass
class ModelResult:
    n: int
    n_groups: int
    n_splits: int
    features: list
    metrics: pd.DataFrame
    importance: pd.DataFrame
    oof: pd.DataFrame


def _feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for key, label in FEATURE_LABELS.items():
        values = data[key].to_numpy(dtype=float)
        cols[label] = np.log10(values) if key in LOG_FEATURES else values
    return pd.DataFrame(cols, index=data.index)


def fit_radius_model(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 0, min_rows: int = 100
) -> ModelResult | None:
    """Predict log10(planet radius) from orbital period and host star properties.

    Cross-validation is grouped by host star. Planets in the same system share star
    properties, so an ordinary random split would let the model see near-copies of
    each test row during training and overstate accuracy.
    """
    data = df.dropna(subset=["pl_rade", "pl_orbper"])
    if len(data) < min_rows:
        return None

    X = _feature_frame(data)
    y = np.log10(data["pl_rade"].to_numpy(dtype=float))
    groups = data["hostname"].to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))

    oof = {"gbm": np.full(len(y), np.nan), "ridge": np.full(len(y), np.nan), "mean": np.full(len(y), np.nan)}
    importances = []
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        gbm = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=seed
        )
        gbm.fit(X.iloc[train], y[train])
        oof["gbm"][test] = gbm.predict(X.iloc[test])

        ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(X.iloc[train], y[train])
        oof["ridge"][test] = ridge.predict(X.iloc[test])

        oof["mean"][test] = y[train].mean()

        pi = permutation_importance(
            gbm, X.iloc[test], y[test], n_repeats=5, random_state=seed, scoring="r2"
        )
        importances.append(pi.importances_mean)

    def r2(pred):
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot

    rows = []
    for key, name in (("gbm", "Gradient boosting"), ("ridge", "Linear (ridge)"), ("mean", "Predict the average")):
        rmse = float(np.sqrt(np.mean((y - oof[key]) ** 2)))
        rows.append({"model": name, "r2": r2(oof[key]), "rmse_dex": rmse, "error_factor": 10**rmse})
    metrics = pd.DataFrame(rows)

    imp = np.vstack(importances)
    importance = (
        pd.DataFrame({"feature": list(X.columns), "mean_drop": imp.mean(axis=0), "std_drop": imp.std(axis=0)})
        .sort_values("mean_drop", ascending=False)
        .reset_index(drop=True)
    )
    oof_df = pd.DataFrame(
        {
            "hostname": groups,
            "observed_re": 10**y,
            "predicted_re": 10 ** oof["gbm"],
        }
    )
    return ModelResult(
        n=len(y),
        n_groups=n_groups,
        n_splits=n_splits,
        features=list(X.columns),
        metrics=metrics,
        importance=importance,
        oof=oof_df,
    )
