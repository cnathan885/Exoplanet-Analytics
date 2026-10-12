import numpy as np
import pandas as pd
import pytest

from exoplanets import analysis
from exoplanets.data import clean, to_duckdb
from tests.synthetic import make_planets


@pytest.fixture(scope="module")
def planets():
    df, _ = clean(make_planets())
    return df


def test_discovery_counts_match_rows_and_fold_into_other(planets):
    con = to_duckdb(planets)
    long = analysis.discoveries_by_year(con, top_n=3)
    wide = analysis.discoveries_wide(long)
    assert int(wide.to_numpy().sum()) == planets["disc_year"].notna().sum()
    assert wide.shape[1] <= 4
    assert wide.columns[-1] == "Other"
    assert list(wide.index) == list(range(wide.index.min(), wide.index.max() + 1))
    con.close()


def test_method_totals_sum_to_planet_count(planets):
    con = to_duckdb(planets)
    totals = analysis.method_totals(con)
    assert totals["planets"].sum() == len(planets)
    con.close()


def test_radius_gap_recovers_planted_gap(planets):
    gap = analysis.radius_gap(planets, n_boot=100, seed=1)
    assert gap.found
    assert 1.5 < gap.gap_re < 2.0
    low, high = gap.peaks_re
    assert low < gap.gap_re < high
    assert gap.ci_low <= gap.gap_re <= gap.ci_high
    assert gap.detect_rate > 0.9


def test_radius_gap_reports_none_for_a_single_population():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"pl_orbper": rng.uniform(1, 50, 800), "pl_rade": 10 ** rng.normal(np.log10(2.0), 0.12, 800)}
    )
    gap = analysis.radius_gap(df, n_boot=50)
    assert not gap.found
    assert gap.reason


def test_radius_gap_handles_too_few_planets():
    df = pd.DataFrame({"pl_orbper": [1.0, 2.0], "pl_rade": [1.5, 2.5]})
    gap = analysis.radius_gap(df)
    assert not gap.found
    assert "only 2" in gap.reason


def test_model_beats_the_mean_baseline_on_held_out_systems(planets):
    model = analysis.fit_radius_model(planets)
    m = model.metrics.set_index("model")
    assert m.loc["Gradient boosting", "r2"] > 0.05
    assert m.loc["Gradient boosting", "r2"] > m.loc["Predict the average", "r2"]
    assert m.loc["Predict the average", "r2"] <= 0.0


def test_model_predictions_cover_every_row_once(planets):
    model = analysis.fit_radius_model(planets)
    assert len(model.oof) == model.n
    assert model.oof["predicted_re"].notna().all()


def test_model_needs_enough_rows():
    df = pd.DataFrame(
        {"hostname": ["a", "b"], "pl_orbper": [1.0, 2.0], "pl_rade": [1.0, 2.0],
         "st_teff": [5000.0, 5100.0], "st_rad": [1.0, 1.0], "st_mass": [1.0, 1.0], "sy_dist": [10.0, 20.0]}
    )
    assert analysis.fit_radius_model(df) is None


def test_precision_filter_keeps_only_well_measured_planets():
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({
        "pl_orbper": rng.uniform(1, 50, n),
        "pl_rade": 10 ** rng.normal(np.log10(2.0), 0.15, n),
        "pl_rade_relerr": np.where(np.arange(n) % 2 == 0, 0.05, 0.5),
    })
    all_ = analysis.radius_gap(df, n_boot=20)
    precise = analysis.radius_gap(df, n_boot=20, max_rel_err=0.10)
    assert precise.n < all_.n
    assert precise.max_rel_err == 0.10
    # planets without a stated uncertainty are excluded, not assumed precise
    df["pl_rade_relerr"] = np.nan
    assert analysis.radius_gap(df, n_boot=20, max_rel_err=0.10).n == 0


def test_clean_computes_relative_radius_error():
    raw = make_planets()
    raw["pl_radeerr1"] = raw["pl_rade"] * 0.1
    raw["pl_radeerr2"] = -raw["pl_rade"] * 0.3
    df, _ = clean(raw)
    ok = df["pl_rade_relerr"].dropna()
    assert len(ok) > 0 and np.allclose(ok, 0.2)
