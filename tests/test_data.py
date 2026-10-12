import numpy as np
import pandas as pd
import pytest

from exoplanets.data import clean, load_planets, to_duckdb


def _frame(**overrides):
    base = {
        "pl_name": ["A b", "A c", "B b"],
        "hostname": ["A", "A", "B"],
        "discoverymethod": ["Transit", "Transit", "Radial Velocity"],
        "disc_year": [2014, 2015, 2010],
        "pl_orbper": [3.0, 10.0, 200.0],
        "pl_rade": [1.5, 2.5, 11.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_load_skips_only_leading_comment_lines(tmp_path):
    path = tmp_path / "pscomppars.csv"
    path.write_text(
        "# header line one\n# header line two\n"
        "pl_name,hostname,discoverymethod,disc_year,pl_rade\n"
        "A b,A,Transit,2014,1.5\n"
        "C b,C,Transit,2016,2.0 # not a comment\n"
    )
    df, meta = load_planets(path)
    assert len(df) == 2
    assert meta["header_comments"] == ["header line one", "header line two"]
    assert "pl_orbper" in meta["missing_columns"]
    assert df["pl_orbper"].isna().all()


def test_missing_required_column_raises():
    raw = _frame().drop(columns=["discoverymethod"])
    with pytest.raises(ValueError, match="missing required"):
        clean(raw)


def test_nonpositive_values_become_missing():
    raw = _frame(pl_rade=[1.5, -2.0, 0.0])
    df, notes = clean(raw)
    assert df["pl_rade"].isna().tolist() == [False, True, True]
    assert notes["nonpositive_set_missing"]["pl_rade"] == 2


def test_duplicates_and_non_default_rows_are_dropped():
    raw = _frame()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    raw["default_flag"] = [1, 1, 1, 0]
    df, notes = clean(raw)
    assert len(df) == 3
    assert notes["dropped_non_default"] == 1
    assert df["pl_name"].is_unique


def test_missing_hostname_falls_back_to_planet_name():
    raw = _frame(hostname=["A", None, "B"])
    df, _ = clean(raw)
    assert df.loc[1, "hostname"] == "A c"


def test_duckdb_roundtrip_matches_row_count():
    df, _ = clean(_frame())
    con = to_duckdb(df)
    assert con.execute("SELECT COUNT(*) FROM planets").fetchone()[0] == len(df)
    con.close()
