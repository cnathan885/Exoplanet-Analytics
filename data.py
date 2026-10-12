"""Load, validate, and clean exports of the NASA Exoplanet Archive
Planetary Systems Composite Parameters table (pscomppars).

The archive's CSV download starts with a block of comment lines (starting with
"#") followed by a header row. Only those leading lines are skipped, so a "#"
inside a data field is never mistaken for a comment.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["pl_name", "hostname", "discoverymethod", "disc_year"]

# Columns the analysis uses when present. Missing ones are added as NaN so the
# rest of the pipeline still runs, and the report lists what was missing.
NUMERIC_COLUMNS = [
    "pl_orbper",    # orbital period, days
    "pl_rade",      # planet radius, Earth radii
    "pl_bmasse",    # planet mass (or minimum mass), Earth masses
    "pl_eqt",       # equilibrium temperature, K
    "pl_orbsmax",   # orbit semi-major axis, AU
    "st_teff",      # star effective temperature, K
    "st_rad",       # star radius, solar radii
    "st_mass",      # star mass, solar masses
    "sy_dist",      # distance from Earth, parsecs
]

# None of these can be zero or negative physically, and non-positive values
# break log scales, so they are treated as missing.
POSITIVE_ONLY = list(NUMERIC_COLUMNS)

# Derived column: average stated radius uncertainty divided by the radius. It is NaN when the
# archive's uncertainty columns (pl_radeerr1, pl_radeerr2) are absent or empty for a planet.
DERIVED_COLUMNS = ["pl_rade_relerr"]

KEEP_COLUMNS = REQUIRED_COLUMNS + NUMERIC_COLUMNS + DERIVED_COLUMNS


def _leading_comment_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                lines.append(line.rstrip("\n"))
            else:
                break
    return lines


def file_fingerprint(path: Path) -> str:
    """Short SHA-256 of the file, so a report can say exactly which data it used."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and clean a raw table. Returns the cleaned frame and a notes dict."""
    missing_required = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_required:
        raise ValueError(
            f"Input is missing required column(s): {missing_required}. "
            f"Columns found: {list(raw.columns)[:30]}"
        )

    df = raw.copy()
    notes: dict = {"rows_in": int(len(df))}

    # If a multi-row-per-planet table was exported, keep only the default parameter set.
    notes["dropped_non_default"] = 0
    if "default_flag" in df.columns:
        flag = pd.to_numeric(df["default_flag"], errors="coerce")
        keep = flag == 1
        notes["dropped_non_default"] = int((~keep).sum())
        df = df[keep]

    for col in ("pl_name", "hostname", "discoverymethod"):
        df[col] = df[col].astype("string").str.strip()

    df = df[df["pl_name"].notna() & (df["pl_name"] != "")]

    before = len(df)
    df = df.drop_duplicates(subset="pl_name", keep="first")
    notes["dropped_duplicates"] = int(before - len(df))

    # Single-planet rows sometimes lack a host name; fall back to the planet name
    # so each such planet is its own group.
    df["hostname"] = df["hostname"].fillna(df["pl_name"])

    notes["missing_columns"] = [c for c in NUMERIC_COLUMNS if c not in df.columns]
    for col in notes["missing_columns"]:
        df[col] = np.nan
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    notes["nonpositive_set_missing"] = {}
    for col in POSITIVE_ONLY:
        bad = df[col] <= 0
        if bad.any():
            notes["nonpositive_set_missing"][col] = int(bad.sum())
            df.loc[bad, col] = np.nan

    if {"pl_radeerr1", "pl_radeerr2"} <= set(df.columns):
        up = pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs()
        lo = pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs()
        df["pl_rade_relerr"] = ((up + lo) / 2) / df["pl_rade"]
    else:
        df["pl_rade_relerr"] = np.nan

    year = pd.to_numeric(df["disc_year"], errors="coerce").round()
    df["disc_year"] = year.astype("Int64")

    df = df[KEEP_COLUMNS].reset_index(drop=True)
    notes["rows_out"] = int(len(df))
    return df, notes


def load_planets(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read an archive CSV export and return (clean frame, metadata)."""
    path = Path(path)
    comments = _leading_comment_lines(path)
    raw = pd.read_csv(path, skiprows=len(comments), low_memory=False)
    df, notes = clean(raw)
    meta = {
        "file_name": path.name,
        "sha256_short": file_fingerprint(path),
        "header_comments": [c.lstrip("# ").rstrip() for c in comments],
        **notes,
    }
    if meta["missing_columns"]:
        log.warning("Optional columns missing from input: %s", meta["missing_columns"])
    return df, meta


def to_duckdb(df: pd.DataFrame, database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Store the cleaned table in DuckDB as the `planets` table."""
    con = duckdb.connect(database)
    con.register("planets_df", df)
    con.execute("CREATE OR REPLACE TABLE planets AS SELECT * FROM planets_df")
    con.unregister("planets_df")
    return con
