"""End-to-end run: CSV in, DuckDB and HTML report out."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import analysis, data, report

log = logging.getLogger("exoplanets")

PRECISE_REL_ERR = 0.10  # "precise" radii: stated uncertainty at most 10% of the radius


def run(
    csv_path: str | Path,
    out_path: str | Path = "report.html",
    db_path: str = ":memory:",
    synthetic: bool = False,
    seed: int = 0,
    n_boot: int = 300,
) -> Path:
    log.info("Loading %s", csv_path)
    df, meta = data.load_planets(csv_path)
    log.info("Kept %d of %d rows after cleaning", meta["rows_out"], meta["rows_in"])

    con = data.to_duckdb(df, db_path)
    long = analysis.discoveries_by_year(con, top_n=3)
    wide = analysis.discoveries_wide(long)
    totals = analysis.method_totals(con)
    summary = analysis.summarize(df)

    log.info("Looking for the radius gap")
    gap = analysis.radius_gap(df, n_boot=n_boot, seed=seed)
    precise_gap = analysis.radius_gap(df, n_boot=n_boot, seed=seed, max_rel_err=PRECISE_REL_ERR)
    log.info("Fitting the radius model")
    model = analysis.fit_radius_model(df, seed=seed)
    con.close()

    log.info("Building the report")
    html = report.build_report(
        df=df, meta=meta, summary=summary, wide=wide, totals=totals,
        gap=gap, precise_gap=precise_gap, model=model, synthetic=synthetic,
    )
    out = Path(out_path)
    out.write_text(html, encoding="utf-8")
    log.info("Wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Analyze a NASA Exoplanet Archive pscomppars CSV export.")
    p.add_argument("--csv", required=True, help="Path to the CSV downloaded from the archive")
    p.add_argument("--out", default="report.html", help="Where to write the HTML report")
    p.add_argument("--db", default=":memory:", help="Optional DuckDB file to keep the cleaned table")
    p.add_argument("--synthetic", action="store_true", help="Label the report as synthetic test data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=300, help="Bootstrap resamples for the gap interval")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.csv, args.out, args.db, args.synthetic, args.seed, args.n_boot)
