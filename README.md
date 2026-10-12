# Exoplanet Analytics

A small, reproducible analytics pipeline for the NASA Exoplanet Archive. It loads the Planetary Systems Composite Parameters table, cleans and stores it in DuckDB, runs three analyses, and writes one self-contained HTML report.

The three questions:

1. **How did discoveries accumulate?** Planets per year by discovery method, computed in SQL.
2. **Is there a gap in the sizes of small planets?** A search for a dip in the radius distribution of short-period planets between 1 and 4 Earth radii, with a bootstrap interval for where the dip sits, repeated on planets with precisely measured radii as a robustness check.
3. **How much of a planet's size can its orbit and star explain?** A gradient boosting model of planet radius, compared with a linear baseline and a predict-the-average baseline, scored on held-out host stars.

## Get the data

1. Open the archive's **Planetary Systems Composite Parameters** table (`pscomppars`).
2. Use **Download Table** and choose **CSV**. The default columns are enough. Exporting all columns also works.
3. Save the file anywhere, for example `data/pscomppars.csv`.

The loader skips the comment lines at the top of the archive's CSV. It needs `pl_name`, `hostname`, `discoverymethod`, and `disc_year`. It uses these when present and runs without the ones that are missing, and the report says which were absent: `pl_orbper`, `pl_rade`, `pl_radeerr1`/`pl_radeerr2` (precision check only), `pl_bmasse`, `pl_eqt`, `pl_orbsmax`, `st_teff`, `st_rad`, `st_mass`, `sy_dist`.

## Run it

```
pip install -r requirements.txt
python run_pipeline.py --csv data/pscomppars.csv --out report.html
```

Useful options: `--db exoplanets.duckdb` keeps the cleaned table in a DuckDB file you can query, `--n-boot` sets the number of bootstrap resamples, and `--seed` fixes randomness.

## Test it

```
python -m pytest
```

The tests use a synthetic fixture (`tests/synthetic.py`) with a planted two-population size distribution, so the gap analysis has a known answer to recover. The fixture is never real data, and reports built from it carry a banner when run with `--synthetic`.

## Layout

```
exoplanets/
  data.py       load, validate, clean, store in DuckDB
  analysis.py   SQL trends, radius gap, grouped cross-validated model
  plots.py      figures (light and dark themes, fixed color per method)
  report.py     self-contained HTML report, text generated from the results
  pipeline.py   command line entry point
tests/          unit tests and the synthetic fixture
```

## Design choices

- **Grouped cross-validation.** Planets in the same system share star properties. A random split would put near-copies of test rows in the training set, so folds are grouped by host star.
- **Gap detection with a stated threshold.** The dip counts only if it falls to 90% of the lower peak or less. The report gives the depth, the bootstrap interval, and the share of resamples in which a dip appears at all, so a fragile result looks fragile.
- **Prose generated from results.** Every number in the report text is computed from the data, so the text cannot disagree with the figures.
- **Precision check.** The loader derives each planet's relative radius uncertainty from `pl_radeerr1` and `pl_radeerr2` when present. The gap search runs on all planets and again on those with uncertainty at most 10% of the radius. Planets with no stated uncertainty are excluded from the second run.
- **Honest baselines.** The linear model and the predict-the-average model are reported next to gradient boosting. The report does not assume the more complex model wins.

## Limits

- Which planets are in the table depends on what each method can detect, so patterns describe the detected population.
- The composite table mixes measurements of different precision, and the analysis ignores stated uncertainties.
- The model shows association, not cause.
- The latest discovery year in a download may be incomplete.
