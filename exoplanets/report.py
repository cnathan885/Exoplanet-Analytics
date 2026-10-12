"""Build a self-contained HTML report. Every number in the prose is computed from
the data passed in, so the text cannot drift from the results."""

from __future__ import annotations

import base64
import datetime as dt
import html

import pandas as pd
from jinja2 import Environment, select_autoescape

from . import plots
from .analysis import GapResult, ModelResult

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exoplanet Analytics</title>
<style>
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --ring: rgba(11,11,11,0.10); --warn-bg: #fff4d6; --warn-ink: #5c3d00;
  --series-1: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --ring: rgba(255,255,255,0.10); --warn-bg: #3a2c00; --warn-ink: #ffd98a;
    --series-1: #3987e5;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 880px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.75rem; line-height: 1.2; margin: 0 0 4px; }
h2 { font-size: 1.25rem; margin: 40px 0 8px; }
p { margin: 8px 0; }
.sub { color: var(--ink2); margin: 0 0 20px; }
.banner { background: var(--warn-bg); color: var(--warn-ink); border-radius: 8px; padding: 12px 14px; margin: 16px 0; font-weight: 600; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 12px 14px; }
.tile .v { font-size: 1.6rem; font-weight: 650; line-height: 1.2; }
.tile .l { color: var(--ink2); font-size: 0.85rem; }
figure { margin: 16px 0; background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 12px; }
figure img { display: block; width: 100%; height: auto; }
figcaption { color: var(--ink2); font-size: 0.85rem; margin-top: 8px; }
details { margin: 8px 0 0; }
summary { cursor: pointer; color: var(--ink2); font-size: 0.9rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 5px 8px; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink2); font-weight: 600; }
.small { color: var(--ink2); font-size: 0.85rem; }
ul { padding-left: 20px; }
li { margin: 4px 0; }
code { font-size: 0.9em; }
</style>
</head>
<body>
<main>
<h1>Exoplanet Analytics</h1>
<p class="sub">Discovery trends, the small-planet radius gap, and a grouped cross-validated size model, built on the NASA Exoplanet Archive Planetary Systems Composite Parameters table.</p>
{% if synthetic %}<div class="banner">SYNTHETIC TEST DATA. These figures come from a generated fixture used to test the code. They are not real exoplanet results.</div>{% endif %}

<div class="tiles">
{% for v, l in tiles %}<div class="tile"><div class="v">{{ v }}</div><div class="l">{{ l }}</div></div>{% endfor %}
</div>

<h2>1. How discoveries accumulated</h2>
<p>{{ trend_text }}</p>
{{ fig("cumulative", "Line chart of cumulative confirmed planets by discovery method and year.") }}
<figcaption class="small">Methods outside the top {{ n_top }} are grouped as Other. The latest year in the file may be incomplete.</figcaption>
<details><summary>Table view: planets by method</summary>{{ method_table }}</details>

<h2>2. Planet size against orbit</h2>
<p>{{ scatter_text }}</p>
{{ fig("scatter", "Scatter plot of orbital period against planet radius on log axes, colored by discovery method.") }}

<h2>3. Is there a gap in the sizes of small planets?</h2>
<p>{{ gap_text }}</p>
{{ fig("gap", "Histogram of planet radii between 1 and 4 Earth radii with a smoothed density curve.") }}
{% if precise_text %}<p>{{ precise_text }}</p>
{{ fig("gap_precise", "Histogram of planet radii with precise measurements and a smoothed density curve.") }}{% endif %}
<p class="small">{{ gap_context }}</p>

<h2>4. How much of a planet's size can its orbit and star explain?</h2>
<p>{{ model_text }}</p>
{% if model %}
{{ fig("importance", "Bar chart of permutation importance for each model feature.") }}
{{ fig("predobs", "Binned scatter of predicted against observed planet radius on held-out systems.") }}
<details><summary>Table view: model comparison</summary>{{ model_table }}</details>
<details><summary>Table view: feature importance</summary>{{ importance_table }}</details>
{% endif %}

<h2>Method and limits</h2>
<ul>
<li><strong>Pipeline.</strong> The CSV export is validated and cleaned, stored in DuckDB, and the trend counts come from SQL. Planets with non-positive physical values have those values set to missing.</li>
<li><strong>Gap detection.</strong> A kernel density estimate of log radius is searched for two prominent peaks. A dip counts only if it falls to {{ min_depth }} of the lower peak or less. Resampling planets gives the interval.</li>
<li><strong>Model.</strong> Gradient boosting is compared with a ridge regression and with predicting the average. Folds are grouped by host star so planets from one system never appear on both sides of a split.</li>
<li><strong>Selection effects.</strong> Which planets exist in the table depends on what each method can detect. Transit surveys find close-in planets with measurable radii, and radial velocity mostly gives masses. The patterns here describe the detected population, not all planets.</li>
<li><strong>Data quality.</strong> The composite table mixes measurements of different precision, and the model and trend analyses ignore the stated uncertainties. The gap search is repeated on precisely measured planets as a check.</li>
<li><strong>Interpretation.</strong> The model shows association, not cause.</li>
{% if missing_columns %}<li><strong>Columns missing from this file:</strong> {{ missing_columns }}. Related analyses ran on what was available.</li>{% endif %}
</ul>

<h2>Data provenance</h2>
<p class="small">File <code>{{ meta.file_name }}</code> (SHA-256 prefix <code>{{ meta.sha256_short }}</code>). {{ meta.rows_in }} rows read, {{ meta.rows_out }} kept after cleaning ({{ meta.dropped_duplicates }} duplicates and {{ meta.dropped_non_default }} non-default rows removed). Report generated {{ generated }}.</p>
{% if meta.header_comments %}<details><summary>Header lines from the archive file</summary><p class="small">{% for c in meta.header_comments %}{{ c }}<br>{% endfor %}</p></details>{% endif %}
</main>
</body>
</html>
"""


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def _table(df: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if pd.isna(v):
                s = ""
            elif c in formats:
                s = formats[c](v)
            else:
                s = str(v)
            cells.append(f"<td>{html.escape(s)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _gap_text(gap: GapResult) -> str:
    region = (
        f"{gap.n:,} planets with orbital periods under {gap.period_max:g} days and radii "
        f"between {gap.r_min:g} and {gap.r_max:g} Earth radii"
    )
    if gap.found:
        lo, hi = gap.peaks_re
        text = (
            f"Among {region}, the size distribution has two peaks, near {lo:.2f} and {hi:.2f} "
            f"Earth radii, with a dip near {gap.gap_re:.2f}. At the bottom of the dip the density is "
            f"{gap.depth_ratio:.0%} of the lower peak."
        )
        if gap.ci_low is not None:
            text += (
                f" Resampling the planets puts the dip between {gap.ci_low:.2f} and {gap.ci_high:.2f} "
                f"(95% interval), and a dip was found in {gap.detect_rate:.0%} of resamples."
            )
        return text
    return f"Among {region}, no clear gap was found: {gap.reason}."


def _precise_text(gap: GapResult, precise: GapResult) -> str:
    """Robustness check: repeat the search on planets whose radius is well measured."""
    cut = f"{precise.max_rel_err:.0%}"
    head = (
        f"Robustness check: the same search on the {precise.n:,} of those planets whose stated radius "
        f"uncertainty is at most {cut} of the radius."
    )
    if precise.found:
        lo, hi = precise.peaks_re
        text = (
            f" Here the distribution has two peaks, near {lo:.2f} and {hi:.2f} Earth radii, with a dip near "
            f"{precise.gap_re:.2f} ({precise.depth_ratio:.0%} of the lower peak)."
        )
        if precise.ci_low is not None:
            text += (
                f" The 95% resampling interval for the dip is {precise.ci_low:.2f} to {precise.ci_high:.2f}, "
                f"and a dip appeared in {precise.detect_rate:.0%} of resamples."
            )
        if not gap.found:
            text += " The dip is hidden in the full sample, which is consistent with noisy radii blurring it."
        return head + text
    return head + f" No clear gap here either: {precise.reason}."


def _model_text(model: ModelResult | None) -> str:
    if model is None:
        return "There were too few planets with a measured radius and orbital period to fit the model."
    m = model.metrics.set_index("model")
    gbm, ridge = m.loc["Gradient boosting"], m.loc["Linear (ridge)"]
    top = model.importance.iloc[0]["feature"].lower()
    return (
        f"Using orbital period and four host star properties, gradient boosting predicted planet radius "
        f"on {model.n_splits}-fold cross-validation grouped by host star ({model.n:,} planets in "
        f"{model.n_groups:,} systems) with R squared {gbm['r2']:.2f}, against {ridge['r2']:.2f} for a "
        f"linear baseline. A typical prediction is off by a factor of about {gbm['error_factor']:.2f}. "
        f"Shuffling {top} hurt the model most."
    )


def build_report(
    *,
    df: pd.DataFrame,
    meta: dict,
    summary: dict,
    wide: pd.DataFrame,
    totals: pd.DataFrame,
    gap: GapResult,
    model: ModelResult | None,
    precise_gap: GapResult | None = None,
    synthetic: bool = False,
    generated: str | None = None,
) -> str:
    top_methods = [c for c in wide.columns if c != "Other"]
    themes = ("light", "dark")

    images = {
        "cumulative": {t: plots.fig_cumulative(wide, t) for t in themes},
        "scatter": {
            t: plots.fig_period_radius(df, top_methods, t, gap.period_max, gap.r_min, gap.r_max)
            for t in themes
        },
        "gap": {t: plots.fig_radius_gap(gap, t) for t in themes},
    }
    use_precise = precise_gap is not None and precise_gap.n >= 50
    if use_precise:
        images["gap_precise"] = {t: plots.fig_radius_gap(precise_gap, t) for t in themes}
    if model is not None:
        images["importance"] = {t: plots.fig_importance(model.importance, t) for t in themes}
        images["predobs"] = {t: plots.fig_pred_obs(model.oof, t) for t in themes}

    def fig(name: str, alt: str) -> str:
        pair = images[name]
        return (
            "<figure><picture>"
            f'<source media="(prefers-color-scheme: dark)" srcset="data:image/png;base64,{_b64(pair["dark"])}">'
            f'<img src="data:image/png;base64,{_b64(pair["light"])}" alt="{html.escape(alt)}">'
            "</picture></figure>"
        )

    totals_by_year = wide.sum(axis=1)
    peak_year = int(totals_by_year.idxmax())
    lead = wide.sum().idxmax()
    lead_share = wide[lead].sum() / wide.to_numpy().sum()
    trend_text = (
        f"The file lists {summary['n_planets']:,} confirmed planets discovered from {summary['year_min']} "
        f"to {summary['year_max']}. {lead} accounts for {lead_share:.0%} of them, and {peak_year} was the "
        f"busiest year with {int(totals_by_year.max()):,} discoveries."
    )

    n_plotted = int(df[["pl_orbper", "pl_rade"]].dropna().shape[0])
    scatter_text = (
        f"{n_plotted:,} planets have both an orbital period and a radius. The dashed box marks the "
        f"region used in the next section."
    )

    fmt_pct = {"r2": lambda v: f"{v:.3f}", "rmse_dex": lambda v: f"{v:.3f}", "error_factor": lambda v: f"{v:.2f}"}
    context = {
        "synthetic": synthetic,
        "tiles": [
            (f"{summary['n_planets']:,}", "confirmed planets"),
            (f"{summary['n_hosts']:,}", "host stars"),
            (f"{summary['n_multi']:,}", "multi-planet systems"),
            (f"{summary['n_radius']:,}", "with a measured radius"),
            (f"{summary['n_mass']:,}", "with a measured mass"),
        ],
        "trend_text": trend_text,
        "scatter_text": scatter_text,
        "gap_text": _gap_text(gap),
        "precise_text": _precise_text(gap, precise_gap) if use_precise else "",
        "gap_context": (
            "For context, Fulton et al. (2017) reported a dip in this size range using precise "
            "California-Kepler Survey radii. The composite table mixes measurement quality, so a blurrier "
            "dip here would not be surprising."
        ),
        "model": model,
        "model_text": _model_text(model),
        "n_top": len(top_methods),
        "min_depth": "90%",
        # _table escapes every cell itself, so the finished markup is marked safe for Jinja.
        "method_table": _Safe(_table(totals, {"planets": lambda v: f"{int(v):,}", "first_year": lambda v: str(int(v)), "last_year": lambda v: str(int(v))})),
        "model_table": _Safe(_table(
            model.metrics.rename(columns={"model": "Model", "r2": "R squared", "rmse_dex": "Error (log10 units)", "error_factor": "Typical error factor"}),
            {"R squared": fmt_pct["r2"], "Error (log10 units)": fmt_pct["rmse_dex"], "Typical error factor": fmt_pct["error_factor"]},
        )) if model else "",
        "importance_table": _Safe(_table(
            model.importance.rename(columns={"feature": "Feature", "mean_drop": "Mean drop in R squared", "std_drop": "Std across folds"}),
            {"Mean drop in R squared": lambda v: f"{v:.4f}", "Std across folds": lambda v: f"{v:.4f}"},
        )) if model else "",
        "missing_columns": ", ".join(meta.get("missing_columns", [])),
        "meta": meta,
        "generated": generated or dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    env = Environment(autoescape=select_autoescape(["html"]))
    template = env.from_string(TEMPLATE)
    return template.render(fig=lambda n, a: _Safe(fig(n, a)), **context)


class _Safe(str):
    """Marks pre-built HTML as safe for Jinja autoescape."""

    def __html__(self) -> str:
        return str(self)
