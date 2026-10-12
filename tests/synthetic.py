"""Synthetic stand-in for the archive table, used only to test the code.

Short-period small planets are drawn from two size populations (about 1.3 and 2.4
Earth radii), so the radius gap analysis has a known answer to recover. Nothing here
is real exoplanet data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OTHER_METHODS = ["Imaging", "Eclipse Timing Variations", "Transit Timing Variations"]


def make_planets(n_systems: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.arange(1995, 2027)
    year_w = np.exp(0.15 * (years - 1995))
    year_w /= year_w.sum()

    rows = []
    for i in range(n_systems):
        host = f"SYN-{i:05d}"
        teff = max(3000.0, rng.normal(5500, 700))
        st_rad = float(np.clip(rng.lognormal(0.0, 0.25), 0.2, 3.0))
        st_mass = float(np.clip(st_rad**0.8 * rng.lognormal(0, 0.08), 0.1, 2.5))
        dist = float(rng.lognormal(np.log(400), 0.8))
        n_pl = int(rng.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05]))
        for j in range(n_pl):
            log_p = float(np.clip(rng.normal(0.9, 0.7), -0.5, 3.5))
            # Giant planets are more likely around heavier stars and at longer periods.
            # The dependence is strong on purpose so tests can check that the model learns it.
            logit = 5.0 * (st_mass - 1.0) + 1.2 * (log_p - 1.0) - 0.8
            p_giant = float(1.0 / (1.0 + np.exp(-logit)))
            if rng.random() < p_giant:
                log_r = rng.normal(np.log10(11.0), 0.12)
            elif log_p < 2.0:
                center = 1.3 if rng.random() < 0.5 else 2.4
                log_r = rng.normal(np.log10(center), 0.06)
            else:
                log_r = rng.normal(np.log10(2.0), 0.2)
            radius = 10**log_r
            year = int(rng.choice(years, p=year_w))
            if year < 2010:
                method = rng.choice(["Radial Velocity", "Transit", "Microlensing", "Imaging"], p=[0.8, 0.1, 0.05, 0.05])
            else:
                method = rng.choice(
                    ["Transit", "Radial Velocity", "Microlensing", "Imaging", "Eclipse Timing Variations", "Transit Timing Variations"],
                    p=[0.75, 0.13, 0.05, 0.03, 0.02, 0.02],
                )
            rows.append(
                {
                    "pl_name": f"{host} {chr(98 + j)}",
                    "hostname": host,
                    "discoverymethod": str(method),
                    "disc_year": year,
                    "pl_orbper": 10**log_p,
                    "pl_rade": radius if rng.random() > (0.05 if method == "Transit" else 0.6) else np.nan,
                    "pl_bmasse": radius**1.7 * rng.lognormal(0, 0.2) if rng.random() < 0.5 else np.nan,
                    "pl_eqt": float(rng.normal(900, 400)),
                    "pl_orbsmax": float((10**log_p) ** 0.667 / 100),
                    "st_teff": teff if rng.random() > 0.04 else np.nan,
                    "st_rad": st_rad if rng.random() > 0.05 else np.nan,
                    "st_mass": st_mass if rng.random() > 0.10 else np.nan,
                    "sy_dist": dist if rng.random() > 0.01 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_archive_style_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write the frame the way the archive does: comment header lines, then a header row."""
    header = [
        "# SYNTHETIC TEST FILE, not produced by the NASA Exoplanet Archive",
        "# COLUMN pl_name: Planet Name",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        df.to_csv(fh, index=False)
    return path
