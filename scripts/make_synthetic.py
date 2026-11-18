"""Generate synthetic Nifty-like minute data for trying the engine without an API.

Usage: python scripts/make_synthetic.py --years 6 --out data/synthetic.csv
       python scripts/make_synthetic.py --plant "Moon in Rohini nakshatra" ...

The series is a random walk, so a correct engine should find (almost) nothing in
it. ``--plant`` injects an artificial edge on days when a condition holds, to
check that the engine *does* detect a real effect when one exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astro_engine.ephemeris import compute_positions  # noqa: E402
from astro_engine.features import build_features  # noqa: E402
from astro_engine.market import future_bar_times  # noqa: E402


def make(years: float = 6, start: str = "2016-01-01", plant: str | None = None,
         plant_bp: float = 40.0, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(start) + pd.Timedelta(days=int(365 * years))
    idx = future_bar_times(pd.Timestamp(start), end, "1min")
    n = len(idx)
    rets = rng.normal(0.0000015, 0.0006, n)  # ~14%/yr drift, ~1.2% daily vol
    if plant:
        days = idx.normalize().unique()
        close_t = days + pd.Timedelta(hours=15, minutes=30)
        feats = build_features(compute_positions(pd.DatetimeIndex(close_t), "sidereal"))
        # condition at day t's close -> push day t+1 up
        on = pd.Series(feats[plant].to_numpy().astype(bool), index=days).shift(1, fill_value=False)
        per_min = plant_bp / 1e4 / 375
        rets += pd.Series(idx.normalize()).map(on).to_numpy().astype(float) * per_min
    close = 8000 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"timestamp": idx.tz_localize(None), "close": close.round(2)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=6)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--plant", default=None)
    ap.add_argument("--plant-bp", type=float, default=40.0)
    ap.add_argument("--out", default="data/synthetic.csv")
    a = ap.parse_args()
    df = make(a.years, a.start, a.plant, a.plant_bp)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} minutes to {a.out}")
