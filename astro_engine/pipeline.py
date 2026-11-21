"""End-to-end: minute closes -> bars -> astro conditions -> mined rules."""
from __future__ import annotations

import time
from dataclasses import asdict

import pandas as pd

from .ephemeris import compute_positions
from .features import FeatureConfig, build_features
from .market import forward_returns, labels, to_bars
from .miner import MinerConfig, mine

DEFAULT_SETTINGS = {
    "bar": "1D",            # "1D", "60min", "30min", "15min", ...
    "horizon": 1,           # bars ahead the rise/fall is measured over
    "threshold_pct": 0.0,   # a move must exceed this (in %) to count as rise/fall
    "zodiac": "sidereal",   # "sidereal" (Lahiri / Vedic) or "tropical"
    "features": {},
    "miner": {},
}


def analyze(close: pd.Series, settings: dict | None = None) -> dict:
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    t0 = time.time()
    bars = to_bars(close, s["bar"])
    fwd = forward_returns(bars, s["bar"], int(s["horizon"]))
    if int(s["horizon"]) > 1:
        # Returns over h bars measured from every bar overlap, so one market move is
        # counted h times and every significance test becomes overconfident.
        # Keep only non-overlapping windows.
        fwd = fwd.iloc[:: int(s["horizon"])]
    lab = labels(fwd, float(s["threshold_pct"]) / 100.0)
    fcfg = FeatureConfig.from_dict(s["features"])
    mcfg = MinerConfig.from_dict(s["miner"])
    if s["bar"].upper() not in ("1D", "D", "DAILY") and not s["miner"].get("min_support"):
        # Intraday bars are far more numerous; scale default support accordingly.
        mcfg.min_support = 150
        mcfg.min_support_validation = 50
    feats = build_features(compute_positions(fwd.index, s["zodiac"]), fcfg)
    result = mine(feats, fwd, lab, mcfg)
    return {
        "settings": {**s, "features": asdict(fcfg), "miner": asdict(mcfg)},
        "data": {"minutes": int(len(close)), "first": str(close.index[0]), "last": str(close.index[-1])},
        "runtime_seconds": round(time.time() - t0, 2),
        "result": result,
    }
