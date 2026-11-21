"""End-to-end: minute closes -> bars -> astro conditions -> mined rules."""
from __future__ import annotations

import time
from dataclasses import asdict

import pandas as pd

from .ephemeris import compute_positions
from .features import FeatureConfig, build_features
from .market import forward_returns, labels, to_bars
from .miner import MinerConfig, mine

DEFAULT_ASTRO = {
    "ephemeris": "modern",      # "modern" or "surya_siddhanta"
    "ayanamsa": "lahiri",       # see ephemeris.AYANAMSAS; "tropical" = Western
    "node": "mean",             # "mean" or "true" Rahu/Ketu
    "outer_planets": True,      # include Uranus, Neptune, Pluto
}

# How hard a pattern must work to be kept. Each preset fills MinerConfig fields.
STRICTNESS = {
    "strict": {"discovery_method": "fdr", "alpha": 0.05, "validation_method": "fdr", "alpha_validation": 0.10},
    "balanced": {"discovery_method": "fdr", "alpha": 0.20, "validation_method": "raw", "alpha_validation": 0.05},
    "exploratory": {"discovery_method": "raw", "alpha": 0.01, "validation_method": "raw", "alpha_validation": 0.10},
}

DEFAULT_SETTINGS = {
    "bar": "1D",            # "1D", "60min", "30min", "15min", ...
    "horizon": 1,           # bars ahead the rise/fall is measured over
    "threshold_pct": 0.0,   # a move must exceed this (in %) to count as rise/fall
    "zodiac": "sidereal",   # kept for older clients; "astro.ayanamsa" takes precedence
    "astro": DEFAULT_ASTRO,
    "strictness": "strict",  # strict | balanced | exploratory | custom
    "features": {},
    "miner": {},
}


def astro_kwargs(settings: dict) -> dict:
    a = {**DEFAULT_ASTRO, **(settings.get("astro") or {})}
    if "astro" not in settings and settings.get("zodiac") == "tropical":
        a["ayanamsa"] = "tropical"
    return {"ephemeris": a["ephemeris"], "ayanamsa": a["ayanamsa"], "node": a["node"],
            "outer_planets": bool(a["outer_planets"])}


def features_for(index: pd.DatetimeIndex, settings: dict) -> pd.DataFrame:
    return build_features(compute_positions(index, **astro_kwargs(settings)),
                          FeatureConfig.from_dict(settings.get("features")))


def market_labels(close: pd.Series, settings: dict) -> tuple[pd.Series, pd.Series]:
    bar, h = settings["bar"], int(settings["horizon"])
    bars = to_bars(close, bar)
    fwd = forward_returns(bars, bar, h)
    if h > 1:
        # Returns over h bars measured from every bar overlap, so one market move is
        # counted h times and every significance test becomes overconfident.
        # Keep only non-overlapping windows.
        fwd = fwd.iloc[::h]
    return fwd, labels(fwd, float(settings["threshold_pct"]) / 100.0)


def resolve_settings(settings: dict | None) -> dict:
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    s["astro"] = {**DEFAULT_ASTRO, **(s.get("astro") or {})}
    if (settings or {}).get("zodiac") == "tropical" and "astro" not in (settings or {}):
        s["astro"]["ayanamsa"] = "tropical"
    s["zodiac"] = "tropical" if s["astro"]["ayanamsa"] == "tropical" and s["astro"]["ephemeris"] == "modern" else "sidereal"
    miner = dict(s.get("miner") or {})
    if s.get("strictness") in STRICTNESS:
        miner = {**miner, **STRICTNESS[s["strictness"]]}
    if s["bar"].upper() not in ("1D", "D", "DAILY") and "min_support" not in miner:
        # Intraday bars are far more numerous; scale default support accordingly.
        miner.setdefault("min_support", 150)
        miner.setdefault("min_support_validation", 50)
    if int(s["horizon"]) > 1 and "min_support" not in miner:
        miner["min_support"] = max(10, 30 // int(s["horizon"]) * 2)
        miner["min_support_validation"] = 5
    s["miner"] = miner
    return s


def analyze(close: pd.Series, settings: dict | None = None) -> dict:
    s = resolve_settings(settings)
    t0 = time.time()
    fwd, lab = market_labels(close, s)
    fcfg = FeatureConfig.from_dict(s["features"])
    mcfg = MinerConfig.from_dict(s["miner"])
    feats = features_for(fwd.index, s)
    result = mine(feats, fwd, lab, mcfg)
    return {
        "settings": {**s, "features": asdict(fcfg), "miner": asdict(mcfg)},
        "data": {"minutes": int(len(close)), "first": str(close.index[0]), "last": str(close.index[-1])},
        "runtime_seconds": round(time.time() - t0, 2),
        "result": result,
    }
