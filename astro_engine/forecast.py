"""Apply mined rules to future planetary positions, and list upcoming astro events."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ephemeris import BODIES, SIGNS, compute_positions, sign_index, angular_distance
from .features import ASPECTS, FeatureConfig, build_features
from .market import MARKET_TZ, future_bar_times


def forecast(model: dict, start, end, include: str = "validated") -> dict:
    """Score each future bar with the model's rules.

    ``include``: ``"validated"`` (default) uses only rules that passed validation;
    ``"all"`` also uses discovery-only rules (exploratory, not trustworthy).
    """
    bar = model["settings"]["bar"]
    zodiac = model["settings"]["zodiac"]
    fcfg = FeatureConfig.from_dict(model["settings"].get("features"))
    rules = [r for r in model["result"]["rules"] if include == "all" or r["status"] == "validated"]

    times = future_bar_times(start, end, bar)
    if len(times) == 0:
        return {"bar": bar, "rules_used": len(rules), "points": []}
    feats = build_features(compute_positions(times, zodiac), fcfg)

    score = np.zeros(len(times))
    fired: list[list[dict]] = [[] for _ in range(len(times))]
    for rule in rules:
        missing = [c for c in rule["conditions"] if c not in feats.columns]
        if missing:
            continue
        v = np.ones(len(times), dtype=bool)
        for c in rule["conditions"]:
            v &= feats[c].to_numpy().astype(bool)
        score += v * rule["weight"]
        for i in np.nonzero(v)[0]:
            fired[i].append({"conditions": rule["conditions"], "direction": rule["direction"],
                             "weight": round(rule["weight"], 4)})

    points = []
    for i, t in enumerate(times):
        s = float(score[i])
        signal = "POSITIVE" if s > 0 else "NEGATIVE" if s < 0 else "NEUTRAL"
        points.append({"time": t.isoformat(), "signal": signal, "score": round(s, 4), "rules": fired[i]})
    return {
        "bar": bar,
        "rules_used": len(rules),
        "note": "Exchange holidays are not excluded. Signals are only as good as the validation results.",
        "points": points,
    }


def astro_events(start, end, zodiac: str = "sidereal", include_moon: bool = False, astro: dict | None = None) -> list[dict]:
    """Sign ingresses, retrograde/direct stations and exact major aspects."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize(MARKET_TZ) if start.tzinfo is None else start
    end = end.tz_localize(MARKET_TZ) if end.tzinfo is None else end
    times = pd.date_range(start, end, freq="1h")
    if len(times) < 2:
        return []
    if astro:
        from .pipeline import astro_kwargs
        pos = compute_positions(times, **astro_kwargs({"astro": astro}))
    else:
        pos = compute_positions(times, zodiac)
    lon, spd = pos.longitude, pos.speed
    bodies = [b for b in lon.columns if include_moon or b != "Moon"]
    events = []

    for b in bodies:
        s = sign_index(lon[b].to_numpy())
        for i in np.nonzero(s[1:] != s[:-1])[0]:
            events.append({"time": times[i + 1].isoformat(), "type": "ingress", "body": b,
                           "text": f"{b} enters {SIGNS[s[i + 1]]}"})
        if b in ("Sun", "Moon", "Rahu", "Ketu"):
            continue
        sp = spd[b].to_numpy()
        for i in np.nonzero(np.sign(sp[1:]) != np.sign(sp[:-1]))[0]:
            kind = "retrograde" if sp[i + 1] < 0 else "direct"
            events.append({"time": times[i + 1].isoformat(), "type": "station", "body": b,
                           "text": f"{b} stations {kind}"})

    for ai, a in enumerate(bodies):
        for b in bodies[ai + 1:]:
            if {a, b} == {"Rahu", "Ketu"}:
                continue
            sep = angular_distance(lon[a].to_numpy(), lon[b].to_numpy())
            for asp, (angle, _) in ASPECTS.items():
                d = sep - angle
                # exact when the signed distance to the aspect angle crosses zero
                cross = np.nonzero((np.sign(d[1:]) != np.sign(d[:-1])) & (np.abs(d[1:]) < 2))[0]
                if angle == 0.0:
                    cross = np.nonzero((np.diff(np.sign(np.diff(sep))) > 0) & (sep[1:-1] < 1.5))[0]
                if angle == 180.0:
                    cross = np.nonzero((np.diff(np.sign(np.diff(sep))) < 0) & (sep[1:-1] > 178.5))[0]
                for i in cross:
                    events.append({"time": times[i + 1].isoformat(), "type": "aspect", "body": f"{a}/{b}",
                                   "text": f"{a} {asp} {b} (exact)"})
    events.sort(key=lambda e: e["time"])
    return events
