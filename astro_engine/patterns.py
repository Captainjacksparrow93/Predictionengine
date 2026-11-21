"""Saved pattern sets, predictions from them, and checks against real prices.

A *pattern set* is the full result of one search (settings, every rule found,
the periods used). It is stored under ``DATA_DIR/patterns/<id>.json``.

* :func:`predict` scores future (or past) bars with a set's rules.
* :func:`reality_check` compares those calls with what the market actually did.
* The *journal* records predictions made in advance so they can be scored once
  the real prices arrive.
"""
from __future__ import annotations

import json
import re
import uuid

import numpy as np
import pandas as pd
from scipy.stats import binom

from . import storage
from .market import MARKET_TZ, future_bar_times
from .pipeline import features_for, market_labels

# ----------------------------------------------------------------------------- library


def _dir():
    d = storage.data_dir() / "patterns"
    d.mkdir(exist_ok=True)
    return d


def _clean_id(pid: str) -> str:
    if not re.fullmatch(r"[a-z0-9]{6,32}", pid or ""):
        raise KeyError(pid)
    return pid


def save(model: dict, name: str) -> dict:
    pid = uuid.uuid4().hex[:12]
    model = {**model, "id": pid, "name": name}
    (_dir() / f"{pid}.json").write_text(json.dumps(model, default=str))
    if not storage.load_json("active_pattern.json"):
        set_active(pid)
    return model


def load(pid: str) -> dict:
    p = _dir() / f"{_clean_id(pid)}.json"
    if not p.exists():
        raise KeyError(pid)
    return json.loads(p.read_text())


def update(pid: str, **fields) -> dict:
    m = load(pid)
    m.update({k: v for k, v in fields.items() if k in ("name", "notes")})
    (_dir() / f"{pid}.json").write_text(json.dumps(m, default=str))
    return m


def delete(pid: str) -> None:
    (_dir() / f"{_clean_id(pid)}.json").unlink(missing_ok=True)
    if (storage.load_json("active_pattern.json") or {}).get("id") == pid:
        storage.save_json("active_pattern.json", {})


def set_active(pid: str) -> None:
    load(pid)
    storage.save_json("active_pattern.json", {"id": pid})


def active_id() -> str | None:
    return (storage.load_json("active_pattern.json") or {}).get("id")


def summary(m: dict) -> dict:
    r = m["result"]
    rules = r["rules"]
    return {
        "id": m["id"], "name": m["name"], "created": m.get("created"), "label": m.get("label"),
        "market": (m.get("dataset") or {}).get("name"),
        "settings": {k: m["settings"].get(k) for k in ("bar", "horizon", "threshold_pct", "astro", "strictness")},
        "periods": r["periods"],
        "tested": r["search"]["hypotheses_tested"],
        "patterns": sum(x["status"] == "validated" for x in rules),
        "candidates": len(rules),
        "chance": r["null_control"]["mean"],
        "level": r.get("verdict_level"),
        "verdict": r["verdict"],
        "active": m["id"] == active_id(),
    }


def list_all() -> list[dict]:
    out = []
    for p in _dir().glob("*.json"):
        try:
            out.append(summary(json.loads(p.read_text())))
        except Exception:  # noqa: BLE001 - skip unreadable files
            continue
    return sorted(out, key=lambda x: x.get("created") or "", reverse=True)


# ----------------------------------------------------------------------------- scoring

def _target(t: pd.Timestamp, settings: dict) -> str:
    """The day (or time) a call made at bar ``t`` is about."""
    h = int(settings.get("horizon", 1))
    if settings["bar"].upper() in ("1D", "D", "DAILY"):
        return str((t + pd.offsets.BDay(h)).date())
    return (t + pd.Timedelta(settings["bar"]) * (h + 1)).strftime("%Y-%m-%d %H:%M")


def _rules(m: dict, include: str) -> list[dict]:
    keep = {"validated"} if include == "validated" else {"validated", "failed_validation", "not_testable"}
    return [r for r in m["result"]["rules"] if r["status"] in keep]


def score(m: dict, index: pd.DatetimeIndex, include: str = "validated", min_score: float = 0.0):
    """Signal score per timestamp plus which rules fired. Positive = up."""
    rules = _rules(m, include)
    feats = features_for(index, m["settings"])
    s = np.zeros(len(index))
    fired: list[list[dict]] = [[] for _ in range(len(index))]
    for rule in rules:
        if any(c not in feats.columns for c in rule["conditions"]):
            continue
        v = np.ones(len(index), dtype=bool)
        for c in rule["conditions"]:
            v &= feats[c].to_numpy().astype(bool)
        s += v * rule["weight"]
        for i in np.nonzero(v)[0]:
            fired[i].append({"conditions": rule["conditions"], "direction": rule["direction"],
                             "weight": round(rule["weight"], 4), "status": rule["status"]})
    sig = np.where(s > min_score, 1, np.where(s < -min_score, -1, 0))
    return s, sig, fired, len(rules)


def predict(m: dict, start, end, include: str = "validated", min_score: float = 0.0) -> dict:
    bar = m["settings"]["bar"]
    times = future_bar_times(start, end, bar)
    if int(m["settings"].get("horizon", 1)) > 1:
        times = times[:: int(m["settings"]["horizon"])]
    if len(times) == 0:
        return {"bar": bar, "rules_used": 0, "points": []}
    s, sig, fired, n_rules = score(m, times, include, min_score)
    word = {1: "UP", -1: "DOWN", 0: "NO CALL"}
    return {
        "pattern_id": m["id"], "bar": bar, "horizon": m["settings"].get("horizon", 1), "rules_used": n_rules,
        "points": [{"time": t.isoformat(), "target": _target(t, m["settings"]), "call": word[int(sig[i])], "score": round(float(s[i]), 4),
                    "rules": fired[i]} for i, t in enumerate(times)],
    }


def reality_check(m: dict, close: pd.Series, start=None, end=None, include: str = "validated",
                  min_score: float = 0.0) -> dict:
    """Apply a pattern set to real past bars and compare calls with what happened."""
    fwd, lab = market_labels(close, m["settings"])
    idx = fwd.index
    if start:
        idx = idx[idx >= pd.Timestamp(start).tz_localize(MARKET_TZ) if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)]
    if end:
        e = pd.Timestamp(end)
        e = e.tz_localize(MARKET_TZ) if e.tzinfo is None else e
        idx = idx[idx <= e + pd.Timedelta(days=1)]
    if len(idx) == 0:
        raise ValueError("no market data in that period")
    fwd, lab = fwd.loc[idx], lab.loc[idx]
    s, sig, fired, n_rules = score(m, idx, include, min_score)
    y = lab.to_numpy()
    r = fwd.to_numpy()
    called = sig != 0
    right = ((sig == 1) & (y == 1)) | ((sig == -1) & (y == -1))
    n_called = int(called.sum())
    p_up, p_dn = float((y == 1).mean()), float((y == -1).mean())
    share_up = float((sig[called] == 1).mean()) if n_called else 0.0
    chance = share_up * p_up + (1 - share_up) * p_dn
    correct = int(right[called].sum())
    strat = np.where(called, sig * r, 0.0)

    # Overlap with the years used to find the patterns -> optimistic results.
    per = m["result"]["periods"]
    learn_end = pd.Timestamp(per["validation"]["end"])
    overlap = int((idx <= learn_end).sum())

    def block(mask):
        n = int(mask.sum())
        return {"calls": n, "right": int(right[mask].sum()), "rate": float(right[mask].mean()) if n else None}

    rows = [{"time": t.isoformat(), "target": _target(t, m["settings"]), "call": {1: "UP", -1: "DOWN", 0: "NO CALL"}[int(sig[i])],
             "score": round(float(s[i]), 4), "actual_pct": round(float(r[i]) * 100, 3),
             "actual": {1: "UP", -1: "DOWN", 0: "FLAT"}[int(y[i])],
             "right": bool(right[i]) if called[i] else None,
             "rules": [x["conditions"] for x in fired[i]]} for i, t in enumerate(idx)]
    return {
        "pattern_id": m["id"], "rules_used": n_rules, "bars": int(len(idx)),
        "start": str(idx[0]), "end": str(idx[-1]),
        "calls": n_called, "right": correct,
        "hit_rate": correct / n_called if n_called else None,
        "chance_rate": chance if n_called else None,
        "p_value": float(binom.sf(correct - 1, n_called, chance)) if n_called else None,
        "up_calls": block(sig == 1), "down_calls": block(sig == -1),
        "normal_up_rate": p_up,
        "following_return_pct": float((np.prod(1 + strat) - 1) * 100),
        "buy_hold_return_pct": float((np.prod(1 + r) - 1) * 100),
        "equity": {"time": [t.isoformat() for t in idx],
                   "following": (np.cumprod(1 + strat) * 100).round(3).tolist(),
                   "buy_hold": (np.cumprod(1 + r) * 100).round(3).tolist()},
        "overlap_with_learning": overlap,
        "learning_end": str(learn_end),
        "rows": rows,
    }


# ----------------------------------------------------------------------------- journal

def journal_add(m: dict, pred: dict) -> dict:
    j = storage.load_json("journal.json") or []
    entry = {"id": uuid.uuid4().hex[:10], "pattern_id": m["id"], "pattern_name": m["name"],
             "made": pd.Timestamp.now(tz=MARKET_TZ).isoformat(),
             "calls": [{"time": p["time"], "target": p.get("target"), "call": p["call"], "score": p["score"]}
                       for p in pred["points"] if p["call"] != "NO CALL"]}
    j.insert(0, entry)
    storage.save_json("journal.json", j[:200])
    return entry


def journal_scored(close: pd.Series | None) -> list[dict]:
    """Journal entries with each call marked right/wrong where real prices exist."""
    j = storage.load_json("journal.json") or []
    if close is None or not j:
        return j
    cache: dict[str, tuple] = {}
    out = []
    for e in j:
        try:
            m = load(e["pattern_id"])
        except KeyError:
            out.append({**e, "missing_pattern": True})
            continue
        key = json.dumps({k: m["settings"][k] for k in ("bar", "horizon", "threshold_pct")})
        if key not in cache:
            cache[key] = market_labels(close, m["settings"])
        fwd, lab = cache[key]
        calls, right, done = [], 0, 0
        for c in e["calls"]:
            t = pd.Timestamp(c["time"])
            actual = None
            if t in lab.index:
                actual = {1: "UP", -1: "DOWN", 0: "FLAT"}[int(lab.loc[t])]
                done += 1
                right += int(actual == c["call"])
            calls.append({**c, "actual": actual})
        out.append({**e, "calls": calls, "checked": done, "right": right})
    return out
