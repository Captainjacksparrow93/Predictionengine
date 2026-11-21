"""Search planetary-condition combinations for links to index rises and falls.

Testing hundreds of thousands of combinations against one price history will
always turn up "patterns" by chance alone, so the search is built around three
controls:

1. **Chronological three-way split.** Rules are *discovered* on the oldest
   slice, *validated* on the next slice and finally *tested* on the newest
   slice, which no step of the search ever looked at.
2. **False discovery rate.** Every hypothesis evaluated (each combination, in
   each direction) is counted, and the Benjamini-Hochberg procedure controls the
   expected share of false positives, both at discovery and at validation.
3. **Null control.** The whole pipeline is re-run on the same data with the
   price labels circularly shifted in time. That keeps every statistical
   property of both the market and the planetary series but destroys any real
   link between them. If the shifted runs "find" as many rules as the real run,
   the real rules are indistinguishable from chance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom


@dataclass
class MinerConfig:
    max_depth: int = 2               # 1 = single conditions, 2 = pairs, 3 = triples (beam)
    min_support: int = 30            # min bars where the combination is active (discovery)
    min_episodes: int = 3            # min separate occurrences (discovery)
    min_support_validation: int = 10
    alpha: float = 0.05              # FDR level at discovery
    alpha_validation: float = 0.10   # FDR level at validation
    split: tuple[float, float] = (0.6, 0.2)  # discovery, validation share; rest = test
    triple_beam: int = 300           # best pairs extended to triples when max_depth = 3
    max_candidates: int = 2000       # discovery survivors carried to validation
    null_runs: int = 3
    seed: int = 7

    @classmethod
    def from_dict(cls, d: dict | None) -> "MinerConfig":
        d = d or {}
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "split" in kw:
            kw["split"] = tuple(kw["split"])
        return cls(**kw)


# ----------------------------------------------------------------------------- helpers

def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def _episodes(v: np.ndarray) -> int:
    v = v.astype(bool)
    if len(v) == 0:
        return 0
    return int(v[0]) + int(np.count_nonzero(v[1:] & ~v[:-1]))


def _sf(k: np.ndarray, n: np.ndarray, p0: float) -> np.ndarray:
    """One-sided binomial P(X >= k)."""
    return binom.sf(np.asarray(k) - 1, np.asarray(n), p0)


@dataclass
class _Cands:
    combos: list[tuple[int, ...]]
    n: np.ndarray
    k_up: np.ndarray
    k_down: np.ndarray
    r_sum: np.ndarray


def _enumerate(X: np.ndarray, up: np.ndarray, down: np.ndarray, r: np.ndarray,
               cfg: MinerConfig) -> tuple[_Cands, int]:
    """All combinations up to ``max_depth`` with their counts on X's rows."""
    Xf = X.astype(np.float32)
    p = X.shape[1]
    n1 = Xf.sum(0)
    U1 = (Xf * up[:, None]).sum(0)
    D1 = (Xf * down[:, None]).sum(0)
    R1 = (Xf * r[:, None]).sum(0)

    combos: list[tuple[int, ...]] = [(i,) for i in range(p)]
    n_l, u_l, d_l, r_l = [n1], [U1], [D1], [R1]

    if cfg.max_depth >= 2 and p > 1:
        C = Xf.T @ Xf
        U = Xf.T @ (Xf * up[:, None])
        Dn = Xf.T @ (Xf * down[:, None])
        R = Xf.T @ (Xf * r[:, None].astype(np.float32))
        iu, ju = np.triu_indices(p, 1)
        c = C[iu, ju]
        # Keep pairs with enough support where neither condition implies the other
        # (otherwise the pair is just a duplicate of a single condition).
        keep = (c >= cfg.min_support) & (c < n1[iu]) & (c < n1[ju])
        iu, ju = iu[keep], ju[keep]
        combos += list(zip(iu.tolist(), ju.tolist()))
        n_l.append(C[iu, ju]); u_l.append(U[iu, ju]); d_l.append(Dn[iu, ju]); r_l.append(R[iu, ju])
        del C, U, Dn, R

    cands = _Cands(combos, np.concatenate(n_l).astype(np.int64),
                   np.rint(np.concatenate(u_l)).astype(np.int64),
                   np.rint(np.concatenate(d_l)).astype(np.int64),
                   np.concatenate(r_l).astype(float))

    if cfg.max_depth >= 3 and len(combos) > p:
        cands = _extend_triples(X, Xf, up, down, r, cands, cfg, p)
    return cands, len(cands.combos)


def _extend_triples(X, Xf, up, down, r, cands: _Cands, cfg: MinerConfig, p: int) -> _Cands:
    n0 = len(X)
    base_up, base_dn = up.mean(), down.mean()
    pair_idx = np.arange(p, len(cands.combos))
    pv = np.minimum(_sf(cands.k_up[pair_idx], cands.n[pair_idx], base_up),
                    _sf(cands.k_down[pair_idx], cands.n[pair_idx], base_dn))
    best = pair_idx[np.argsort(pv)[: cfg.triple_beam]]
    if len(best) == 0 or n0 == 0:
        return cands
    P = np.stack([X[:, cands.combos[b][0]] & X[:, cands.combos[b][1]] for b in best], axis=1).astype(np.float32)
    C3 = P.T @ Xf
    U3 = (P * up[:, None]).T @ Xf
    D3 = (P * down[:, None]).T @ Xf
    R3 = (P * r[:, None].astype(np.float32)).T @ Xf
    pair_n = P.sum(0)
    n1 = Xf.sum(0)
    seen = set()
    combos, n_l, u_l, d_l, r_l = [], [], [], [], []
    for bi, b in enumerate(best):
        i, j = cands.combos[b]
        for k in np.nonzero((C3[bi] >= cfg.min_support) & (C3[bi] < pair_n[bi]) & (C3[bi] < n1))[0]:
            if k in (i, j):
                continue
            key = tuple(sorted((i, j, int(k))))
            if key in seen:
                continue
            seen.add(key)
            combos.append(key)
            n_l.append(C3[bi, k]); u_l.append(U3[bi, k]); d_l.append(D3[bi, k]); r_l.append(R3[bi, k])
    if not combos:
        return cands
    return _Cands(
        cands.combos + combos,
        np.concatenate([cands.n, np.rint(n_l).astype(np.int64)]),
        np.concatenate([cands.k_up, np.rint(u_l).astype(np.int64)]),
        np.concatenate([cands.k_down, np.rint(d_l).astype(np.int64)]),
        np.concatenate([cands.r_sum, np.asarray(r_l, dtype=float)]),
    )


def _combo_vec(X: np.ndarray, combo: tuple[int, ...]) -> np.ndarray:
    v = X[:, combo[0]].astype(bool)
    for c in combo[1:]:
        v = v & X[:, c].astype(bool)
    return v


def _period_stats(v: np.ndarray, up: np.ndarray, down: np.ndarray, r: np.ndarray) -> dict:
    n = int(v.sum())
    if n == 0:
        return {"bars": 0, "episodes": 0, "up_rate": None, "down_rate": None, "mean_return_pct": None}
    return {
        "bars": n,
        "episodes": _episodes(v),
        "up_rate": float(up[v].mean()),
        "down_rate": float(down[v].mean()),
        "mean_return_pct": float(r[v].mean() * 100),
    }


# ----------------------------------------------------------------------------- core search

def _discover_validate(X, up, down, r, d_end, v_end, cfg: MinerConfig, detail: bool):
    Xd, Xv = X[:d_end], X[d_end:v_end]
    upd, dnd, rd = up[:d_end], down[:d_end], r[:d_end]
    upv, dnv, rv = up[d_end:v_end], down[d_end:v_end], r[d_end:v_end]
    p_up0, p_dn0 = upd.mean(), dnd.mean()

    cands, n_combos = _enumerate(Xd, upd, dnd, rd, cfg)
    supported = cands.n >= cfg.min_support
    p_bull = np.where(supported, _sf(cands.k_up, cands.n, p_up0), 1.0)
    p_bear = np.where(supported, _sf(cands.k_down, cands.n, p_dn0), 1.0)
    n_tests = 2 * int(supported.sum())
    q_all = bh_qvalues(np.concatenate([p_bull[supported], p_bear[supported]]))
    q_bull = np.ones(len(p_bull)); q_bear = np.ones(len(p_bear))
    k = int(supported.sum())
    q_bull[supported] = q_all[:k]
    q_bear[supported] = q_all[k:]

    disc = []
    for direction, q, pv in (("bullish", q_bull, p_bull), ("bearish", q_bear, p_bear)):
        for idx in np.nonzero(q <= cfg.alpha)[0]:
            disc.append((float(pv[idx]), float(q[idx]), int(idx), direction))
    disc.sort()
    disc = disc[: cfg.max_candidates]

    results = []
    p_up0v, p_dn0v = (upv.mean(), dnv.mean()) if len(upv) else (np.nan, np.nan)
    for pval, qval, idx, direction in disc:
        combo = cands.combos[idx]
        vd = _combo_vec(Xd, combo)
        if _episodes(vd) < cfg.min_episodes:
            continue
        vv = _combo_vec(Xv, combo)
        nv = int(vv.sum())
        if direction == "bullish":
            kv = int(upv[vv].sum()) if nv else 0
            pv_val = float(_sf(kv, nv, p_up0v)) if nv >= cfg.min_support_validation else None
        else:
            kv = int(dnv[vv].sum()) if nv else 0
            pv_val = float(_sf(kv, nv, p_dn0v)) if nv >= cfg.min_support_validation else None
        results.append({"combo": combo, "direction": direction, "p_discovery": pval,
                        "q_discovery": qval, "p_validation": pv_val})

    testable = [i for i, rres in enumerate(results) if rres["p_validation"] is not None]
    qv = bh_qvalues(np.array([results[i]["p_validation"] for i in testable]))
    for i in range(len(results)):
        results[i]["q_validation"] = None
        results[i]["status"] = "not_testable"
    for qi, i in enumerate(testable):
        results[i]["q_validation"] = float(qv[qi])
        results[i]["status"] = "validated" if qv[qi] <= cfg.alpha_validation else "failed_validation"

    summary = {
        "combinations_evaluated": n_combos,
        "hypotheses_tested": n_tests,
        "discovery_significant": len(disc),
        "validated": sum(r_["status"] == "validated" for r_ in results),
    }
    if not detail:
        return summary, None
    # "Luck meter": how many very small p-values appeared vs. how many pure chance gives.
    sup_p = np.concatenate([p_bull[supported], p_bear[supported]])
    summary["luck"] = {
        "threshold": 1e-3,
        "observed": int((sup_p < 1e-3).sum()),
        "expected_by_chance": round(n_tests * 1e-3, 1),
    }
    # The best-looking combinations, whatever their fate, so people can see what
    # "looked like a pattern" and what happened to it on newer data.
    best = []
    for direction, pv in (("bullish", p_bull), ("bearish", p_bear)):
        for idx in np.argsort(pv)[:20]:
            if supported[idx]:
                best.append({"combo": cands.combos[int(idx)], "direction": direction, "p_discovery": float(pv[idx])})
    summary["_best"] = best
    return summary, results


def _score(X: np.ndarray, rules: list[dict]) -> np.ndarray:
    s = np.zeros(len(X))
    for rule in rules:
        s += _combo_vec(X, tuple(rule["_cols"])) * rule["weight"]
    return s


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def mine(features: pd.DataFrame, fwd: pd.Series, lab: pd.Series, cfg: MinerConfig | None = None) -> dict:
    """Run discovery -> validation -> untouched test, plus the null control."""
    cfg = cfg or MinerConfig()
    idx = features.index.intersection(lab.index)
    F = features.loc[idx]
    # Collapse identical columns (e.g. "Ketu in Libra" == "Rahu in Aries").
    aliases: dict[str, list[str]] = {}
    seen: dict[bytes, str] = {}
    keep_names = []
    for c in F.columns:
        key = F[c].to_numpy().tobytes()
        if key in seen:
            aliases.setdefault(seen[key], []).append(c)
        else:
            seen[key] = c
            keep_names.append(c)
    F = F[keep_names]
    # Drop columns that are always on/off over the whole history.
    var = F.sum(axis=0)
    F = F.loc[:, (var > 0) & (var < len(F))]
    names = list(F.columns)
    X = F.to_numpy(dtype=np.uint8)
    y = lab.loc[idx].to_numpy()
    r = fwd.loc[idx].to_numpy(dtype=float)
    up = (y == 1).astype(np.float32)
    down = (y == -1).astype(np.float32)
    n = len(X)
    d_end = int(n * cfg.split[0])
    v_end = int(n * (cfg.split[0] + cfg.split[1]))
    if d_end < cfg.min_support * 2 or v_end - d_end < cfg.min_support_validation or n - v_end < 10:
        raise ValueError(f"not enough bars ({n}) for the chosen split and support settings")

    summary, results = _discover_validate(X, up, down, r, d_end, v_end, cfg, detail=True)

    # Rule weights use discovery + validation only (never the test slice).
    up_dv, dn_dv = up[:v_end], down[:v_end]
    base = {"bullish": up_dv.mean(), "bearish": dn_dv.mean()}
    spans = {
        "discovery": (0, d_end), "validation": (d_end, v_end), "test": (v_end, n),
    }
    rules = []
    for res in results:
        combo = res["combo"]
        v_full = _combo_vec(X, combo)
        hit = up if res["direction"] == "bullish" else down
        k = hit[:v_end][v_full[:v_end]].sum()
        m = v_full[:v_end].sum()
        rate = (k + 1) / (m + 2)
        w = float(_logit(rate) - _logit(base[res["direction"]]))
        if res["direction"] == "bearish":
            w = -w
        stats = {name: _period_stats(v_full[a:b], up[a:b], down[a:b], r[a:b]) for name, (a, b) in spans.items()}
        rules.append({
            "conditions": [names[c] for c in combo],
            "aliases": {names[c]: aliases[names[c]] for c in combo if names[c] in aliases},
            "direction": res["direction"],
            "status": res["status"],
            "p_discovery": res["p_discovery"],
            "q_discovery": res["q_discovery"],
            "p_validation": res["p_validation"],
            "q_validation": res["q_validation"],
            "weight": w,
            "stats": stats,
            "_cols": list(combo),
        })
    rules.sort(key=lambda r_: (r_["status"] != "validated", len(r_["_cols"]),
                               r_["q_validation"] if r_["q_validation"] is not None else 2, r_["p_discovery"]))
    _mark_redundant(X[:v_end], rules)
    rules.sort(key=lambda r_: (r_["status"] != "validated",
                               r_["q_validation"] if r_["q_validation"] is not None else 2, r_["p_discovery"]))

    validated = [r_ for r_ in rules if r_["status"] == "validated"]
    summary["validated_non_redundant"] = len(validated)
    test =_evaluate_test(X[v_end:], y[v_end:], r[v_end:], validated)

    # Null control: shift labels in time so any planet<->market link is broken.
    rng = np.random.default_rng(cfg.seed)
    null_counts = []
    for _ in range(cfg.null_runs):
        shift = int(rng.integers(int(0.15 * n), int(0.85 * n)))
        s_sum, _ = _discover_validate(X, np.roll(up, shift), np.roll(down, shift), np.roll(r, shift),
                                      d_end, v_end, cfg, detail=False)
        null_counts.append(s_sum["validated"])

    for r_ in rules:
        r_.pop("_cols")
    ts = F.index
    best = []
    for b in summary.pop("_best"):
        v = _combo_vec(X, b["combo"])
        hit = up if b["direction"] == "bullish" else down
        def rate(a, e):
            m = int(v[a:e].sum())
            return {"days": m, "rate": float(hit[a:e][v[a:e]].mean()) if m else None,
                    "normal_rate": float(hit[a:e].mean())}
        best.append({"conditions": [names[c] for c in b["combo"]], "direction": b["direction"],
                     "p_discovery": b["p_discovery"], "past": rate(0, d_end), "later": rate(d_end, n)})
    return {
        "config": asdict(cfg),
        "bars": n,
        "conditions_considered": len(names),
        "periods": {
            name: {"start": str(ts[a]), "end": str(ts[b - 1]), "bars": b - a,
                   "baseline_up_rate": float(up[a:b].mean()), "baseline_down_rate": float(down[a:b].mean())}
            for name, (a, b) in spans.items()
        },
        "search": summary,
        "null_control": {
            "runs": cfg.null_runs,
            "validated_rules_per_run": null_counts,
            "mean": float(np.mean(null_counts)) if null_counts else None,
        },
        "test": test,
        "verdict": _verdict(summary["validated"], null_counts, test),
        "verdict_level": _verdict_level(summary["validated"], null_counts, test),
        "rules": rules,
        "best_looking": best,
    }


def _mark_redundant(X: np.ndarray, rules: list[dict], jaccard: float = 0.8) -> None:
    """A validated rule that fires on nearly the same bars as a simpler/stronger kept
    rule with the same direction adds nothing but double counting: mark it redundant."""
    kept: list[tuple[str, np.ndarray]] = []
    for rule in rules:
        if rule["status"] != "validated":
            continue
        v = _combo_vec(X, tuple(rule["_cols"]))
        for direction, kv in kept:
            if direction != rule["direction"]:
                continue
            inter = np.count_nonzero(v & kv)
            union = np.count_nonzero(v | kv)
            if union and inter / union >= jaccard:
                rule["status"] = "redundant"
                break
        else:
            kept.append((rule["direction"], v))


def _evaluate_test(X, y, r, validated: list[dict]) -> dict:
    if not validated:
        return {"signals": 0, "note": "no validated rules, nothing to test"}
    rules = [dict(rr, _cols=rr["_cols"]) for rr in validated]
    s = _score(X, rules)
    sig = np.sign(s)
    active = sig != 0
    n_sig = int(active.sum())
    if n_sig == 0:
        return {"signals": 0, "bars": len(X), "note": "validated rules never fired in the test period"}
    correct = int(((sig == 1) & (y == 1)).sum() + ((sig == -1) & (y == -1)).sum())
    # Chance hit rate if signals were unrelated to outcomes.
    p_up, p_dn = (y == 1).mean(), (y == -1).mean()
    share_long = (sig[active] == 1).mean()
    chance = share_long * p_up + (1 - share_long) * p_dn
    strat = (sig * r)[active]
    return {
        "bars": int(len(X)),
        "signals": n_sig,
        "long_signals": int((sig == 1).sum()),
        "short_signals": int((sig == -1).sum()),
        "hit_rate": correct / n_sig,
        "chance_hit_rate": float(chance),
        "p_value": float(binom.sf(correct - 1, n_sig, chance)),
        "strategy_return_pct": float((np.prod(1 + strat) - 1) * 100),
        "buy_and_hold_return_pct": float((np.prod(1 + r) - 1) * 100),
    }


def _verdict_level(n_valid: int, null_counts: list[int], test: dict) -> str:
    """none | luck | unconfirmed | promising  (drives the traffic light in the UI)."""
    null_mean = float(np.mean(null_counts)) if null_counts else 0.0
    if n_valid == 0:
        return "none"
    if n_valid <= max(null_counts or [0]) or n_valid < 2 * null_mean:
        return "luck"
    if test.get("signals", 0) and test.get("p_value", 1) < 0.05 and test["hit_rate"] > test["chance_hit_rate"]:
        return "promising"
    return "unconfirmed"


def _verdict(n_valid: int, null_counts: list[int], test: dict) -> str:
    null_mean = float(np.mean(null_counts)) if null_counts else 0.0
    if n_valid == 0:
        return ("No planetary combination survived validation. On this data the planets show no "
                "reliable link to Nifty moves at the chosen bar size and threshold.")
    if n_valid <= max(null_counts or [0]) or n_valid < 2 * null_mean:
        return (f"{n_valid} rules validated, but time-shifted (meaningless) data produced a similar "
                f"number ({null_mean:.1f} on average). Treat these rules as chance findings.")
    if test.get("signals", 0) and test.get("p_value", 1) < 0.05 and test["hit_rate"] > test["chance_hit_rate"]:
        return (f"{n_valid} rules validated (null control: {null_mean:.1f}) and their signals beat chance on "
                f"the untouched test period (p={test['p_value']:.3g}). Still verify with fresh data before trading.")
    return (f"{n_valid} rules validated (null control: {null_mean:.1f}), but their combined signals did not "
            f"beat chance on the untouched test period. Do not trade on them.")
