"""FastAPI service: market data, settings, pattern search, library, prediction, reality check."""
from __future__ import annotations

import copy
import os
import secrets
import threading
import traceback
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from astro_engine import hf_dataset, patterns, storage
from astro_engine.data_sources import PRESETS, ApiSource, fetch_api, missing_env, read_csv_bytes
from astro_engine.ephemeris import AYANAMSAS, EPHEMERIDES, NAKSHATRAS, SIGNS, compute_positions, nakshatra_index, sign_index
from astro_engine.features import FEATURE_GROUPS, FeatureConfig, build_features
from astro_engine.forecast import astro_events
from astro_engine.market import MARKET_TZ
from astro_engine.miner import MinerConfig
from astro_engine.pipeline import DEFAULT_ASTRO, STRICTNESS, analyze, astro_kwargs

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Planet Market Finder", version="0.3.0")


def require_token(request: Request) -> None:
    """If APP_TOKEN is set, every /api call needs header ``X-App-Token``."""
    expected = os.environ.get("APP_TOKEN")
    if expected and not secrets.compare_digest(request.headers.get("x-app-token", ""), expected):
        raise HTTPException(401, "missing or invalid X-App-Token")


auth = [Depends(require_token)]

# ----------------------------------------------------------------------------- jobs
_job_lock = threading.Lock()
_job: dict = {"state": "idle"}


def _run_job(kind: str, fn) -> dict:
    with _job_lock:
        if _job.get("state") == "running":
            raise HTTPException(409, f"a {_job.get('kind')} job is already running, please wait")
        _job.clear()
        _job.update(state="running", kind=kind, started=pd.Timestamp.now(tz=MARKET_TZ).isoformat())

    def target():
        try:
            out = fn(lambda msg: _job.update(progress=msg))
            _job.update(state="done", result=out)
        except Exception as e:  # noqa: BLE001 - surfaced to the UI
            _job.update(state="error", error=str(e), trace=traceback.format_exc(limit=3))

    threading.Thread(target=target, daemon=True).start()
    return {"state": "running", "kind": kind}


@app.get("/api/job", dependencies=auth)
def job_status():
    return {k: v for k, v in _job.items() if k != "trace"}


# ----------------------------------------------------------------------------- settings
DEFAULT_USER_SETTINGS = {
    "astro": dict(DEFAULT_ASTRO),
    "strictness": "strict",
    "features": {k: v for k, v in vars(FeatureConfig()).items() if k != "natal_longitudes"},
    "miner": {k: v for k, v in vars(MinerConfig()).items() if k not in ("seed", "max_candidates")},
    "predict": {"include": "validated", "min_score": 0.0},
}
DEFAULT_USER_SETTINGS["miner"].update(null_runs=5, triple_beam=500)
DEFAULT_USER_SETTINGS["miner"]["split"] = list(DEFAULT_USER_SETTINGS["miner"]["split"])


def user_settings() -> dict:
    saved = storage.load_json("settings.json") or {}
    s = copy.deepcopy(DEFAULT_USER_SETTINGS)
    for k, v in saved.items():
        if isinstance(v, dict) and isinstance(s.get(k), dict):
            s[k].update(v)
        else:
            s[k] = v
    return s


@app.get("/api/settings", dependencies=auth)
def get_settings():
    return {
        "settings": user_settings(),
        "defaults": DEFAULT_USER_SETTINGS,
        "options": {
            "ephemerides": EPHEMERIDES,
            "ayanamsas": {k: v[0] for k, v in AYANAMSAS.items()},
            "feature_groups": FEATURE_GROUPS,
            "strictness": STRICTNESS,
        },
    }


@app.put("/api/settings", dependencies=auth)
def put_settings(s: dict):
    clean = {k: s[k] for k in DEFAULT_USER_SETTINGS if k in s}
    storage.save_json("settings.json", clean)
    return get_settings()


@app.delete("/api/settings", dependencies=auth)
def reset_settings():
    storage.save_json("settings.json", {})
    return get_settings()


# ----------------------------------------------------------------------------- data
@app.get("/api/source/presets", dependencies=auth)
def presets():
    return PRESETS


@app.get("/api/source", dependencies=auth)
def get_source():
    cfg = storage.load_json("source.json")
    if cfg is None:
        return {"source": None}
    return {"source": cfg, "missing_env": missing_env(ApiSource.from_dict(cfg))}


@app.put("/api/source", dependencies=auth)
def put_source(cfg: dict):
    try:
        src = ApiSource.from_dict(cfg)
    except TypeError as e:
        raise HTTPException(400, str(e)) from e
    storage.save_json("source.json", src.to_dict())
    return {"source": src.to_dict(), "missing_env": missing_env(src)}


class FetchReq(BaseModel):
    start: str
    end: str


def _set_dataset(symbol: str, name: str, source: str) -> None:
    storage.save_json("dataset.json", {"symbol": symbol, "name": name, "source": source,
                                       "loaded": pd.Timestamp.now(tz=MARKET_TZ).isoformat()})


@app.post("/api/data/fetch", dependencies=auth)
def fetch(req: FetchReq):
    cfg = storage.load_json("source.json")
    if cfg is None:
        raise HTTPException(400, "configure the data API first")
    src = ApiSource.from_dict(cfg)
    if missing := missing_env(src):
        raise HTTPException(400, f"set these environment variables on the server: {', '.join(missing)}")

    def work(progress):
        progress("Downloading from your API")
        s = fetch_api(src, req.start, req.end)
        full = storage.save_prices(s)
        _set_dataset(src.name or "custom API", src.name or "Custom API", "your data API")
        return {"fetched_minutes": len(s), "stored_minutes": len(full)}

    return _run_job("fetch", work)


@app.post("/api/data/upload", dependencies=auth)
async def upload(file: UploadFile = File(...), replace: bool = False):
    raw = await file.read()
    try:
        s = read_csv_bytes(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read CSV: {e}") from e
    full = storage.save_prices(s, merge=not replace)
    _set_dataset(file.filename or "upload", file.filename or "Uploaded file", "CSV upload")
    return {"uploaded_minutes": len(s), "stored_minutes": len(full)}


@app.delete("/api/data", dependencies=auth)
def clear_data():
    storage.clear_prices()
    return {"ok": True}


@app.get("/api/catalog", dependencies=auth)
def get_catalog():
    try:
        return hf_dataset.catalog()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"could not reach the market-data library: {e}") from e


class HfReq(BaseModel):
    symbol: str


@app.post("/api/data/load_hf", dependencies=auth)
def load_hf(req: HfReq):
    entry = next((c for c in hf_dataset.catalog() if c["symbol"] == req.symbol), None)
    if entry is None:
        raise HTTPException(404, f"unknown symbol {req.symbol}")

    def work(progress):
        s = hf_dataset.load_symbol(req.symbol, progress)
        progress("Saving")
        storage.save_prices(s, merge=False)
        _set_dataset(req.symbol, entry["name"], hf_dataset.SOURCE_NAME)
        return {"symbol": req.symbol, "stored_minutes": len(s)}

    return _run_job("load", work)


@app.get("/api/data/summary", dependencies=auth)
def data_summary():
    s = storage.load_prices()
    if s is None or s.empty:
        return {"minutes": 0, "dataset": storage.load_json("dataset.json")}
    daily = s.groupby(s.index.normalize()).last()
    intraday = s[(s.index.hour != 15) | (s.index.minute != 30)]
    return {
        "dataset": storage.load_json("dataset.json"),
        "intraday_from": str(intraday.index[0]) if len(intraday) > 1000 else None,
        "minutes": int(len(s)),
        "days": int(len(daily)),
        "first": str(s.index[0]),
        "last": str(s.index[-1]),
        "daily_close": [{"date": str(d.date()), "close": round(float(c), 2)} for d, c in daily.items()],
    }


# ----------------------------------------------------------------------------- pattern search
QUESTION_NAMES = {"day": "Tomorrow", "week": "Next week", "big": "Big moves", "hour": "Next hour", "custom": "Custom"}


@app.post("/api/analyze", dependencies=auth)
def run_analysis(body: dict):
    """Body: {question: {bar, horizon, threshold_pct, label}, name?}; the rest comes from saved settings."""
    s = storage.load_prices()
    if s is None or s.empty:
        raise HTTPException(400, "no price data loaded")
    us = user_settings()
    q = body.get("question") or {}
    settings = {
        "bar": q.get("bar", "1D"), "horizon": int(q.get("horizon", 1)), "threshold_pct": float(q.get("threshold_pct", 0)),
        "astro": us["astro"], "strictness": us["strictness"], "features": us["features"],
        "miner": {k: v for k, v in us["miner"].items()},
    }
    if us["strictness"] in STRICTNESS:
        for k in STRICTNESS[us["strictness"]]:
            settings["miner"].pop(k, None)
    ds = storage.load_json("dataset.json") or {}
    label = q.get("label", "custom")

    def work(progress):
        progress("Calculating planet positions and testing every combination")
        model = analyze(s, settings)
        model["dataset"] = ds
        model["label"] = label
        model["created"] = pd.Timestamp.now(tz=MARKET_TZ).isoformat()
        eph = "Surya Siddhanta" if settings["astro"]["ephemeris"] == "surya_siddhanta" else "Modern"
        name = body.get("name") or (f"{ds.get('name', 'Market')} · {QUESTION_NAMES.get(label, label)} · "
                                    f"{eph}/{settings['astro']['ayanamsa']} · {settings['strictness']}")
        saved = patterns.save(model, name)
        storage.save_json("model.json", saved)
        r = model["result"]
        return {"pattern_id": saved["id"], "search": r["search"], "verdict": r["verdict"],
                "runtime_seconds": model["runtime_seconds"]}

    return _run_job("analyze", work)


# ----------------------------------------------------------------------------- library
def _get_pattern(pid: str) -> dict:
    try:
        return patterns.load(pid)
    except KeyError as e:
        raise HTTPException(404, "pattern set not found") from e


@app.get("/api/patterns", dependencies=auth)
def list_patterns():
    return {"active": patterns.active_id(), "items": patterns.list_all()}


@app.get("/api/patterns/{pid}", dependencies=auth)
def get_pattern(pid: str, status: str = "all", limit: int = 200):
    m = _get_pattern(pid)
    rr = m["result"]["rules"]
    if status != "all":
        rr = [r for r in rr if r["status"] == status]
    m["result"]["rules_total"] = len(m["result"]["rules"])
    m["result"]["rules"] = rr[:limit]
    m["active"] = m["id"] == patterns.active_id()
    return m


class PatternPatch(BaseModel):
    name: str | None = None
    notes: str | None = None
    active: bool | None = None


@app.patch("/api/patterns/{pid}", dependencies=auth)
def patch_pattern(pid: str, p: PatternPatch):
    _get_pattern(pid)
    if p.name is not None or p.notes is not None:
        patterns.update(pid, **{k: v for k, v in (("name", p.name), ("notes", p.notes)) if v is not None})
    if p.active:
        patterns.set_active(pid)
    return patterns.summary(patterns.load(pid))


@app.delete("/api/patterns/{pid}", dependencies=auth)
def delete_pattern(pid: str):
    _get_pattern(pid)
    patterns.delete(pid)
    return {"ok": True}


def _predict_opts(include: str | None, min_score: float | None) -> tuple[str, float]:
    ps = user_settings()["predict"]
    return include or ps["include"], float(ps["min_score"] if min_score is None else min_score)


def _limit_range(start: str, end: str, days: int = 800) -> None:
    if pd.Timestamp(end) - pd.Timestamp(start) > pd.Timedelta(days=days):
        raise HTTPException(400, f"please choose a range of at most {days} days")


@app.get("/api/patterns/{pid}/predict", dependencies=auth)
def predict(pid: str, start: str, end: str, include: str | None = None, min_score: float | None = None):
    _limit_range(start, end)
    inc, ms = _predict_opts(include, min_score)
    return patterns.predict(_get_pattern(pid), start, end, inc, ms)


class CheckReq(BaseModel):
    start: str | None = None
    end: str | None = None
    include: str | None = None
    min_score: float | None = None


@app.post("/api/patterns/{pid}/check", dependencies=auth)
def check(pid: str, req: CheckReq):
    s = storage.load_prices()
    if s is None or s.empty:
        raise HTTPException(400, "no price data loaded")
    inc, ms = _predict_opts(req.include, req.min_score)
    m = _get_pattern(pid)
    start = req.start
    if not start:
        # Default: only the years the search never used (after the checking years).
        start = str((pd.Timestamp(m["result"]["periods"]["validation"]["end"]) + pd.Timedelta(days=1)).date())
    try:
        return patterns.reality_check(m, s, start, req.end, inc, ms)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


class JournalReq(BaseModel):
    pattern_id: str
    start: str
    end: str
    include: str | None = None
    min_score: float | None = None


@app.post("/api/journal", dependencies=auth)
def journal_add(req: JournalReq):
    _limit_range(req.start, req.end)
    m = _get_pattern(req.pattern_id)
    inc, ms = _predict_opts(req.include, req.min_score)
    return patterns.journal_add(m, patterns.predict(m, req.start, req.end, inc, ms))


@app.get("/api/journal", dependencies=auth)
def journal():
    return patterns.journal_scored(storage.load_prices())


@app.delete("/api/journal/{jid}", dependencies=auth)
def journal_delete(jid: str):
    j = [e for e in (storage.load_json("journal.json") or []) if e["id"] != jid]
    storage.save_json("journal.json", j)
    return {"ok": True}


@app.get("/api/model", dependencies=auth)
def latest_model():
    m = storage.load_json("model.json")
    if not m:
        raise HTTPException(404, "no analysis has been run yet")
    return get_pattern(m["id"]) if m.get("id") else m


# ----------------------------------------------------------------------------- sky
@app.get("/api/events", dependencies=auth)
def get_events(start: str, end: str, moon: bool = False):
    _limit_range(start, end)
    return astro_events(start, end, include_moon=moon, astro=user_settings()["astro"])


TITHI_NAMES = ["Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami", "Ashtami",
               "Navami", "Dashami", "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi"]


@app.get("/api/positions", dependencies=auth)
def positions(at: str | None = None):
    t = pd.Timestamp(at) if at else pd.Timestamp.now(tz=MARKET_TZ)
    t = t.tz_localize(MARKET_TZ) if t.tzinfo is None else t
    us = user_settings()
    p = compute_positions(pd.DatetimeIndex([t]), **astro_kwargs({"astro": us["astro"]}))
    feats = build_features(p, FeatureConfig(**{**{k: True for k in FEATURE_GROUPS}, "latitude": us["features"]["latitude"],
                                                 "longitude": us["features"]["longitude"]}))
    on = [c for c in feats.columns if feats[c].iloc[0]]
    pick = lambda prefix: next((c[len(prefix):] for c in on if c.startswith(prefix)), None)  # noqa: E731
    tithi = pick("Tithi ")
    num = int(tithi.split()[0]) if tithi else None
    paksha = "Shukla" if num and num <= 15 else "Krishna"
    tname = None
    if num:
        k = (num - 1) % 15
        tname = ("Purnima" if num == 15 else "Amavasya") if k == 14 else TITHI_NAMES[k]
    bodies = []
    for b in p.longitude.columns:
        lon = float(p.longitude[b].iloc[0])
        flags = [c.split(" ", 1)[1] for c in on if c.startswith(b + " ") and c.split(" ", 1)[1] in
                 ("retrograde", "combust", "exalted", "debilitated", "in own sign")]
        bodies.append({
            "body": b, "longitude": round(lon, 4), "sign": SIGNS[int(sign_index(lon))],
            "degree_in_sign": round(lon % 30, 2), "nakshatra": NAKSHATRAS[int(nakshatra_index(lon))],
            "pada": int(lon // (360 / 108)) % 4 + 1,
            "retrograde": "retrograde" in flags, "flags": flags,
            "speed_deg_per_day": round(float(p.speed[b].iloc[0]), 4),
        })
    return {
        "time": t.isoformat(), "astro": us["astro"], "bodies": bodies,
        "panchang": {"tithi": tname, "tithi_number": num, "paksha": paksha, "yoga": pick("Yoga "),
                     "karana": pick("Karana "), "vara": pick("Vara "), "lagna": pick("Lagna "),
                     "moon_nakshatra": bodies[[b["body"] for b in bodies].index("Moon")]["nakshatra"]},
    }


@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
