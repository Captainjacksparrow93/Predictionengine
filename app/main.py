"""FastAPI service: configure the data API, load history, mine rules, forecast."""
from __future__ import annotations

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

from astro_engine import storage
from astro_engine.data_sources import PRESETS, ApiSource, fetch_api, missing_env, read_csv_bytes
from astro_engine.ephemeris import BODIES, NAKSHATRAS, SIGNS, compute_positions, nakshatra_index, sign_index
from astro_engine import hf_dataset
from astro_engine.forecast import astro_events, forecast
from astro_engine.market import MARKET_TZ
from astro_engine.pipeline import DEFAULT_SETTINGS, analyze

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Nifty Astro Prediction Engine", version="0.1.0")


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
            raise HTTPException(409, f"a {_job.get('kind')} job is already running")
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


# ----------------------------------------------------------------------------- data source
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


@app.get("/api/data/summary", dependencies=auth)
def data_summary():
    s = storage.load_prices()
    if s is None or s.empty:
        return {"minutes": 0}
    daily = s.groupby(s.index.normalize()).last()
    intraday = s[(s.index.hour != 15) | (s.index.minute != 30)]
    return {
        "dataset": storage.load_json("dataset.json"),
        "intraday_from": str(intraday.index[0]) if len(intraday) > 1000 else None,
        "minutes": int(len(s)),
        "days": int(len(daily)),
        "first": str(s.index[0]),
        "last": str(s.index[-1]),
        "daily_close": [{"date": str(d.date()), "close": float(c)} for d, c in daily.items()],
    }


# ----------------------------------------------------------------------------- analysis
@app.get("/api/settings/default", dependencies=auth)
def default_settings():
    return DEFAULT_SETTINGS


@app.post("/api/analyze", dependencies=auth)
def run_analysis(settings: dict):
    s = storage.load_prices()
    if s is None or s.empty:
        raise HTTPException(400, "no price data loaded")

    ds = storage.load_json("dataset.json") or {}
    label = settings.pop("label", None)

    def work(progress):
        progress("Calculating planet positions and testing every combination")
        model = analyze(s, settings)
        model["dataset"] = ds
        model["label"] = label
        model["created"] = pd.Timestamp.now(tz=MARKET_TZ).isoformat()
        storage.save_json("model.json", model)
        r = model["result"]
        hist = storage.load_json("history.json") or []
        hist.insert(0, {
            "created": model["created"], "label": label, "market": ds.get("name"),
            "zodiac": model["settings"]["zodiac"], "bar": model["settings"]["bar"],
            "tested": r["search"]["hypotheses_tested"], "validated": r["search"]["validated_non_redundant"],
            "chance": r["null_control"]["mean"], "level": r["verdict_level"], "verdict": r["verdict"],
        })
        storage.save_json("history.json", hist[:50])
        return {"search": r["search"], "verdict": r["verdict"], "runtime_seconds": model["runtime_seconds"]}

    return _run_job("analyze", work)


@app.get("/api/history", dependencies=auth)
def history():
    return storage.load_json("history.json") or []


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
        storage.save_json("model.json", None)
        return {"symbol": req.symbol, "stored_minutes": len(s)}

    return _run_job("load", work)


def _set_dataset(symbol: str, name: str, source: str) -> None:
    storage.save_json("dataset.json", {"symbol": symbol, "name": name, "source": source,
                                       "loaded": pd.Timestamp.now(tz=MARKET_TZ).isoformat()})


@app.get("/api/model", dependencies=auth)
def get_model(rules: str = "all", limit: int = 300):
    m = storage.load_json("model.json")
    if not m:
        raise HTTPException(404, "no analysis has been run yet")
    rr = m["result"]["rules"]
    if rules != "all":
        rr = [r for r in rr if r["status"] == rules]
    m["result"]["rules_total"] = len(m["result"]["rules"])
    m["result"]["rules"] = rr[:limit]
    return m


@app.get("/api/forecast", dependencies=auth)
def get_forecast(start: str, end: str, include: str = "validated"):
    m = storage.load_json("model.json")
    if not m:
        raise HTTPException(404, "run an analysis first")
    if pd.Timestamp(end) - pd.Timestamp(start) > pd.Timedelta(days=800):
        raise HTTPException(400, "forecast range is limited to ~2 years")
    return forecast(m, start, end, include)


@app.get("/api/events", dependencies=auth)
def get_events(start: str, end: str, zodiac: str = "sidereal", moon: bool = False):
    if pd.Timestamp(end) - pd.Timestamp(start) > pd.Timedelta(days=800):
        raise HTTPException(400, "range is limited to ~2 years")
    return astro_events(start, end, zodiac, include_moon=moon)


@app.get("/api/positions", dependencies=auth)
def positions(at: str | None = None, zodiac: str = "sidereal"):
    t = pd.Timestamp(at) if at else pd.Timestamp.now(tz=MARKET_TZ)
    t = t.tz_localize(MARKET_TZ) if t.tzinfo is None else t
    p = compute_positions(pd.DatetimeIndex([t]), zodiac)
    out = []
    for b in BODIES:
        lon = float(p.longitude[b].iloc[0])
        out.append({
            "body": b,
            "longitude": round(lon, 4),
            "sign": SIGNS[int(sign_index(lon))],
            "degree_in_sign": round(lon % 30, 2),
            "nakshatra": NAKSHATRAS[int(nakshatra_index(lon))],
            "retrograde": bool(p.speed[b].iloc[0] < 0) and b not in ("Rahu", "Ketu"),
            "speed_deg_per_day": round(float(p.speed[b].iloc[0]), 4),
        })
    return {"time": t.isoformat(), "zodiac": zodiac, "bodies": out}


@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
