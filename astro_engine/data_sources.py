"""Pull historical minute candles from a configurable REST API, or from CSV.

The API connector is generic: you describe the URL, headers, query params and
where the timestamp and close live in the JSON response. ``{from}`` / ``{to}``
placeholders are filled per request window, so long histories are fetched in
chunks (most brokers cap minute history per call, e.g. Kite: 60 days).

Secrets never need to be saved in the config: any ``${ENV_NAME}`` inside a
URL, header or param is replaced with that environment variable at fetch time.
"""
from __future__ import annotations

import io
import os
import re
import time
from dataclasses import asdict, dataclass, field

import httpx
import pandas as pd

from .market import MARKET_TZ, normalize_minutes

PRESETS = {
    "zerodha_kite": {
        "name": "Zerodha Kite Connect - NIFTY 50 (token 256265)",
        "url": "https://api.kite.trade/instruments/historical/256265/minute",
        "headers": {"X-Kite-Version": "3", "Authorization": "token ${KITE_API_KEY}:${KITE_ACCESS_TOKEN}"},
        "params": {"from": "{from}", "to": "{to}"},
        "date_format": "%Y-%m-%d %H:%M:%S",
        "records_path": "data.candles",
        "timestamp_field": "0",
        "close_field": "4",
        "chunk_days": 60,
    },
    "generic_json": {
        "name": "Generic JSON API (list of {timestamp, close} objects)",
        "url": "https://example.com/api/candles?symbol=NIFTY50&interval=1m&from={from}&to={to}",
        "headers": {"Authorization": "Bearer ${MARKET_API_TOKEN}"},
        "params": {},
        "date_format": "%Y-%m-%d",
        "records_path": "data",
        "timestamp_field": "timestamp",
        "close_field": "close",
        "chunk_days": 30,
    },
}


@dataclass
class ApiSource:
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    date_format: str = "%Y-%m-%d"
    records_path: str = "data"        # dotted path to the list of candles in the JSON
    timestamp_field: str = "timestamp"  # key, or list index ("0") for array candles
    close_field: str = "close"
    timestamp_unit: str | None = None  # "s"/"ms" if timestamps are epoch numbers
    chunk_days: int = 30
    timezone: str = MARKET_TZ        # zone for naive timestamps in the response
    name: str = "custom"

    @classmethod
    def from_dict(cls, d: dict) -> "ApiSource":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand(s: str, frm: str, to: str) -> str:
    s = _ENV.sub(lambda m: os.environ.get(m.group(1), ""), s)
    return s.replace("{from}", frm).replace("{to}", to)


def _dig(obj, path: str):
    for part in [p for p in path.split(".") if p]:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def _field(rec, key: str):
    if isinstance(rec, (list, tuple)):
        return rec[int(key)]
    return rec[key]


def missing_env(src: ApiSource) -> list[str]:
    blobs = [src.url, *src.headers.values(), *map(str, src.params.values())]
    names = {m for b in blobs for m in _ENV.findall(b)}
    return sorted(n for n in names if not os.environ.get(n))


def fetch_api(src: ApiSource, start, end, client: httpx.Client | None = None,
              pause: float = 0.35) -> pd.Series:
    """Fetch [start, end] in ``chunk_days`` windows and return a minute close Series."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    own = client is None
    client = client or httpx.Client(timeout=60)
    frames = []
    try:
        cur = start
        while cur <= end:
            nxt = min(cur + pd.Timedelta(days=src.chunk_days) - pd.Timedelta(seconds=1), end)
            frm, to = cur.strftime(src.date_format), nxt.strftime(src.date_format)
            resp = client.get(
                _expand(src.url, frm, to),
                headers={k: _expand(v, frm, to) for k, v in src.headers.items()},
                params={k: _expand(str(v), frm, to) for k, v in src.params.items()},
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"API returned {resp.status_code} for {frm}..{to}: {resp.text[:300]}")
            recs = _dig(resp.json(), src.records_path)
            if recs:
                ts = [_field(r, src.timestamp_field) for r in recs]
                cl = [_field(r, src.close_field) for r in recs]
                frames.append(pd.DataFrame({"timestamp": ts, "close": cl}))
            cur = nxt + pd.Timedelta(seconds=1)
            if pause:
                time.sleep(pause)
    finally:
        if own:
            client.close()
    if not frames:
        raise RuntimeError("API returned no candles for the requested range")
    df = pd.concat(frames, ignore_index=True)
    if src.timestamp_unit:
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit=src.timestamp_unit, utc=True)
    return normalize_minutes(df, tz=src.timezone)


def read_csv_bytes(raw: bytes, tz: str = MARKET_TZ) -> pd.Series:
    return normalize_minutes(pd.read_csv(io.BytesIO(raw)), tz=tz)
