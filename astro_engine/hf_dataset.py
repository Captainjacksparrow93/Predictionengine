"""Load any NSE index or stock from the public Hugging Face dataset
``xxparthparekhxx/indian-stock-market-minute-data`` (MIT licence).

The dataset has daily candles from 2000 and 1-minute candles from 2022 for
3,182 symbols, sorted by symbol. Only the row groups that can contain the
requested symbol are downloaded (via parquet statistics), so loading one
symbol transfers tens of MB, not the full 14 GB.
"""
from __future__ import annotations

import io
import time

import pandas as pd

from .market import MARKET_TZ

REPO = "xxparthparekhxx/indian-stock-market-minute-data"
BASE = f"datasets/{REPO}/"
SOURCE_NAME = "Hugging Face: indian-stock-market-minute-data (NSE via Upstox, MIT)"
MINUTE_SHARDS = 10

# Shown first in the picker, in this order.
POPULAR = ["NIFTY", "BANKNIFTY", "NIFTY 500", "NIFTY 100", "FINNIFTY", "NIFTY IT", "NIFTY AUTO",
           "NIFTY PHARMA", "NIFTY FMCG", "NIFTY METAL", "NIFTY REALTY", "NIFTY MIDCAP 100", "INDIA VIX"]

_catalog_cache: dict = {}


def catalog(max_age_s: int = 6 * 3600) -> list[dict]:
    """Every symbol in the dataset with its name, type and date coverage."""
    if _catalog_cache and time.time() - _catalog_cache["t"] < max_age_s:
        return _catalog_cache["rows"]
    import requests

    r = requests.get(f"https://huggingface.co/datasets/{REPO}/resolve/main/symbols.csv", timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df[df["listed"].astype(str).str.lower() == "yes"]
    rank = {s: i for i, s in enumerate(POPULAR)}
    df["popular"] = df["symbol"].map(rank)
    df["type_rank"] = (df["type"] != "INDEX").astype(int)
    df = df.sort_values(["popular", "type_rank", "symbol"], na_position="last")
    rows = [
        {
            "symbol": s.symbol, "name": s.name, "type": s.type, "popular": pd.notna(s.popular),
            "daily_from": s.day_first, "minute_from": s.minute_first if pd.notna(s.minute_first) else None,
            "to": s.day_last,
        }
        for s in df.itertuples()
    ]
    _catalog_cache.update(t=time.time(), rows=rows)
    return rows


def _read_symbol(fs, path: str, symbol: str):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    f = pq.ParquetFile(fs.open(BASE + path, block_size=8 << 20))
    md = f.metadata
    col = f.schema_arrow.get_field_index("symbol")
    groups = []
    for g in range(md.num_row_groups):
        st = md.row_group(g).column(col).statistics
        if st is None or not st.has_min_max or st.min <= symbol <= st.max:
            groups.append(g)
    if not groups:
        return None
    t = f.read_row_groups(groups, columns=["symbol", "timestamp", "close"])
    t = t.filter(pc.equal(t["symbol"], symbol))
    return t.to_pandas() if t.num_rows else None


def load_symbol(symbol: str, progress=None) -> pd.Series:
    """Daily closes before minute data begins, then every 1-minute close.

    Daily closes are stamped at the 15:30 session close, so the engine's daily
    bars use the full history and intraday bars use only the minute data.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    say = progress or (lambda msg: None)
    say("Downloading daily history")
    day = _read_symbol(fs, "day/train-00000.parquet", symbol)
    minute_parts = []
    for i in range(MINUTE_SHARDS):
        say(f"Scanning minute data ({i + 1}/{MINUTE_SHARDS})")
        part = _read_symbol(fs, f"minute/train-{i:05d}.parquet", symbol)
        if part is not None:
            minute_parts.append(part)
    if day is None and not minute_parts:
        raise ValueError(f"symbol {symbol!r} not found in the dataset")

    frames = []
    minute = pd.concat(minute_parts) if minute_parts else None
    if minute is not None:
        minute["timestamp"] = pd.to_datetime(minute["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    if day is not None:
        day["timestamp"] = pd.to_datetime(day["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
        if minute is not None:
            day = day[day["timestamp"] < minute["timestamp"].min().normalize()]
        day["timestamp"] = day["timestamp"].dt.normalize() + pd.Timedelta(hours=15, minutes=30)
        frames.append(day)
    if minute is not None:
        frames.append(minute)
    df = pd.concat(frames).sort_values("timestamp")
    s = pd.Series(df["close"].astype(float).to_numpy(), index=pd.DatetimeIndex(df["timestamp"]), name="close")
    s = s[~s.index.duplicated(keep="last")]
    s.index.name = "timestamp"
    return s[s > 0]
