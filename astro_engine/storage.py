"""Tiny file-based persistence under ``DATA_DIR`` (default ./data)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .market import MARKET_TZ


def data_dir() -> Path:
    p = Path(os.environ.get("DATA_DIR", "data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _prices_path() -> Path:
    return data_dir() / "nifty_minute.csv.gz"


def save_prices(close: pd.Series, merge: bool = True) -> pd.Series:
    if merge:
        old = load_prices()
        if old is not None:
            close = pd.concat([old, close])
            close = close[~close.index.duplicated(keep="last")].sort_index()
    close.rename("close").to_csv(_prices_path(), index_label="timestamp", compression="gzip")
    return close


def load_prices() -> pd.Series | None:
    p = _prices_path()
    if not p.exists():
        return None
    df = pd.read_csv(p, compression="gzip")
    idx = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    return pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(idx), name="close")


def clear_prices() -> None:
    _prices_path().unlink(missing_ok=True)


def save_json(name: str, obj) -> None:
    tmp = data_dir() / f".{name}.tmp"
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(data_dir() / name)


def load_json(name: str):
    p = data_dir() / name
    return json.loads(p.read_text()) if p.exists() else None
