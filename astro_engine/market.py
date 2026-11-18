"""Minute price handling: cleaning, resampling to bars and up/down labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_TZ = "Asia/Kolkata"
SESSION_OPEN = "09:15"
SESSION_CLOSE = "15:30"


def normalize_minutes(df: pd.DataFrame, tz: str = MARKET_TZ) -> pd.Series:
    """Accept a frame with ``timestamp`` and ``close`` columns -> sorted close Series.

    Naive timestamps are assumed to be in market time (IST).
    """
    cols = {c.lower().strip(): c for c in df.columns}
    if "timestamp" not in cols:
        for alt in ("datetime", "date", "time", "ts"):
            if alt in cols:
                cols["timestamp"] = cols[alt]
                break
    if "timestamp" not in cols or "close" not in cols:
        raise ValueError("data needs 'timestamp' (or datetime/date) and 'close' columns")
    ts = pd.to_datetime(df[cols["timestamp"]], utc=False, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(tz)
    else:
        ts = ts.dt.tz_convert(tz)
    s = pd.Series(pd.to_numeric(df[cols["close"]], errors="coerce").to_numpy(), index=pd.DatetimeIndex(ts))
    s = s[s.index.notna()].dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[s > 0]
    s.name = "close"
    s.index.name = "timestamp"
    return s


def to_bars(close: pd.Series, bar: str) -> pd.Series:
    """Resample minute closes to bars.

    ``bar``: ``"1D"`` (daily close), or an intraday pandas offset like ``"15min"``/``"60min"``.
    Intraday bars are labelled by their *start* time and never span sessions.
    """
    if bar.upper() in ("1D", "D", "DAILY"):
        daily = close.groupby(close.index.normalize()).last()
        # Stamp each daily bar at the session close.
        h, m = map(int, SESSION_CLOSE.split(":"))
        daily.index = daily.index + pd.Timedelta(hours=h, minutes=m)
        return daily
    session = close.between_time(SESSION_OPEN, SESSION_CLOSE)
    return session.resample(bar, label="left", closed="left", origin="start_day",
                            offset=pd.Timedelta(hours=9, minutes=15)).last().dropna()


def forward_returns(bars: pd.Series, bar: str, horizon: int = 1) -> pd.Series:
    """Return from bar t's close to bar t+horizon's close.

    For intraday bars, returns that would cross into the next session are dropped,
    so overnight gaps are not mixed with intraday moves.
    """
    fwd = bars.shift(-horizon) / bars - 1.0
    if bar.upper() not in ("1D", "D", "DAILY"):
        same_day = bars.index.normalize() == pd.Series(bars.index, index=bars.index).shift(-horizon).dt.normalize()
        fwd = fwd.where(same_day.to_numpy())
    return fwd.dropna()


def labels(fwd: pd.Series, threshold: float) -> pd.Series:
    """+1 rise (> threshold), -1 fall (< -threshold), 0 flat."""
    lab = np.where(fwd > threshold, 1, np.where(fwd < -threshold, -1, 0))
    return pd.Series(lab, index=fwd.index, name="label")


def future_bar_times(start: pd.Timestamp, end: pd.Timestamp, bar: str) -> pd.DatetimeIndex:
    """NSE trading-bar timestamps (weekdays; exchange holidays are not excluded)."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize(MARKET_TZ) if start.tzinfo is None else start.tz_convert(MARKET_TZ)
    end = end.tz_localize(MARKET_TZ) if end.tzinfo is None else end.tz_convert(MARKET_TZ)
    days = pd.bdate_range(start.normalize(), end.normalize(), tz=MARKET_TZ)
    if bar.upper() in ("1D", "D", "DAILY"):
        h, m = map(int, SESSION_CLOSE.split(":"))
        return pd.DatetimeIndex(days + pd.Timedelta(hours=h, minutes=m))
    step = pd.Timedelta(bar)
    out = []
    o_h, o_m = map(int, SESSION_OPEN.split(":"))
    c_h, c_m = map(int, SESSION_CLOSE.split(":"))
    for d in days:
        t = d + pd.Timedelta(hours=o_h, minutes=o_m)
        close_t = d + pd.Timedelta(hours=c_h, minutes=c_m)
        while t < close_t:
            out.append(t)
            t += step
    idx = pd.DatetimeIndex(out)
    return idx[(idx >= start) & (idx <= end)] if len(idx) else idx
