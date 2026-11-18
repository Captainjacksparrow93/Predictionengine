"""Planetary positions for arbitrary timestamps.

Positions are computed with PyEphem (VSOP87 / ELP2000 theories, arc-second
level accuracy for 1900-2100) on a regular time grid and interpolated with a
cubic spline, which is accurate to well under an arc-second for the Moon on a
2-hour grid and makes evaluating ~1M minute bars cheap.

Longitudes are geocentric ecliptic longitudes of date. In ``sidereal`` mode
the Lahiri (Chitrapaksha) ayanamsa is subtracted, which is the convention used
in Indian (Vedic) astrology. Rahu/Ketu are the mean lunar nodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import ephem
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

BODIES = [
    "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn",
    "Rahu", "Ketu", "Uranus", "Neptune", "Pluto",
]
# Bodies that can be retrograde (nodes are always retrograde in the mean model).
RETRO_BODIES = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu",
    "Pushya", "Ashlesha", "Magha", "PurvaPhalguni", "UttaraPhalguni", "Hasta",
    "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "PurvaAshadha",
    "UttaraAshadha", "Shravana", "Dhanishta", "Shatabhisha", "PurvaBhadrapada",
    "UttaraBhadrapada", "Revati",
]

_EPHEM_BODIES = {
    "Sun": ephem.Sun, "Moon": ephem.Moon, "Mercury": ephem.Mercury,
    "Venus": ephem.Venus, "Mars": ephem.Mars, "Jupiter": ephem.Jupiter,
    "Saturn": ephem.Saturn, "Uranus": ephem.Uranus, "Neptune": ephem.Neptune,
    "Pluto": ephem.Pluto,
}

_UNIX_EPOCH_JD = 2440587.5
_J2000_JD = 2451545.0
# PyEphem dates are Dublin Julian Days (JD - 2415020).
_DJD_OFFSET = 2415020.0

GRID_HOURS = 2.0


def to_julian(ts: pd.DatetimeIndex) -> np.ndarray:
    """UTC DatetimeIndex -> Julian Day (UT)."""
    if ts.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    days = (ts.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1)
    return np.asarray(days, dtype=np.float64) + _UNIX_EPOCH_JD


def lahiri_ayanamsa(jd: np.ndarray) -> np.ndarray:
    """Lahiri ayanamsa in degrees (linear precession model, ~1" accurate 1900-2100)."""
    years = (jd - _J2000_JD) / 365.25
    return 23.85305 + years * (50.2888 / 3600.0)


def mean_rahu(jd: np.ndarray) -> np.ndarray:
    """Tropical longitude of the mean ascending lunar node (Meeus, ch. 47)."""
    t = (jd - _J2000_JD) / 36525.0
    om = 125.0445479 - 1934.1362891 * t + 0.0020754 * t**2 + t**3 / 467441.0 - t**4 / 60616000.0
    return np.mod(om, 360.0)


def _raw_tropical(jd: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    objs = {name: cls() for name, cls in _EPHEM_BODIES.items()}
    lon = {name: np.empty(len(jd)) for name in objs}
    for i, j in enumerate(jd):
        d = ephem.Date(j - _DJD_OFFSET)
        for name, obj in objs.items():
            obj.compute(d)
            lon[name][i] = float(ephem.Ecliptic(obj, epoch=d).lon)
    for name, arr in lon.items():
        out[name] = np.degrees(arr)
    rahu = mean_rahu(jd)
    out["Rahu"] = rahu
    out["Ketu"] = np.mod(rahu + 180.0, 360.0)
    return out


@dataclass
class Positions:
    """Longitudes (deg, 0-360) and daily speeds (deg/day) per body."""

    index: pd.DatetimeIndex
    longitude: pd.DataFrame
    speed: pd.DataFrame
    zodiac: str

    def at(self, i: int) -> dict:
        return {
            b: {"longitude": float(self.longitude[b].iloc[i]), "speed": float(self.speed[b].iloc[i])}
            for b in BODIES
        }


def compute_positions(ts: pd.DatetimeIndex, zodiac: str = "sidereal") -> Positions:
    """Compute longitudes and speeds for each timestamp.

    ``zodiac``: ``"sidereal"`` (Lahiri, Vedic) or ``"tropical"`` (Western).
    """
    if zodiac not in ("sidereal", "tropical"):
        raise ValueError("zodiac must be 'sidereal' or 'tropical'")
    if len(ts) == 0:
        empty = pd.DataFrame(columns=BODIES, index=ts, dtype=float)
        return Positions(ts, empty, empty.copy(), zodiac)

    jd = to_julian(ts)
    step = GRID_HOURS / 24.0
    start = np.floor(jd.min() / step) * step - 2 * step
    stop = np.ceil(jd.max() / step) * step + 2 * step
    grid = np.arange(start, stop + step / 2, step)
    raw = _raw_tropical(grid)
    ayan_grid = lahiri_ayanamsa(grid) if zodiac == "sidereal" else 0.0

    lon_cols, spd_cols = {}, {}
    for body in BODIES:
        g = np.degrees(np.unwrap(np.radians(raw[body] - ayan_grid)))
        spline = CubicSpline(grid, g)
        lon_cols[body] = np.mod(spline(jd), 360.0)
        spd_cols[body] = spline(jd, 1)  # degrees per day
    lon = pd.DataFrame(lon_cols, index=ts)[BODIES]
    spd = pd.DataFrame(spd_cols, index=ts)[BODIES]
    return Positions(ts, lon, spd, zodiac)


def sign_index(lon: np.ndarray) -> np.ndarray:
    return (np.floor(np.asarray(lon) / 30.0).astype(int)) % 12


def nakshatra_index(lon: np.ndarray) -> np.ndarray:
    return (np.floor(np.asarray(lon) / (360.0 / 27.0)).astype(int)) % 27


def angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest separation between two longitudes, 0..180 deg."""
    d = np.abs(np.mod(np.asarray(a) - np.asarray(b), 360.0))
    return np.minimum(d, 360.0 - d)
