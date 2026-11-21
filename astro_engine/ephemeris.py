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

from . import surya_siddhanta

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

GRID_HOURS = 6.0  # cubic spline error < 0.01 arc-second even for the Moon


def to_julian(ts: pd.DatetimeIndex) -> np.ndarray:
    """UTC DatetimeIndex -> Julian Day (UT)."""
    if ts.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    days = (ts.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1)
    return np.asarray(days, dtype=np.float64) + _UNIX_EPOCH_JD


# Ayanamsa at J2000.0 (degrees) for each sidereal convention; all precess at
# ~50.29"/year. Values agree with the standard tables to within a few arcminutes.
AYANAMSAS = {
    "lahiri": ("Lahiri (Chitrapaksha), Govt. of India standard", 23.85305),
    "raman": ("B.V. Raman", 22.41093),
    "kp": ("Krishnamurti (KP)", 23.75780),
    "yukteshwar": ("Sri Yukteshwar", 22.47853),
    "fagan_bradley": ("Fagan-Bradley (Western sidereal)", 24.74034),
    "tropical": ("Tropical / Western (no ayanamsa)", 0.0),
}
PRECESSION_DEG_PER_YEAR = 50.2888 / 3600.0


def ayanamsa_deg(jd: np.ndarray, name: str = "lahiri") -> np.ndarray:
    if name not in AYANAMSAS:
        raise ValueError(f"unknown ayanamsa {name!r}; choose from {list(AYANAMSAS)}")
    base = AYANAMSAS[name][1]
    if base == 0.0:
        return np.zeros_like(np.asarray(jd, dtype=float))
    return base + (np.asarray(jd) - _J2000_JD) / 365.25 * PRECESSION_DEG_PER_YEAR


def lahiri_ayanamsa(jd: np.ndarray) -> np.ndarray:
    return ayanamsa_deg(jd, "lahiri")


def mean_rahu(jd: np.ndarray) -> np.ndarray:
    """Tropical longitude of the mean ascending lunar node (Meeus, ch. 47)."""
    t = (jd - _J2000_JD) / 36525.0
    om = 125.0445479 - 1934.1362891 * t + 0.0020754 * t**2 + t**3 / 467441.0 - t**4 / 60616000.0
    return np.mod(om, 360.0)


def true_rahu(jd: np.ndarray) -> np.ndarray:
    """True (osculating) node: mean node plus Meeus's periodic terms (~0.01 deg)."""
    t = (np.asarray(jd) - _J2000_JD) / 36525.0
    d = np.radians(297.8501921 + 445267.1114034 * t)
    m = np.radians(357.5291092 + 35999.0502909 * t)
    mp = np.radians(134.9633964 + 477198.8675055 * t)
    f = np.radians(93.2720950 + 483202.0175233 * t)
    corr = (-1.4979 * np.sin(2 * (d - f)) - 0.1500 * np.sin(m) - 0.1226 * np.sin(2 * d)
            + 0.1176 * np.sin(2 * f) - 0.0801 * np.sin(2 * (mp - f)))
    return np.mod(mean_rahu(jd) + corr, 360.0)


def ascendant_tropical(jd: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """Tropical longitude of the rising degree (lagna) at a place."""
    jd = np.asarray(jd, dtype=float)
    t = (jd - _J2000_JD) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - _J2000_JD) + 0.000387933 * t**2
    ramc = np.radians(np.mod(gmst + lon_deg, 360.0))
    eps = np.radians(23.4392911 - 0.0130042 * t)
    phi = np.radians(lat_deg)
    asc = np.arctan2(np.cos(ramc), -(np.sin(ramc) * np.cos(eps) + np.tan(phi) * np.sin(eps)))
    return np.mod(np.degrees(asc), 360.0)


def _raw_tropical(jd: np.ndarray, only: set[str] | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    objs = {name: cls() for name, cls in _EPHEM_BODIES.items() if only is None or name in only}
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


EPHEMERIDES = {
    "modern": "Modern astronomy (VSOP87 / ELP, arc-second accuracy)",
    "surya_siddhanta": "Surya Siddhanta (classical Indian model)",
}


@dataclass
class Positions:
    """Longitudes (deg, 0-360) and daily speeds (deg/day) per body."""

    index: pd.DatetimeIndex
    longitude: pd.DataFrame
    speed: pd.DataFrame
    zodiac: str
    jd: np.ndarray | None = None
    ayanamsa: np.ndarray | None = None   # degrees subtracted from tropical positions
    bodies: list[str] | None = None

    def at(self, i: int) -> dict:
        return {
            b: {"longitude": float(self.longitude[b].iloc[i]), "speed": float(self.speed[b].iloc[i])}
            for b in self.longitude.columns
        }


def _grid(jd: np.ndarray) -> np.ndarray:
    step = GRID_HOURS / 24.0
    start = np.floor(jd.min() / step) * step - 2 * step
    stop = np.ceil(jd.max() / step) * step + 2 * step
    return np.arange(start, stop + step / 2, step)


def compute_positions(ts: pd.DatetimeIndex, zodiac: str = "sidereal", ephemeris: str = "modern",
                      ayanamsa: str | None = None, node: str = "mean", outer_planets: bool = True) -> Positions:
    """Longitudes and speeds for each timestamp.

    ``ephemeris``: ``"modern"`` or ``"surya_siddhanta"``.
    ``ayanamsa``: a key of :data:`AYANAMSAS` (``"tropical"`` = Western). If omitted,
    ``zodiac`` decides (``"sidereal"`` -> Lahiri, ``"tropical"`` -> none).
    Surya Siddhanta longitudes are already sidereal in the text's own frame; the
    ayanamsa is then used only for Uranus/Neptune/Pluto and the lagna.
    ``node``: ``"mean"`` or ``"true"`` Rahu/Ketu (modern ephemeris only).
    """
    if zodiac not in ("sidereal", "tropical"):
        raise ValueError("zodiac must be 'sidereal' or 'tropical'")
    if ephemeris not in EPHEMERIDES:
        raise ValueError(f"ephemeris must be one of {list(EPHEMERIDES)}")
    ayanamsa = ayanamsa or ("lahiri" if zodiac == "sidereal" else "tropical")
    if ephemeris == "surya_siddhanta" and ayanamsa == "tropical":
        ayanamsa = "lahiri"
    bodies = [b for b in BODIES if outer_planets or b not in ("Uranus", "Neptune", "Pluto")]
    if len(ts) == 0:
        empty = pd.DataFrame(columns=bodies, index=ts, dtype=float)
        return Positions(ts, empty, empty.copy(), zodiac, np.array([]), np.array([]), bodies)

    jd = to_julian(ts)
    grid = _grid(jd)
    need = {"Uranus", "Neptune", "Pluto"} & set(bodies) if ephemeris == "surya_siddhanta" else None
    raw = _raw_tropical(grid, need)
    if node == "true":
        raw["Rahu"] = true_rahu(grid)
        raw["Ketu"] = np.mod(raw["Rahu"] + 180.0, 360.0)
    ayan_grid = ayanamsa_deg(grid, ayanamsa)
    sid = {b: raw[b] - ayan_grid for b in raw}
    if ephemeris == "surya_siddhanta":
        sid.update(surya_siddhanta.longitudes(grid))

    lon_cols, spd_cols = {}, {}
    for body in bodies:
        g = np.degrees(np.unwrap(np.radians(sid[body])))
        spline = CubicSpline(grid, g)
        lon_cols[body] = np.mod(spline(jd), 360.0)
        spd_cols[body] = spline(jd, 1)  # degrees per day
    lon = pd.DataFrame(lon_cols, index=ts)[bodies]
    spd = pd.DataFrame(spd_cols, index=ts)[bodies]
    zod = "tropical" if ayanamsa == "tropical" and ephemeris == "modern" else "sidereal"
    return Positions(ts, lon, spd, zod, jd, ayanamsa_deg(jd, ayanamsa), bodies)


def sign_index(lon: np.ndarray) -> np.ndarray:
    return (np.floor(np.asarray(lon) / 30.0).astype(int)) % 12


def nakshatra_index(lon: np.ndarray) -> np.ndarray:
    return (np.floor(np.asarray(lon) / (360.0 / 27.0)).astype(int)) % 27


def angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest separation between two longitudes, 0..180 deg."""
    d = np.abs(np.mod(np.asarray(a) - np.asarray(b), 360.0))
    return np.minimum(d, 360.0 - d)
