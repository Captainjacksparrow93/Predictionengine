"""Planetary longitudes by the Surya Siddhanta (classical Indian astronomy).

Implements the text's own model (Burgess's translation, chapters I-II):

* Mean motions from the number of revolutions in a Mahayuga
  (4,320,000 years = 1,577,917,828 civil days), counted from the start of
  the Kali Yuga (midnight at Ujjain, 17/18 Feb 3102 BCE).
* Apsides (mandocca) from their revolutions in a Kalpa; their positions at the
  Kali epoch follow from the 1,955,880,000 years of motion before it, which
  gives, for example, the Sun's apogee at 77.1 deg, the Moon's apogee at 90 deg
  and its node at 180 deg, as the text states.
* The equation of centre (manda) and, for the five star-planets, the equation
  of conjunction (shighra), with epicycles that vary between their values at
  the ends of the even and odd quadrants, using the text's four-step procedure.

Longitudes are sidereal in the Surya Siddhanta's own frame (no separate
ayanamsa). Uranus, Neptune and Pluto are not part of the text.
"""
from __future__ import annotations

import numpy as np

CIVIL_DAYS_MAHAYUGA = 1_577_917_828
YEARS_MAHAYUGA = 4_320_000
YEARS_KALPA = 4_320_000_000
YEARS_BEFORE_KALI = 1_955_880_000      # years of planetary motion from creation to Kali Yuga
KALI_EPOCH_JD = 588_465.5              # Julian day of the Kali epoch (midnight)
UJJAIN_LON_DEG = 75.77                 # the prime meridian of the text

# Revolutions in a Mahayuga (sidereal). Mercury/Venus: their shighra (fast) revolutions.
REV_MAHAYUGA = {
    "Sun": 4_320_000, "Moon": 57_753_336, "Mars": 2_296_832, "Mercury": 17_937_060,
    "Jupiter": 364_220, "Venus": 7_022_376, "Saturn": 146_568,
    "MoonApogee": 488_203, "MoonNode": -232_238,
}
# Revolutions of the apsides (mandocca) in a Kalpa.
APSIS_REV_KALPA = {"Sun": 387, "Mars": 204, "Mercury": 368, "Jupiter": 900, "Venus": 535, "Saturn": 39}
# Manda epicycles (degrees) at the end of (even, odd) quadrants.
MANDA_EPI = {"Sun": (14.0, 13 + 40 / 60), "Moon": (32.0, 31 + 40 / 60), "Mars": (75.0, 72.0),
             "Mercury": (30.0, 28.0), "Jupiter": (33.0, 32.0), "Venus": (12.0, 11.0), "Saturn": (49.0, 48.0)}
# Shighra epicycles (degrees) at the end of (even, odd) quadrants.
SHIGHRA_EPI = {"Mars": (235.0, 232.0), "Mercury": (133.0, 132.0), "Jupiter": (72.0, 70.0),
               "Venus": (262.0, 260.0), "Saturn": (40.0, 39.0)}

_D2R = np.pi / 180.0


def ahargana(jd_ut: np.ndarray) -> np.ndarray:
    """Civil days elapsed since the Kali epoch, reckoned at Ujjain."""
    return np.asarray(jd_ut, dtype=float) + UJJAIN_LON_DEG / 360.0 - KALI_EPOCH_JD


def _mean(revs_mahayuga: float, days: np.ndarray, start_deg: float = 0.0) -> np.ndarray:
    # Exact rational arithmetic on revolutions keeps precision over 5,000 years.
    revs = revs_mahayuga * days / CIVIL_DAYS_MAHAYUGA
    return np.mod(start_deg + 360.0 * (revs - np.floor(revs)), 360.0)


def _start_deg(revs_mahayuga: float) -> float:
    revs = revs_mahayuga * (YEARS_BEFORE_KALI / YEARS_MAHAYUGA)
    return 360.0 * (revs - np.floor(revs))


def _apsis(body: str, days: np.ndarray) -> np.ndarray:
    per_mahayuga = APSIS_REV_KALPA[body] * YEARS_MAHAYUGA / YEARS_KALPA
    return _mean(per_mahayuga, days, _start_deg(per_mahayuga))


def _epi(table: tuple[float, float], anomaly_deg: np.ndarray) -> np.ndarray:
    even, odd = table
    return even - (even - odd) * np.abs(np.sin(anomaly_deg * _D2R))


def _manda_eq(body: str, planet: np.ndarray, apogee: np.ndarray) -> np.ndarray:
    k = planet - apogee
    r = _epi(MANDA_EPI[body], k) / 360.0
    return -np.degrees(np.arcsin(np.clip(r * np.sin(k * _D2R), -1, 1)))


def _shighra_eq(body: str, planet: np.ndarray, shighrocca: np.ndarray) -> np.ndarray:
    k = shighrocca - planet
    r = _epi(SHIGHRA_EPI[body], k) / 360.0
    return np.degrees(np.arctan2(r * np.sin(k * _D2R), 1.0 + r * np.cos(k * _D2R)))


def longitudes(jd_ut: np.ndarray) -> dict[str, np.ndarray]:
    """True (sphuta) longitudes in degrees for Sun..Saturn, Rahu and Ketu."""
    days = ahargana(jd_ut)
    out: dict[str, np.ndarray] = {}
    sun_mean = _mean(REV_MAHAYUGA["Sun"], days)
    out["Sun"] = np.mod(sun_mean + _manda_eq("Sun", sun_mean, _apsis("Sun", days)), 360)

    moon_mean = _mean(REV_MAHAYUGA["Moon"], days)
    moon_apogee = _mean(REV_MAHAYUGA["MoonApogee"], days, _start_deg(REV_MAHAYUGA["MoonApogee"]))
    out["Moon"] = np.mod(moon_mean + _manda_eq("Moon", moon_mean, moon_apogee), 360)

    for body in ("Mars", "Mercury", "Jupiter", "Venus", "Saturn"):
        if body in ("Mercury", "Venus"):
            mean = sun_mean                                   # the planet's mean = mean Sun
            shighrocca = _mean(REV_MAHAYUGA[body], days)      # its own fast revolution
        else:
            mean = _mean(REV_MAHAYUGA[body], days)
            shighrocca = sun_mean
        apogee = _apsis(body, days)
        # Four-step procedure of Surya Siddhanta II.43-45.
        p1 = mean + 0.5 * _shighra_eq(body, mean, shighrocca)
        p2 = p1 + 0.5 * _manda_eq(body, p1, apogee)
        p3 = mean + _manda_eq(body, p2, apogee)
        out[body] = np.mod(p3 + _shighra_eq(body, p3, shighrocca), 360)

    node = _mean(REV_MAHAYUGA["MoonNode"], days, _start_deg(REV_MAHAYUGA["MoonNode"]))
    out["Rahu"] = node
    out["Ketu"] = np.mod(node + 180.0, 360)
    return out
