"""Turn planetary positions into a boolean "condition" matrix.

Every column is a yes/no astrological state at a moment, for example
``Jupiter in Taurus``, ``Mercury retrograde``, ``Saturn aspects Moon (Vedic)``,
``Yoga Siddhi`` or ``Saturn in house 12 from Moon``. Combinations of
conditions are mined against market moves; a combination is the logical AND
of its columns.

Groups (all switchable in :class:`FeatureConfig`):

* Rashi (sign), nakshatra, Moon's nakshatra pada, navamsa (D9) sign
* Vakri (retrograde) and asta (combustion by the Sun)
* Dignity: exalted (uchcha), debilitated (neecha), own sign (swakshetra)
* Graha drishti (Vedic whole-sign aspects) and planets sharing a sign
* Degree aspects (conjunction, sextile, square, trine, opposition)
* Panchang: tithi, paksha, yoga, karana, vara (weekday lord)
* Lagna (rising sign at the market's location) and houses counted from the
  lagna and from the Moon (Chandra lagna)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .ephemeris import (
    NAKSHATRAS, RETRO_BODIES, SIGNS, Positions,
    angular_distance, ascendant_tropical, nakshatra_index, sign_index,
)
from .market import MARKET_TZ

# Major (Ptolemaic) aspects: angle -> default orb in degrees.
ASPECTS = {
    "conjunct": (0.0, 8.0),
    "sextile": (60.0, 5.0),
    "square": (90.0, 7.0),
    "trine": (120.0, 7.0),
    "opposite": (180.0, 8.0),
}

GRAHAS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
EXALTATION = {"Sun": 0, "Moon": 1, "Mars": 9, "Mercury": 5, "Jupiter": 3, "Venus": 11, "Saturn": 6,
              "Rahu": 1, "Ketu": 7}
OWN_SIGNS = {"Sun": [4], "Moon": [3], "Mars": [0, 7], "Mercury": [2, 5], "Jupiter": [8, 11],
             "Venus": [1, 6], "Saturn": [9, 10]}
# Distance from the Sun (deg) within which a planet is combust (asta); (direct, retrograde).
COMBUSTION_ORB = {"Moon": (12, 12), "Mars": (17, 17), "Mercury": (14, 12), "Jupiter": (11, 11),
                  "Venus": (10, 8), "Saturn": (15, 15)}
# Graha drishti: signs counted from the planet (7th house = +6 signs).
VEDIC_ASPECTS = {"Sun": [6], "Moon": [6], "Mercury": [6], "Venus": [6], "Mars": [3, 6, 7],
                 "Jupiter": [4, 6, 8], "Saturn": [2, 6, 9]}
YOGAS = ["Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda", "Sukarma",
         "Dhriti", "Shula", "Ganda", "Vriddhi", "Dhruva", "Vyaghata", "Harshana", "Vajra", "Siddhi",
         "Vyatipata", "Variyana", "Parigha", "Shiva", "Siddha", "Sadhya", "Shubha", "Shukla",
         "Brahma", "Indra", "Vaidhriti"]
MOVABLE_KARANAS = ["Bava", "Balava", "Kaulava", "Taitila", "Gara", "Vanija", "Vishti"]
VARA = ["Monday (Moon)", "Tuesday (Mars)", "Wednesday (Mercury)", "Thursday (Jupiter)",
        "Friday (Venus)", "Saturday (Saturn)", "Sunday (Sun)"]   # pandas weekday order
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd"}

MUMBAI = (19.0760, 72.8777)

FEATURE_GROUPS = {
    "signs": "Planet in each sign (rashi)",
    "nakshatras": "Planet in each nakshatra",
    "moon_pada": "Moon's nakshatra pada (quarter)",
    "navamsa": "Planet's navamsa (D9) sign",
    "retrograde": "Retrograde (vakri)",
    "combustion": "Combust by the Sun (asta)",
    "dignity": "Exalted, debilitated, own sign",
    "vedic_aspects": "Graha drishti (Vedic aspects)",
    "same_sign": "Two planets in the same sign",
    "aspects": "Degree aspects (conjunction, square, trine ...)",
    "tithi": "Tithi and paksha",
    "yoga": "Panchang yoga",
    "karana": "Karana",
    "vara": "Vara (weekday and its lord)",
    "lagna": "Lagna (rising sign at the market's city)",
    "houses_from_lagna": "Planet's house counted from the lagna",
    "houses_from_moon": "Planet's house counted from the Moon",
}


@dataclass
class FeatureConfig:
    signs: bool = True
    nakshatras: bool = True
    nakshatra_bodies: list[str] = field(default_factory=lambda: list(GRAHAS + ["Uranus", "Neptune", "Pluto"]))
    moon_pada: bool = False
    navamsa: bool = False
    retrograde: bool = True
    combustion: bool = True
    dignity: bool = True
    vedic_aspects: bool = True
    same_sign: bool = True
    aspects: bool = True
    aspect_orb_scale: float = 1.0  # multiply all default orbs
    moon_aspect_orb_scale: float = 0.5  # Moon moves ~13 deg/day: keep its orbs tight
    tithi: bool = True
    yoga: bool = True
    karana: bool = True
    vara: bool = True
    lagna: bool = True
    houses_from_lagna: bool = True
    houses_from_moon: bool = True
    latitude: float = MUMBAI[0]
    longitude: float = MUMBAI[1]
    # Optional natal chart (e.g. NSE / Nifty "birth" moment). Transit-to-natal aspects.
    natal_longitudes: dict[str, float] | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "FeatureConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _ord(n: int) -> str:
    return ORDINAL.get(n, f"{n}th")


def _pair_ok(a: str, b: str) -> bool:
    # Rahu/Ketu are always exactly opposite: that pair carries no information.
    return {a, b} != {"Rahu", "Ketu"}


def build_features(pos: Positions, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Return a DataFrame of uint8 condition columns indexed like ``pos``."""
    cfg = cfg or FeatureConfig()
    lon, spd = pos.longitude, pos.speed
    bodies = list(lon.columns)
    grahas = [b for b in GRAHAS if b in bodies]
    L = {b: lon[b].to_numpy() for b in bodies}
    S = {b: sign_index(L[b]) for b in bodies}
    cols: dict[str, np.ndarray] = {}

    if cfg.signs:
        for b in bodies:
            for i, name in enumerate(SIGNS):
                cols[f"{b} in {name}"] = S[b] == i

    if cfg.nakshatras:
        for b in cfg.nakshatra_bodies:
            if b in L:
                n = nakshatra_index(L[b])
                for i, name in enumerate(NAKSHATRAS):
                    cols[f"{b} in {name} nakshatra"] = n == i

    if cfg.moon_pada and "Moon" in L:
        q = np.floor(L["Moon"] / (360.0 / 108.0)).astype(int) % 108
        for i in range(108):
            cols[f"Moon in {NAKSHATRAS[i // 4]} pada {i % 4 + 1}"] = q == i

    if cfg.navamsa:
        for b in bodies:
            nv = np.floor(L[b] / (30.0 / 9.0)).astype(int) % 12
            for i, name in enumerate(SIGNS):
                cols[f"{b} in {name} navamsa"] = nv == i

    if cfg.retrograde:
        for b in RETRO_BODIES:
            if b in spd:
                cols[f"{b} retrograde"] = spd[b].to_numpy() < 0

    if cfg.combustion and "Sun" in L:
        for b, (orb_d, orb_r) in COMBUSTION_ORB.items():
            if b in L:
                retro = spd[b].to_numpy() < 0
                orb = np.where(retro, orb_r, orb_d)
                cols[f"{b} combust"] = angular_distance(L[b], L["Sun"]) <= orb

    if cfg.dignity:
        for b in grahas:
            ex = EXALTATION[b]
            cols[f"{b} exalted"] = S[b] == ex
            cols[f"{b} debilitated"] = S[b] == (ex + 6) % 12
            if b in OWN_SIGNS:
                cols[f"{b} in own sign"] = np.isin(S[b], OWN_SIGNS[b])

    if cfg.vedic_aspects:
        for src, offsets in VEDIC_ASPECTS.items():
            if src not in S:
                continue
            for tgt in grahas:
                if tgt == src:
                    continue
                rel = (S[tgt] - S[src]) % 12
                cols[f"{src} aspects {tgt} (Vedic)"] = np.isin(rel, offsets)

    if cfg.same_sign:
        for a, b in combinations(grahas, 2):
            if _pair_ok(a, b):
                cols[f"{a} with {b} in same sign"] = S[a] == S[b]

    if cfg.aspects:
        for a, b in combinations(bodies, 2):
            if not _pair_ok(a, b):
                continue
            sep = angular_distance(L[a], L[b])
            moon = "Moon" in (a, b)
            for asp, (angle, orb) in ASPECTS.items():
                o = orb * cfg.aspect_orb_scale * (cfg.moon_aspect_orb_scale if moon else 1.0)
                cols[f"{a} {asp} {b}"] = np.abs(sep - angle) <= o

    if "Sun" in L and "Moon" in L:
        elong = np.mod(L["Moon"] - L["Sun"], 360.0)
        if cfg.tithi:
            tithi = (np.floor(elong / 12.0).astype(int) % 30) + 1
            for t in range(1, 31):
                paksha = "Shukla" if t <= 15 else "Krishna"
                cols[f"Tithi {t} ({paksha})"] = tithi == t
            cols["Shukla paksha (waxing Moon)"] = elong < 180.0
            cols["Krishna paksha (waning Moon)"] = elong >= 180.0
        if cfg.yoga:
            y = np.floor(np.mod(L["Sun"] + L["Moon"], 360.0) / (360.0 / 27.0)).astype(int) % 27
            for i, name in enumerate(YOGAS):
                cols[f"Yoga {name}"] = y == i
        if cfg.karana:
            k = np.floor(elong / 6.0).astype(int) % 60
            name = np.where(k == 0, "Kimstughna", np.where(k == 57, "Shakuni", np.where(
                k == 58, "Chatushpada", np.where(k == 59, "Naga", np.array(MOVABLE_KARANAS)[(k - 1) % 7]))))
            for nm in ["Kimstughna", *MOVABLE_KARANAS, "Shakuni", "Chatushpada", "Naga"]:
                cols[f"Karana {nm}"] = name == nm

    if cfg.vara:
        wd = pos.index.tz_convert(MARKET_TZ).weekday
        for i, nm in enumerate(VARA):
            cols[f"Vara {nm}"] = np.asarray(wd == i)

    if (cfg.lagna or cfg.houses_from_lagna) and pos.jd is not None and len(pos.jd):
        asc = np.mod(ascendant_tropical(pos.jd, cfg.latitude, cfg.longitude) - pos.ayanamsa, 360.0)
        lag = sign_index(asc)
        if cfg.lagna:
            for i, name in enumerate(SIGNS):
                cols[f"Lagna {name}"] = lag == i
        if cfg.houses_from_lagna:
            for b in bodies:
                h = (S[b] - lag) % 12 + 1
                for n in range(1, 13):
                    cols[f"{b} in {_ord(n)} house from Lagna"] = h == n

    if cfg.houses_from_moon and "Moon" in S:
        for b in bodies:
            if b == "Moon":
                continue
            h = (S[b] - S["Moon"]) % 12 + 1
            for n in range(1, 13):
                cols[f"{b} in {_ord(n)} house from Moon"] = h == n

    if cfg.natal_longitudes:
        for tb in bodies:
            for nb, nlon in cfg.natal_longitudes.items():
                sep = angular_distance(L[tb], float(nlon))
                for asp, (angle, orb) in ASPECTS.items():
                    o = orb * cfg.aspect_orb_scale * (cfg.moon_aspect_orb_scale if tb == "Moon" else 1.0)
                    cols[f"transit {tb} {asp} natal {nb}"] = np.abs(sep - angle) <= o

    df = pd.DataFrame(cols, index=pos.index)
    return df.astype(np.uint8)


def active_conditions(features: pd.DataFrame, i: int) -> list[str]:
    row = features.iloc[i]
    return [c for c, v in row.items() if v]
