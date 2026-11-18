"""Turn planetary positions into a boolean "condition" matrix.

Every column is a yes/no astrological state, e.g. ``Jupiter in Taurus``,
``Mercury retrograde``, ``Mars square Saturn``, ``Moon in Rohini`` or
``Tithi 15``. Combinations of conditions are later mined against market
moves; a combination is simply the logical AND of its columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .ephemeris import (
    BODIES, NAKSHATRAS, RETRO_BODIES, SIGNS, Positions,
    angular_distance, nakshatra_index, sign_index,
)

# Major (Ptolemaic) aspects: angle -> default orb in degrees.
ASPECTS = {
    "conjunct": (0.0, 8.0),
    "sextile": (60.0, 5.0),
    "square": (90.0, 7.0),
    "trine": (120.0, 7.0),
    "opposite": (180.0, 8.0),
}


@dataclass
class FeatureConfig:
    signs: bool = True
    nakshatras: bool = True
    nakshatra_bodies: list[str] = field(default_factory=lambda: list(BODIES))
    retrograde: bool = True
    aspects: bool = True
    aspect_orb_scale: float = 1.0  # multiply all default orbs
    moon_aspect_orb_scale: float = 0.5  # Moon moves ~13 deg/day: keep its orbs tight
    tithi: bool = True
    # Optional natal chart (e.g. NSE / Nifty "birth" moment). Transit-to-natal aspects.
    natal_longitudes: dict[str, float] | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "FeatureConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _pair_ok(a: str, b: str) -> bool:
    # Rahu/Ketu are always exactly opposite: that pair carries no information.
    return {a, b} != {"Rahu", "Ketu"}


def build_features(pos: Positions, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Return a DataFrame of uint8 condition columns indexed like ``pos``."""
    cfg = cfg or FeatureConfig()
    lon, spd = pos.longitude, pos.speed
    cols: dict[str, np.ndarray] = {}

    if cfg.signs:
        for b in BODIES:
            s = sign_index(lon[b].to_numpy())
            for i, name in enumerate(SIGNS):
                cols[f"{b} in {name}"] = s == i

    if cfg.nakshatras:
        for b in cfg.nakshatra_bodies:
            if b not in lon:
                continue
            n = nakshatra_index(lon[b].to_numpy())
            for i, name in enumerate(NAKSHATRAS):
                cols[f"{b} in {name} nakshatra"] = n == i

    if cfg.retrograde:
        for b in RETRO_BODIES:
            cols[f"{b} retrograde"] = spd[b].to_numpy() < 0

    if cfg.aspects:
        for a, b in combinations(BODIES, 2):
            if not _pair_ok(a, b):
                continue
            sep = angular_distance(lon[a].to_numpy(), lon[b].to_numpy())
            moon = "Moon" in (a, b)
            for asp, (angle, orb) in ASPECTS.items():
                o = orb * cfg.aspect_orb_scale * (cfg.moon_aspect_orb_scale if moon else 1.0)
                cols[f"{a} {asp} {b}"] = np.abs(sep - angle) <= o

    if cfg.tithi:
        elong = np.mod(lon["Moon"].to_numpy() - lon["Sun"].to_numpy(), 360.0)
        tithi = (np.floor(elong / 12.0).astype(int) % 30) + 1
        for t in range(1, 31):
            paksha = "Shukla" if t <= 15 else "Krishna"
            cols[f"Tithi {t} ({paksha})"] = tithi == t
        cols["Shukla paksha (waxing Moon)"] = elong < 180.0
        cols["Krishna paksha (waning Moon)"] = elong >= 180.0

    if cfg.natal_longitudes:
        for tb in BODIES:
            for nb, nlon in cfg.natal_longitudes.items():
                sep = angular_distance(lon[tb].to_numpy(), float(nlon))
                for asp, (angle, orb) in ASPECTS.items():
                    o = orb * cfg.aspect_orb_scale * (cfg.moon_aspect_orb_scale if tb == "Moon" else 1.0)
                    cols[f"transit {tb} {asp} natal {nb}"] = np.abs(sep - angle) <= o

    df = pd.DataFrame(cols, index=pos.index)
    return df.astype(np.uint8)


def active_conditions(features: pd.DataFrame, i: int) -> list[str]:
    row = features.iloc[i]
    return [c for c, v in row.items() if v]
