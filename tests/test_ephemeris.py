import pandas as pd

from astro_engine.ephemeris import SIGNS, compute_positions, sign_index


def _at(s, zodiac="sidereal"):
    return compute_positions(pd.DatetimeIndex([pd.Timestamp(s, tz="Asia/Kolkata")]), zodiac)


def sign(p, body):
    return SIGNS[int(sign_index(p.longitude[body].iloc[0]))]


def test_known_vedic_transits():
    assert sign(_at("2023-01-16 12:00"), "Saturn") == "Capricorn"
    assert sign(_at("2023-01-18 12:00"), "Saturn") == "Aquarius"   # Saturn ingress 17 Jan 2023
    assert sign(_at("2024-04-29 12:00"), "Jupiter") == "Aries"
    assert sign(_at("2024-05-02 12:00"), "Jupiter") == "Taurus"    # Jupiter ingress 1 May 2024
    assert sign(_at("2024-01-01 12:00"), "Rahu") == "Pisces"


def test_mercury_retrograde_april_2024():
    assert _at("2024-03-30 12:00").speed["Mercury"].iloc[0] > 0
    assert _at("2024-04-10 12:00").speed["Mercury"].iloc[0] < 0
    assert _at("2024-04-28 12:00").speed["Mercury"].iloc[0] > 0


def test_eclipse_sun_position_tropical():
    p = _at("2024-04-08 23:50", "tropical")  # total solar eclipse at ~19.4 deg Aries
    assert abs(p.longitude["Sun"].iloc[0] - 19.4) < 0.3
    assert abs(p.longitude["Moon"].iloc[0] - p.longitude["Sun"].iloc[0]) < 0.6
