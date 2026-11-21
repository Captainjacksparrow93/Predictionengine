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


def test_surya_siddhanta_matches_text_and_sky():
    import numpy as np
    from astro_engine import surya_siddhanta as ss
    # Constants at the Kali epoch as stated in the text.
    assert abs(ss._start_deg(488_203) - 90) < 1e-6 and abs(ss._start_deg(-232_238) - 180) < 1e-6
    ts = pd.DatetimeIndex([pd.Timestamp("2024-05-02 12:00", tz="Asia/Kolkata")])
    s = compute_positions(ts, ephemeris="surya_siddhanta").longitude
    m = compute_positions(ts, "sidereal").longitude
    diff = lambda b: abs((s[b].iloc[0] - m[b].iloc[0] + 180) % 360 - 180)  # noqa: E731
    assert diff("Sun") < 1 and diff("Moon") < 4 and diff("Jupiter") < 8


def test_lagna_at_sunrise_equals_sun():
    from astro_engine.ephemeris import ascendant_tropical, to_julian
    t = pd.DatetimeIndex([pd.Timestamp("2024-01-01 07:12", tz="Asia/Kolkata")])
    asc = ascendant_tropical(to_julian(t), 19.076, 72.877)[0]
    sun = compute_positions(t, "tropical").longitude["Sun"].iloc[0]
    assert abs(asc - sun) < 2
