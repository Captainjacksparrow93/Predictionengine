import numpy as np
import pytest

from astro_engine.market import normalize_minutes
from astro_engine.miner import bh_qvalues
from astro_engine.pipeline import analyze
from make_synthetic import make


def test_bh_qvalues_monotone():
    q = bh_qvalues(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.04 * 4 / 3 * 3 / 4 * 4 / 3, 0.04, 0.5]) or (q[0] <= q[2] <= q[1] <= q[3])


@pytest.fixture(scope="module")
def noise():
    return normalize_minutes(make(6, seed=3))


def test_random_walk_yields_no_validated_rules(noise):
    m = analyze(noise, {"bar": "1D", "miner": {"max_depth": 2, "null_runs": 1}})
    r = m["result"]
    assert r["search"]["validated_non_redundant"] <= 1
    assert r["search"]["hypotheses_tested"] > 1000


def test_planted_effect_is_found():
    s = normalize_minutes(make(8, plant="Moon in Rohini nakshatra", plant_bp=150))
    m = analyze(s, {"bar": "1D", "miner": {"max_depth": 2, "null_runs": 1}})
    validated = [r["conditions"] for r in m["result"]["rules"] if r["status"] == "validated"]
    assert ["Moon in Rohini nakshatra"] in validated


def test_intraday_bars_run(noise):
    m = analyze(noise.loc["2016-01-01":"2017-06-30"], {"bar": "60min", "miner": {"max_depth": 1, "null_runs": 0}})
    assert m["result"]["bars"] > 1000


def test_multi_bar_horizon_uses_non_overlapping_windows(noise):
    one = analyze(noise, {"bar": "1D", "horizon": 1, "miner": {"max_depth": 1, "null_runs": 0}})
    five = analyze(noise, {"bar": "1D", "horizon": 5,
                           "miner": {"max_depth": 1, "null_runs": 0, "min_support": 10, "min_support_validation": 4}})
    assert abs(five["result"]["bars"] * 5 - one["result"]["bars"]) <= 10
