import io
import time

import pytest
from fastapi.testclient import TestClient

from make_synthetic import make


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("APP_TOKEN", raising=False)
    from app.main import app
    return TestClient(app)


def _wait(client):
    for _ in range(600):
        j = client.get("/api/job").json()
        if j["state"] != "running":
            return j
        time.sleep(0.5)
    raise AssertionError("job timed out")


def test_full_flow(client):
    buf = io.BytesIO()
    make(5, seed=5).to_csv(buf, index=False)
    r = client.post("/api/data/upload", files={"file": ("n.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    assert client.get("/api/data/summary").json()["days"] > 1000

    s = client.get("/api/settings").json()["settings"]
    s["strictness"] = "exploratory"
    s["miner"]["max_depth"] = 1
    s["miner"]["null_runs"] = 0
    s["astro"]["ephemeris"] = "surya_siddhanta"
    assert client.put("/api/settings", json=s).json()["settings"]["strictness"] == "exploratory"

    assert client.post("/api/analyze", json={"question": {"bar": "1D", "horizon": 1, "label": "day"}}).status_code == 200
    j = _wait(client)
    assert j["state"] == "done", j
    pid = j["result"]["pattern_id"]

    lib = client.get("/api/patterns").json()
    assert lib["active"] == pid and lib["items"][0]["id"] == pid
    assert client.get(f"/api/patterns/{pid}").json()["settings"]["astro"]["ephemeris"] == "surya_siddhanta"

    p = client.get(f"/api/patterns/{pid}/predict", params={"start": "2027-01-01", "end": "2027-02-01", "include": "all"}).json()
    assert len(p["points"]) >= 20 and {"target", "call"} <= set(p["points"][0])

    ck = client.post(f"/api/patterns/{pid}/check", json={}).json()
    assert ck["overlap_with_learning"] == 0 and ck["bars"] > 100

    assert client.post("/api/journal", json={"pattern_id": pid, "start": "2027-01-01", "end": "2027-01-10"}).status_code == 200
    assert len(client.get("/api/journal").json()) == 1

    assert client.patch(f"/api/patterns/{pid}", json={"name": "renamed"}).json()["name"] == "renamed"
    ev = client.get("/api/events", params={"start": "2027-01-01", "end": "2027-03-01"}).json()
    assert any(e["type"] == "ingress" for e in ev)
    pos = client.get("/api/positions").json()
    assert pos["panchang"]["yoga"] and len(pos["bodies"]) == 12
    assert client.delete(f"/api/patterns/{pid}").status_code == 200


def test_source_config_reports_missing_env(client):
    from astro_engine.data_sources import PRESETS
    r = client.put("/api/source", json=PRESETS["zerodha_kite"]).json()
    assert "KITE_ACCESS_TOKEN" in r["missing_env"]


def test_token_guard(client, monkeypatch):
    monkeypatch.setenv("APP_TOKEN", "s3cret")
    assert client.get("/api/data/summary").status_code == 401
    assert client.get("/api/data/summary", headers={"X-App-Token": "s3cret"}).status_code == 200
