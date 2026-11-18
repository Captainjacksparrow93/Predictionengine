import io

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
    import time
    for _ in range(600):
        j = client.get("/api/job").json()
        if j["state"] != "running":
            return j
        time.sleep(0.5)
    raise AssertionError("job timed out")


def test_full_flow(client):
    buf = io.BytesIO()
    make(4, seed=5).to_csv(buf, index=False)
    r = client.post("/api/data/upload", files={"file": ("n.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    assert client.get("/api/data/summary").json()["days"] > 900

    assert client.post("/api/analyze", json={"bar": "1D", "miner": {"max_depth": 1, "null_runs": 0}}).status_code == 200
    j = _wait(client)
    assert j["state"] == "done", j
    m = client.get("/api/model").json()
    assert "verdict" in m["result"]

    f = client.get("/api/forecast", params={"start": "2027-01-01", "end": "2027-02-01", "include": "all"}).json()
    assert len(f["points"]) >= 20
    ev = client.get("/api/events", params={"start": "2027-01-01", "end": "2027-03-01"}).json()
    assert any(e["type"] == "ingress" for e in ev)
    assert len(client.get("/api/positions").json()["bodies"]) == 12


def test_source_config_reports_missing_env(client):
    from astro_engine.data_sources import PRESETS
    r = client.put("/api/source", json=PRESETS["zerodha_kite"]).json()
    assert "KITE_ACCESS_TOKEN" in r["missing_env"]


def test_token_guard(client, monkeypatch):
    monkeypatch.setenv("APP_TOKEN", "s3cret")
    assert client.get("/api/data/summary").status_code == 401
    assert client.get("/api/data/summary", headers={"X-App-Token": "s3cret"}).status_code == 200
