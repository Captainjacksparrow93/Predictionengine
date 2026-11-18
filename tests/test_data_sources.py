import httpx

from astro_engine.data_sources import PRESETS, ApiSource, fetch_api


def test_kite_style_array_candles_chunked(monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "k")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "t")
    calls = []

    def handler(req: httpx.Request):
        calls.append(req)
        assert req.headers["Authorization"] == "token k:t"
        frm = req.url.params["from"][:10]
        candles = [[f"{frm}T09:15:00+0530", 1, 1, 1, 22000.5, 0], [f"{frm}T09:16:00+0530", 1, 1, 1, 22001.0, 0]]
        return httpx.Response(200, json={"status": "success", "data": {"candles": candles}})

    src = ApiSource.from_dict(PRESETS["zerodha_kite"])
    client = httpx.Client(transport=httpx.MockTransport(handler))
    s = fetch_api(src, "2024-01-01", "2024-04-29 23:59:59", client=client, pause=0)
    assert len(calls) == 2  # 60-day chunks
    assert len(s) == 4 and s.iloc[0] == 22000.5
    assert str(s.index.tz) == "Asia/Kolkata"
