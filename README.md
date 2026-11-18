# Nifty 50 Astro Prediction Engine

Loads minute-level Nifty 50 history from your market-data API, computes where
every planet was at each bar, and searches all planetary conditions and their
combinations for ones that line up with **rises (positive)** or **falls
(negative)** of the index. Rules that hold up on later data are then applied to
**future planetary positions** to give a positive / negative / neutral signal
per day (or per intraday bar). It uses closing prices only, not volume.

## What it considers

| Family | Conditions |
|---|---|
| Bodies | Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Rahu, Ketu, Uranus, Neptune, Pluto |
| Signs | each body in each of the 12 signs (sidereal/Lahiri or tropical) |
| Nakshatras | each body in each of the 27 nakshatras |
| Motion | retrograde for Mercury to Pluto |
| Aspects | conjunction, sextile, square, trine, opposition for every pair of bodies (tighter orbs for the Moon) |
| Lunar | all 30 tithis, Shukla / Krishna paksha |
| Natal (optional) | transit aspects to a natal chart (for example a chart for the Nifty base date) via `features.natal_longitudes` |

That is about 830 conditions. Depth 2 tests every pair of them (hundreds of
thousands of candidates, pruned to those that occur often enough). Depth 3
extends the strongest pairs to triples.

Positions come from PyEphem (VSOP87/ELP theories, accurate to arc-seconds for
1900–2100), computed on a 2-hour grid and spline-interpolated to every bar.
Rahu/Ketu are the mean nodes. Tests check the results against real events:
Saturn entering Aquarius on 17 Jan 2023, Jupiter entering Taurus on 1 May 2024,
Mercury retrograde in April 2024, and the 8 Apr 2024 eclipse.

## Why the statistics matter

If you test enough combinations against one price history, some will always
look like strong predictors by pure luck. The engine is built to separate
those from anything real:

1. **Chronological split.** Rules are discovered on the oldest 60% of bars,
   validated on the next 20%, and scored on the newest 20%, which the search
   never looks at.
2. **False discovery rate (Benjamini–Hochberg).** Every combination tested, in
   both directions, counts toward the correction, at discovery and again at
   validation.
3. **Null control.** The whole search is re-run with the price series shifted in
   time, which breaks any real planet–market link. If the shifted data produces
   as many "validated" rules as the real data, the rules are chance findings.
4. **Redundancy pruning.** A combination that fires on the same days as a
   simpler rule (for example "Uranus in Aries AND Moon in Rohini" when Uranus
   sat in Aries for the whole period) is marked redundant, so it is not counted
   twice.

The analysis ends with a plain-language **verdict** that combines all four.
On a pure random walk the engine finds nothing. On synthetic data with a
planted effect it finds that exact condition (see `tests/test_miner.py`).
Expect real-market results to look like the random walk. If a rule survives all
the checks, test it on fresh data before you trade on it.

## Running locally

```bash
pip install -r requirements-dev.txt
```

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000. To try the engine without an API, generate sample data
and upload it in the UI:

```bash
python scripts/make_synthetic.py --years 8 --out data/synthetic.csv
```

Run the tests with:

```bash
pytest -q
```

## Configuring your data API

In the UI, section 1, edit the JSON config:

```json
{
  "url": "https://api.kite.trade/instruments/historical/256265/minute",
  "headers": {"X-Kite-Version": "3", "Authorization": "token ${KITE_API_KEY}:${KITE_ACCESS_TOKEN}"},
  "params": {"from": "{from}", "to": "{to}"},
  "date_format": "%Y-%m-%d %H:%M:%S",
  "records_path": "data.candles",
  "timestamp_field": "0",
  "close_field": "4",
  "chunk_days": 60
}
```

- `{from}` / `{to}` are filled for each request window, formatted with `date_format`;
  history is fetched in `chunk_days` windows.
- `${NAME}` is replaced from server environment variables, so keys are never stored.
- `records_path` is the dotted path to the candle list. `timestamp_field` / `close_field`
  are object keys, or list indexes for array candles.
- Set `timestamp_unit` to `"s"` or `"ms"` for epoch timestamps. Naive times are read as IST.

There is a preset for **Zerodha Kite Connect** (NIFTY 50 instrument token 256265).
Other Indian providers with minute history, such as Upstox, Angel One SmartAPI,
Fyers, TrueData and Global Datafeeds, work by changing the URL, headers and field
mapping. You can also upload a CSV with `timestamp,close` columns.

## API

| Method | Path | Purpose |
|---|---|---|
| PUT/GET | `/api/source` | save / read the data-API config |
| POST | `/api/data/fetch` | `{start, end}` fetch into storage (background job) |
| POST | `/api/data/upload` | CSV upload (`?replace=true` to overwrite) |
| GET | `/api/data/summary` | coverage + daily closes |
| POST | `/api/analyze` | settings: `bar`, `horizon`, `threshold_pct`, `zodiac`, `features{}`, `miner{}` (background job) |
| GET | `/api/job` | job state / result |
| GET | `/api/model` | full results and rules (`?rules=validated`) |
| GET | `/api/forecast` | `start`, `end`, `include=validated\|all` → signal per bar plus the rules that fired |
| GET | `/api/events` | ingresses, retrograde stations and exact aspects in a range |
| GET | `/api/positions` | planetary positions at `at` (default now) |

When the server has `APP_TOKEN` set, every `/api` call needs the header `X-App-Token`.

## Deploying on your own VPS (e.g. Hostinger) with Docker

`docker-compose.yml` runs the app behind Caddy, which gets and renews HTTPS
certificates automatically.

1. Optional, for HTTPS: point a domain at the VPS. In your DNS, add an **A
   record**, e.g. `astro.yourdomain.com` → your VPS IP.
2. On the VPS (via SSH), get the code and configure it:

   ```bash
   git clone -b claude/wonderful-goldberg-en15qi https://github.com/captainjacksparrow93/predictionengine.git
   ```

   ```bash
   cd predictionengine && cp .env.example .env && nano .env
   ```

   Set `DOMAIN` (or `:80` if you have no domain yet), a long random `APP_TOKEN`
   (`openssl rand -hex 24`), and your Kite keys.
3. Open the firewall (Hostinger panel → VPS → Firewall, and/or `ufw`) for ports 80 and 443, then start:

   ```bash
   docker compose up -d --build
   ```

4. Open `https://astro.yourdomain.com` (or `http://<VPS-IP>`), paste `APP_TOKEN`
   into the "App token" box, and fetch or upload data.

Day-to-day:

| Task | Command |
|---|---|
| Update the daily Kite access token | edit `.env`, then `docker compose up -d` |
| Update to new code | `git pull && docker compose up -d --build` |
| Logs | `docker compose logs -f app` |
| Back up data | `docker run --rm -v predictionengine_astro_data:/d -v $PWD:/b alpine tar czf /b/astro-backup.tgz -C /d .` |

Price history and results live in the `astro_data` Docker volume, so they
survive restarts and rebuilds. Use `docker compose down -v` only if you want to
erase them.

## Deployment

The app is a single Docker container that needs a small persistent disk
(price history and model live in `DATA_DIR`) and about 1–2 GB RAM for intraday
analysis. One analysis of 10 years of minute data takes about 10 s and peaks at
about 550 MB.

**Recommended: Render** (`render.yaml` included). Push the repo, then choose
New → Blueprint. It builds the Dockerfile, mounts a 2 GB disk at `/data`,
generates an `APP_TOKEN`, and runs in Singapore, the closest region to India. The
Standard plan (2 GB) costs about $25/month plus disk. Set `KITE_API_KEY` /
`KITE_ACCESS_TOKEN` in the dashboard.

Alternatives:
- **Railway**: same Dockerfile, add a volume at `/data`. Usage-based pricing,
  usually cheaper for light use.
- **AWS Lightsail / DigitalOcean / Hetzner VM in Mumbai or Bangalore**: lowest
  latency to Indian brokers and cheapest at a fixed price. Run
  `docker run -v astro:/data -p 80:8000 ...` behind Caddy for HTTPS. Choose this
  if your broker needs a static IP (SEBI rules for broker API access are moving
  that way).
- **Google Cloud Run (asia-south1)**: scales to zero, but needs Cloud Storage or
  Filestore for `/data`, because its disk is ephemeral.

Vercel and Netlify are not a good fit: analysis runs longer than their function
time limits, and they have no persistent disk.

Kite access tokens expire daily, so update `KITE_ACCESS_TOKEN` (or add a login
flow) before each fetch.
