"""Generate a self-contained docker-compose file for a VPS that sits behind an
existing Traefik reverse proxy (e.g. Hostinger's "Ubuntu with n8n" template).

The app source is embedded in an inline Dockerfile, so the VPS can build the
image from the compose file alone: no repository access needed. This suits
deploy tools that accept only a compose file (such as Hostinger's Docker
Manager API).

    python scripts/build_vps_compose.py --host astro.srv1275499.hstgr.cloud > deploy.yml

Secrets (APP_TOKEN, KITE_*) come from the project environment, never this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "requirements.txt",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "astro_engine").glob("*.py")),
    "app/main.py",
    "app/static/index.html",
    "scripts/make_synthetic.py",
]


def dockerfile() -> str:
    out = [
        "# syntax=docker/dockerfile:1.7",
        "FROM python:3.12-slim",
        "ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data PORT=8000",
        "WORKDIR /app",
    ]
    for i, rel in enumerate(FILES):
        body = (ROOT / rel).read_text()
        tag = f"__EOF_{i}__"
        assert tag not in body
        out.append(f'COPY <<"{tag}" /app/{rel}')
        out.append(body.rstrip("\n"))
        out.append(tag)
        if rel == "requirements.txt":
            out.append("RUN pip install --no-cache-dir -r requirements.txt")
    out.append("RUN mkdir -p /data")
    out.append('CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]')
    return "\n".join(out) + "\n"


def compose(host: str, router: str, network: str, resolver: str, mem: str) -> str:
    df = dockerfile().replace("$", "$$")  # escape compose interpolation
    df_block = "\n".join(("        " + line) if line else "" for line in df.splitlines())
    return f"""name: {router}
services:
  app:
    build:
      context: .
      dockerfile_inline: |
{df_block}
    image: {router}:latest
    restart: unless-stopped
    environment:
      DATA_DIR: /data
      APP_TOKEN: ${{APP_TOKEN:?set APP_TOKEN}}
      KITE_API_KEY: ${{KITE_API_KEY:-}}
      KITE_ACCESS_TOKEN: ${{KITE_ACCESS_TOKEN:-}}
      TZ: Asia/Kolkata
    volumes:
      - astro_data:/data
    mem_limit: {mem}
    networks:
      - {network}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"]
      interval: 30s
      timeout: 5s
      retries: 3
    labels:
      - "traefik.enable=true"
      - "traefik.docker.network={network}"
      - "traefik.http.routers.{router}.rule=Host(`{host}`)"
      - "traefik.http.routers.{router}.entrypoints=websecure"
      - "traefik.http.routers.{router}.tls.certresolver={resolver}"
      - "traefik.http.services.{router}.loadbalancer.server.port=8000"
volumes:
  astro_data:
networks:
  {network}:
    external: true
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--router", default="astro-engine")
    ap.add_argument("--network", default="n8n_default")
    ap.add_argument("--resolver", default="mytlschallenge")
    ap.add_argument("--mem", default="2g")
    a = ap.parse_args()
    print(compose(a.host, a.router, a.network, a.resolver, a.mem), end="")
