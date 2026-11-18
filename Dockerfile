FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data PORT=8000
WORKDIR /app

# build-essential only needed if a wheel is missing for the platform (ephem ships wheels for x86_64/arm64)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY astro_engine ./astro_engine
COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
