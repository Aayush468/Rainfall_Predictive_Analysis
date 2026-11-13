from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from sqlalchemy import text

from ..config import ROOT
from ..database.session import create_schema, make_engine

logger = logging.getLogger(__name__)
API_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE = "Open-Meteo forecast (best_match)"


def _locations(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Unexpected Open-Meteo response format")


def ingest_forecast(max_cells: int = 200, session: requests.Session | None = None) -> int:
    """Fetch seven-day forecast for a deterministic sample of observed IMD grid cells."""
    if max_cells < 1:
        raise ValueError("max_cells must be at least 1")
    engine = make_engine()
    create_schema(engine)
    with engine.connect() as connection:
        cells = connection.execute(text("""
            SELECT grid_latitude, grid_longitude
            FROM rainfall_grid_observations
            WHERE rainfall_mm IS NOT NULL
            GROUP BY grid_latitude, grid_longitude
            ORDER BY COUNT(DISTINCT observation_date) DESC,
                     md5(grid_latitude::text || ',' || grid_longitude::text)
            LIMIT :limit
        """), {"limit": max_cells}).mappings().all()
    if not cells:
        raise RuntimeError("No gridded observation coordinates found; import IMD NetCDF data first")

    raw_dir = ROOT / "data" / "raw" / "open_meteo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    issued_at = datetime.now(timezone.utc)
    stamp = issued_at.strftime("%Y%m%dT%H%M%SZ")
    client = session or requests.Session()
    rows: list[dict] = []
    # Multi-location requests keep refreshes small; free service is rate-limited.
    for start in range(0, len(cells), 50):
        batch = cells[start:start + 50]
        response = client.get(API_URL, params={
            "latitude": ",".join(str(cell["grid_latitude"]) for cell in batch),
            "longitude": ",".join(str(cell["grid_longitude"]) for cell in batch),
            "daily": "precipitation_sum,precipitation_probability_max",
            "forecast_days": 7,
            "timezone": "Asia/Kolkata",
        }, timeout=(10, 45))
        response.raise_for_status()
        payload = response.json()
        raw_path = raw_dir / f"{stamp}_forecast_batch_{start // 50 + 1}.json"
        raw_path.write_text(json.dumps({
            "source": SOURCE, "fetched_at_utc": stamp, "payload": payload,
        }, ensure_ascii=False), encoding="utf-8")
        locations = _locations(payload)
        if len(locations) != len(batch):
            raise ValueError(f"Open-Meteo returned {len(locations)} locations for {len(batch)} requested")
        for cell, location in zip(batch, locations):
            daily = location.get("daily") or {}
            dates = daily.get("time") or []
            amounts = daily.get("precipitation_sum") or []
            probabilities = daily.get("precipitation_probability_max") or [None] * len(dates)
            if len(amounts) != len(dates):
                raise ValueError("Open-Meteo daily date and precipitation arrays have different lengths")
            for index, date_value in enumerate(dates):
                amount = amounts[index]
                probability = probabilities[index] if index < len(probabilities) else None
                rows.append({
                    "lat": cell["grid_latitude"], "lon": cell["grid_longitude"],
                    "date": date_value, "rainfall": amount, "probability": probability,
                    "issued_at": issued_at, "source": SOURCE,
                })

    with engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO grid_rainfall_forecasts
                (grid_latitude, grid_longitude, forecast_date, rainfall_mm,
                 precipitation_probability_pct, issued_at, data_source)
            VALUES (:lat, :lon, CAST(:date AS DATE), :rainfall, :probability, :issued_at, :source)
            ON CONFLICT (grid_latitude, grid_longitude, forecast_date, data_source)
            DO UPDATE SET rainfall_mm = EXCLUDED.rainfall_mm,
                precipitation_probability_pct = EXCLUDED.precipitation_probability_pct,
                issued_at = EXCLUDED.issued_at
        """), rows)
        connection.execute(text("""
            INSERT INTO ingestion_runs(source, endpoint, status_code, records_received, error_message)
            VALUES (:source, '7-day-grid-forecast', 200, :count, NULL)
        """), {"source": SOURCE, "count": len(rows)})
    logger.info("Stored %d Open-Meteo grid forecasts for %d cells", len(rows), len(cells))
    return len(rows)
