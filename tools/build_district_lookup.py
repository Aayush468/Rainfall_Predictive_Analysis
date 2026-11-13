from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from src.config import ROOT
from src.database.session import make_engine

DISTRICT_API = "https://api.openadmindata.org/api/v1/in/district.json"


def main() -> None:
    response = requests.get(DISTRICT_API, timeout=(10, 45))
    response.raise_for_status()
    payload = response.json()
    entities = payload.get("entities") if isinstance(payload, dict) else None
    if not isinstance(entities, list) or not entities:
        raise SystemExit("District API response did not contain an entities list")

    districts = pd.DataFrame(entities)
    required = {"name_en", "lat", "lon", "parent_name_en"}
    if not required.issubset(districts.columns):
        raise SystemExit(f"District API is missing fields: {sorted(required - set(districts.columns))}")
    districts = districts.dropna(subset=["name_en", "lat", "lon", "parent_name_en"]).copy()
    if districts.empty:
        raise SystemExit("District API contains no records with names and coordinates")

    engine = make_engine()
    with engine.connect() as connection:
        rows = connection.execute(text("""
            SELECT grid_latitude, grid_longitude
            FROM rainfall_grid_observations
            WHERE rainfall_mm IS NOT NULL
            GROUP BY grid_latitude, grid_longitude
            ORDER BY COUNT(DISTINCT observation_date) DESC,
                     md5(grid_latitude::text || chr(44) || grid_longitude::text)
            LIMIT 200
        """)).all()
    engine.dispose()
    if not rows:
        raise SystemExit("No historical grid coordinates found in the database")
    grid_points = {
        (round(float(row[0]), 5), round(float(row[1]), 5)) for row in rows
    }

    district_lat = np.radians(pd.to_numeric(districts.lat).to_numpy())
    district_lon = np.radians(pd.to_numeric(districts.lon).to_numpy())
    rows = []
    for latitude, longitude in grid_points:
        lat = np.radians(latitude)
        lon = np.radians(longitude)
        dlat = district_lat - lat
        dlon = district_lon - lon
        haversine = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(district_lat) * np.sin(dlon / 2) ** 2
        index = int(np.argmin(haversine))
        nearest = districts.iloc[index]
        distance_km = 6371.0088 * 2 * np.arcsin(np.sqrt(haversine[index]))
        rows.append({
            "latitude": f"{latitude:.5f}", "longitude": f"{longitude:.5f}",
            "district_name": str(nearest.name_en), "state_name": str(nearest.parent_name_en),
            "distance_to_district_centroid_km": round(float(distance_km), 1),
        })

    geography_dir = ROOT / "data" / "geography"
    geography_dir.mkdir(parents=True, exist_ok=True)
    (geography_dir / "india_districts.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lookup_path = geography_dir / "grid_district_names.csv"
    pd.DataFrame(rows).to_csv(lookup_path, index=False)
    print(f"Loaded {len(districts)} district centroids; matched {len(rows)} grid points.")
    print(f"Saved district source data and coordinate lookup under {geography_dir}.")


if __name__ == "__main__":
    main()
