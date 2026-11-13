from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ..config import ROOT
from ..database.session import make_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Export official IMD grid observations for model training")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "grid_observations.csv")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--max-cells", type=int, default=200,
                        help="Deterministic spatial sample size; set to a larger value for wider coverage")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    query = text("""
        WITH eligible AS (
            SELECT grid_latitude, grid_longitude, observation_date, rainfall_mm
            FROM rainfall_grid_observations
            WHERE rainfall_mm IS NOT NULL
              AND (CAST(:start_date AS DATE) IS NULL OR observation_date >= CAST(:start_date AS DATE))
              AND (CAST(:end_date AS DATE) IS NULL OR observation_date <= CAST(:end_date AS DATE))
        ), selected_cells AS (
            SELECT grid_latitude, grid_longitude
            FROM eligible
            GROUP BY grid_latitude, grid_longitude
            ORDER BY COUNT(DISTINCT observation_date) DESC,
                     md5(grid_latitude::text || ',' || grid_longitude::text)
            LIMIT :max_cells
        )
        SELECT e.grid_latitude, e.grid_longitude, e.observation_date, e.rainfall_mm
        FROM eligible e JOIN selected_cells c USING (grid_latitude, grid_longitude)
        ORDER BY e.grid_latitude, e.grid_longitude, e.observation_date
    """)
    if args.max_cells < 1:
        parser.error("--max-cells must be at least 1")
    frame = pd.read_sql(query, make_engine(), params={
        "start_date": args.start_date, "end_date": args.end_date,
        "max_cells": args.max_cells,
    })
    if frame.empty:
        raise SystemExit("No gridded observations found. Import official IMD NetCDF files first.")
    frame.to_csv(args.out, index=False)
    print(f"Exported {len(frame):,} official grid observations to {args.out}")


if __name__ == "__main__":
    main()
