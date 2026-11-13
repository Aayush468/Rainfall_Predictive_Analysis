from __future__ import annotations

import argparse
import logging
from sqlalchemy import text

from ..database.session import create_schema, make_engine
from ..logging_config import configure_logging
from .imd_client import ENDPOINTS, IMDClient
from .netcdf_import import import_netcdf
from .loaders import (load_current_weather, load_district_rainfall, load_forecasts,
                      load_state_rainfall, load_warnings)
from .transform import (transform_district_forecast, transform_district_rainfall,
                        transform_district_warnings, transform_state_rainfall,
                        transform_current_weather)

logger = logging.getLogger(__name__)


def run_api(endpoints: list[str]) -> int:
    client = IMDClient()
    engine = make_engine()
    create_schema(engine)
    successes = 0
    for name in endpoints:
        status, count, error = None, 0, None
        try:
            payload, status, raw_path = client.fetch(name)
            if name == "district_rainfall":
                frame = transform_district_rainfall(payload)
                count, rejected = load_district_rainfall(frame)
                if rejected:
                    error = f"Rejected {rejected} rows: no unambiguous existing district mapping"
            elif name == "state_rainfall":
                count = load_state_rainfall(transform_state_rainfall(payload))
            elif name == "district_forecast":
                count, rejected = load_forecasts(transform_district_forecast(payload))
                if rejected:
                    error = f"Rejected {rejected} forecast rows without a valid district/date"
            elif name == "district_warnings":
                count, rejected = load_warnings(transform_district_warnings(payload))
                if rejected:
                    error = f"Rejected {rejected} warning rows without a valid district/date"
            elif name == "current_weather":
                count = load_current_weather(transform_current_weather(payload))
            else:
                from .transform import records_from_payload
                count = len(records_from_payload(payload))
            successes += 1
            logger.info("Fetched %s: %d records in raw payload (%s)", name, count, raw_path)
        except Exception as exc:  # Endpoint failures must not halt other endpoints.
            status = getattr(exc, "status_code", None)
            if status is None and getattr(exc, "response", None) is not None:
                status = exc.response.status_code
            error = str(exc)
            logger.exception("IMD endpoint %s failed", name)
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO ingestion_runs(source, endpoint, status_code, records_received, error_message)
                VALUES (:source, :endpoint, :status, :count, :error)
            """), {"source": "IMD API", "endpoint": name, "status": status,
                  "count": count, "error": error})
    return successes


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest official IMD rainfall information")
    parser.add_argument("--endpoint", choices=sorted(ENDPOINTS), action="append",
                        help="Documented endpoint to fetch; repeat for multiple. Defaults to all.")
    parser.add_argument("--netcdf", help="Import a downloaded IMD gridded rainfall NetCDF file")
    parser.add_argument("--variable", help="Rainfall data variable name in NetCDF")
    parser.add_argument("--max-cells", type=int,
                        help="Import a deterministic spatial sample of cells from NetCDF instead of all cells")
    parser.add_argument("--open-meteo-forecast", action="store_true",
                        help="Fetch a keyless 7-day Open-Meteo forecast for existing grid cells")
    args = parser.parse_args()
    configure_logging()
    if args.max_cells is not None and args.max_cells < 1:
        parser.error("--max-cells must be at least 1")
    if args.netcdf:
        inserted = import_netcdf(args.netcdf, args.variable, max_cells=args.max_cells)
        print(f"Loaded {inserted} grid observations")
        return
    if args.open_meteo_forecast:
        from .open_meteo import ingest_forecast
        count = ingest_forecast(args.max_cells or 200)
        print(f"Loaded {count} Open-Meteo forecast rows")
        return
    endpoints = args.endpoint or list(ENDPOINTS)
    succeeded = run_api(endpoints)
    print(f"IMD API endpoints succeeded: {succeeded}/{len(endpoints)}. See logs/rainfall.log and data/raw/imd/.")


if __name__ == "__main__":
    main()
