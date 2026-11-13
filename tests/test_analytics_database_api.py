import json

import pandas as pd
import xarray as xr

from src.analytics.analysis import quality_summary, rainfall_figures
from src.analytics.analysis import imd_daily_rain_category
from src.database.models import Base
from src.ingestion.imd_client import IMDClient
from src.ingestion.netcdf_import import _coord


def test_schema_contains_required_entities_and_grid_separation():
    required = {
        "states", "districts", "rainfall_observations", "weather_observations",
        "rainfall_forecasts", "district_warnings", "ml_predictions",
        "rainfall_grid_observations", "grid_rainfall_forecasts",
        "grid_ml_predictions", "ingestion_runs",
    }
    assert required.issubset(Base.metadata.tables)
    assert "district_id" not in Base.metadata.tables["rainfall_grid_observations"].columns


def test_rainfall_analytics_and_quality_summary():
    frame = pd.DataFrame({
        "state_name": ["Kerala", "Kerala", "Delhi"],
        "district_name": ["A", "A", "B"],
        "observation_date": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-02"]),
        "actual_rainfall_mm": [10.0, 20.0, 0.0],
        "departure_percentage": [5.0, None, -100.0],
        "rainfall_category": ["N", "E", "NR"],
    })
    figures = rainfall_figures(frame)
    assert {"state_average", "district_highest", "district_lowest", "monthly", "seasonal", "daily", "departure", "categories"} == set(figures)
    quality = quality_summary(frame)
    assert quality["rows"] == 3
    assert quality["missing_cells"] == 1


def test_daily_rainfall_categories_follow_imd_amount_bands():
    assert imd_daily_rain_category(0) == "No measurable rain"
    assert imd_daily_rain_category(2.4) == "Very light"
    assert imd_daily_rain_category(15.5) == "Light"
    assert imd_daily_rain_category(64.5) == "Heavy"
    assert imd_daily_rain_category(204.5) == "Extremely heavy"


def test_netcdf_coordinate_discovery_uses_axis_metadata():
    ds = xr.Dataset(coords={"when": ("when", [0], {"axis": "T"}),
                            "northing": ("northing", [10.0], {"axis": "Y"}),
                            "easting": ("easting", [75.0], {"axis": "X"})})
    assert _coord(ds, ("time", "date"), "T") == "when"
    assert _coord(ds, ("lat", "latitude"), "Y") == "northing"
    assert _coord(ds, ("lon", "longitude"), "X") == "easting"


def test_imd_client_persists_raw_payload(tmp_path):
    client = IMDClient(raw_dir=tmp_path)

    class Response:
        status_code = 200
        url = "https://api.imd.gov.in/api/v1/districtrainfall"
        text = "[]"

        @staticmethod
        def json():
            return [{"District": "Example"}]

        @staticmethod
        def raise_for_status():
            return None

    client.session.get = lambda *args, **kwargs: Response()
    payload, status, raw_path = client.fetch("district_rainfall")
    assert status == 200
    assert payload[0]["District"] == "Example"
    assert json.loads(raw_path.read_text(encoding="utf-8"))["payload"] == payload
