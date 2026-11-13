import json

import pandas as pd
import pytest

from src.analytics.weather_summary import (
    weatherapi_daily_frame,
    weatherapi_hourly_frame,
    weatherapi_narrative,
    weatherapi_next_hours_frame,
    weatherapi_nowcast_narrative,
)
from src.ingestion.weatherapi_client import WeatherAPIClient, WeatherAPIError
from src.ingestion.open_meteo_location import OpenMeteoLocationClient
from src.dashboard.radar import radar_map_html


PAYLOAD = {
    "location": {"name": "Pune", "region": "Maharashtra", "country": "India", "lat": 18.5, "lon": 73.8},
    "current": {
        "last_updated": "2026-11-13 10:00", "temp_c": 24.0, "feelslike_c": 25.0,
        "humidity": 60, "wind_kph": 8.0, "precip_mm": 0.0,
        "pressure_mb": 1010.0, "vis_km": 10.0, "uv": 5.0,
        "condition": {"text": "Partly cloudy"},
    },
    "forecast": {"forecastday": [
        {"date": "2026-11-13", "day": {"totalprecip_mm": 0.0, "daily_chance_of_rain": 10,
          "maxtemp_c": 30.0, "mintemp_c": 20.0, "maxwind_kph": 12.0, "condition": {"text": "Sunny"}},
         "hour": [{"time": "2026-11-13 12:00", "temp_c": 29.0, "precip_mm": 0.0,
                   "chance_of_rain": 10, "condition": {"text": "Sunny"}}]},
        {"date": "2026-11-14", "day": {"totalprecip_mm": 12.5, "daily_chance_of_rain": 70,
          "maxtemp_c": 27.0, "mintemp_c": 19.0, "maxwind_kph": 14.0, "condition": {"text": "Light rain"}},
         "hour": []},
    ]},
}


def test_weatherapi_summary_and_forecast_frames():
    daily = weatherapi_daily_frame(PAYLOAD)
    assert len(daily) == 2
    assert daily.iloc[1].rainfall_band == "Light"
    assert "rain chance is at least 50% on sat 14 nov" in weatherapi_narrative(PAYLOAD).lower()
    hourly = weatherapi_hourly_frame(PAYLOAD, pd.Timestamp("2026-11-13").date())
    assert len(hourly) == 1
    assert hourly.iloc[0].temperature_c == 29.0
    next_hours = weatherapi_next_hours_frame(PAYLOAD)
    assert len(next_hours) == 1
    assert "No measurable rain" in weatherapi_nowcast_narrative(PAYLOAD, next_hours)


def test_weatherapi_requires_key_and_location(tmp_path):
    with pytest.raises(WeatherAPIError, match="WEATHERAPI_KEY"):
        WeatherAPIClient(api_key="", raw_dir=tmp_path).fetch_forecast("Pune")
    with pytest.raises(WeatherAPIError, match="Enter a city"):
        WeatherAPIClient(api_key="test-key", raw_dir=tmp_path).fetch_forecast(" ")


def test_weatherapi_client_saves_payload_without_saving_key(tmp_path):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return PAYLOAD

    class Session:
        def get(self, url, params, timeout):
            assert url.endswith("/forecast.json")
            assert params["key"] == "private-test-key"
            assert params["days"] == 3
            return Response()

    payload, raw_path = WeatherAPIClient(
        api_key="private-test-key", raw_dir=tmp_path, session=Session()
    ).fetch_forecast("Pune, India")
    assert payload["location"]["name"] == "Pune"
    saved = raw_path.read_text(encoding="utf-8")
    assert "private-test-key" not in saved
    assert json.loads(saved)["location_query"] == "Pune, India"


def test_open_meteo_city_client_normalizes_and_saves_forecast(tmp_path):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Session:
        def get(self, url, params, timeout):
            if "geocoding" in url:
                return Response({"results": [{"name": "Pune", "admin1": "Maharashtra", "country": "India",
                                               "latitude": 18.5, "longitude": 73.8, "timezone": "Asia/Kolkata"}]})
            return Response({
                "current": {"time": "2026-11-13T10:00", "temperature_2m": 24, "relative_humidity_2m": 60,
                            "apparent_temperature": 25, "precipitation": 0, "weather_code": 2,
                            "wind_speed_10m": 8, "wind_direction_10m": 180, "pressure_msl": 1010},
                "hourly": {"time": ["2026-11-13T10:00"], "temperature_2m": [24], "precipitation": [0],
                           "precipitation_probability": [10], "weather_code": [2]},
                "daily": {"time": ["2026-11-13"], "precipitation_sum": [1.2],
                          "precipitation_probability_max": [30], "temperature_2m_max": [30],
                          "temperature_2m_min": [20], "wind_speed_10m_max": [12], "weather_code": [61]},
            })

    payload, raw_path = OpenMeteoLocationClient(session=Session(), raw_dir=tmp_path).fetch("Pune")
    assert payload["location"]["name"] == "Pune"
    assert payload["current"]["temp_c"] == 24
    daily = weatherapi_daily_frame(payload)
    assert daily.iloc[0].forecast_rain_mm == 1.2
    assert daily.iloc[0].condition == "Light rain"
    assert len(weatherapi_hourly_frame(payload, pd.Timestamp("2026-11-13").date())) == 1
    assert json.loads(raw_path.read_text(encoding="utf-8"))["source"].startswith("Open-Meteo")


def test_weatherapi_search_and_separate_current_forecast_endpoints(tmp_path):
    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, params, timeout):
            self.calls.append((url, params.copy()))
            if url.endswith("search.json"):
                return Response([{"id": 123, "name": "Pune", "region": "Maharashtra", "country": "India",
                                  "lat": 18.5, "lon": 73.8}])
            if url.endswith("current.json"):
                return Response({"location": PAYLOAD["location"], "current": PAYLOAD["current"]})
            return Response(PAYLOAD)

    session = Session()
    client = WeatherAPIClient(api_key="private-test-key", raw_dir=tmp_path, session=session)
    matches = client.search_locations("Pune")
    selected_location = f"{matches[0]['lat']},{matches[0]['lon']}"
    payload, raw_path = client.fetch_location_weather(selected_location)
    assert [call[0].rsplit("/", 1)[-1] for call in session.calls] == ["search.json", "current.json", "forecast.json"]
    assert all(call[1]["key"] == "private-test-key" for call in session.calls)
    assert session.calls[1][1]["q"] == "18.5,73.8"
    assert payload["current"]["temp_c"] == PAYLOAD["current"]["temp_c"]
    saved = raw_path.read_text(encoding="utf-8")
    assert "private-test-key" not in saved
    assert json.loads(saved)["endpoints"] == ["/v1/current.json", "/v1/forecast.json"]


def test_radar_map_uses_selected_coordinates_and_rejects_invalid_values():
    page = radar_map_html(18.52, 73.85)
    assert "api.rainviewer.com/public/weather-maps.json" in page
    assert "recent observed echoes" in page
    assert "18.520000" in page and "73.850000" in page
    with pytest.raises(ValueError, match="out of range"):
        radar_map_html(95, 0)
