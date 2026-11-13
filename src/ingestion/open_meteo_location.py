from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from ..config import ROOT

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CONDITIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Heavy freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain",
    67: "Heavy freezing rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Light rain showers", 81: "Rain showers",
    82: "Heavy rain showers", 85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with light hail", 99: "Thunderstorm with heavy hail",
}


def _weather_text(code) -> str:
    try:
        return WMO_CONDITIONS.get(int(code), "Variable conditions")
    except (TypeError, ValueError):
        return "Conditions unavailable"


def _at(values, index: int, default=None):
    if not isinstance(values, list) or index >= len(values):
        return default
    return values[index]


class OpenMeteoLocationClient:
    """Keyless city current-condition estimate and 3-day forecast."""

    def __init__(self, session: requests.Session | None = None, raw_dir: Path | None = None):
        self.session = session or requests.Session()
        self.raw_dir = raw_dir or ROOT / "data" / "raw" / "open_meteo_location"

    def fetch(self, query: str, days: int = 3) -> tuple[dict, Path]:
        query = query.strip()
        if len(query) < 2:
            raise ValueError("Enter at least two characters for the location")
        if not 1 <= days <= 7:
            raise ValueError("days must be between 1 and 7")
        try:
            geo_response = self.session.get(GEOCODING_URL, params={
                "name": query, "count": 5, "language": "en", "format": "json",
            }, timeout=(10, 25))
            geo_response.raise_for_status()
            geocoding = geo_response.json()
            results = geocoding.get("results", [])
            if not results:
                raise ValueError(f"No matching location found for {query!r}")
            place = results[0]
            timezone_name = place.get("timezone", "Asia/Kolkata")
            forecast_response = self.session.get(FORECAST_URL, params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,dew_point_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m,pressure_msl",
                "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation_probability,precipitation,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
                "forecast_days": days,
                "timezone": timezone_name,
            }, timeout=(10, 30))
            forecast_response.raise_for_status()
            forecast = forecast_response.json()
        except requests.RequestException as exc:
            raise RuntimeError("Could not connect to the keyless weather service. Check the network and try again.") from exc
        if not isinstance(forecast, dict) or not forecast.get("current") or not forecast.get("daily"):
            raise RuntimeError("Open-Meteo returned an incomplete location forecast")

        current = forecast["current"]
        hourly = forecast.get("hourly", {})
        daily = forecast["daily"]
        hourly_rows = []
        for index, time_value in enumerate(hourly.get("time", [])):
            hourly_rows.append({
                "time": time_value,
                "temp_c": _at(hourly.get("temperature_2m"), index),
                "precip_mm": _at(hourly.get("precipitation"), index),
                "chance_of_rain": _at(hourly.get("precipitation_probability"), index),
                "humidity": _at(hourly.get("relative_humidity_2m"), index),
                "dewpoint_c": _at(hourly.get("dew_point_2m"), index),
                "condition": {"text": _weather_text(_at(hourly.get("weather_code"), index))},
            })
        forecast_days = []
        for index, day in enumerate(daily.get("time", [])):
            forecast_days.append({
                "date": day,
                "day": {
                    "totalprecip_mm": _at(daily.get("precipitation_sum"), index, 0),
                    "daily_chance_of_rain": _at(daily.get("precipitation_probability_max"), index, 0),
                    "maxtemp_c": _at(daily.get("temperature_2m_max"), index),
                    "mintemp_c": _at(daily.get("temperature_2m_min"), index),
                    "maxwind_kph": _at(daily.get("wind_speed_10m_max"), index),
                    "condition": {"text": _weather_text(_at(daily.get("weather_code"), index))},
                },
                "hour": [hour for hour in hourly_rows if hour["time"].startswith(day)],
            })

        normalized = {
            "location": {
                "name": place.get("name", query), "region": place.get("admin1", ""),
                "country": place.get("country", ""), "lat": place["latitude"],
                "lon": place["longitude"], "localtime": current.get("time", ""),
            },
            "current": {
                "last_updated": current.get("time", ""),
                "temp_c": current.get("temperature_2m"),
                "feelslike_c": current.get("apparent_temperature"),
                "humidity": current.get("relative_humidity_2m"),
                "dewpoint_c": current.get("dew_point_2m"),
                "wind_kph": current.get("wind_speed_10m"),
                "wind_degree": current.get("wind_direction_10m"),
                "precip_mm": current.get("precipitation"),
                "pressure_mb": current.get("pressure_msl"),
                "condition": {"text": _weather_text(current.get("weather_code"))},
            },
            "forecast": {"forecastday": forecast_days},
        }
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        raw_path = self.raw_dir / f"{timestamp}_forecast.json"
        raw_path.write_text(json.dumps({
            "source": "Open-Meteo model current conditions and forecast",
            "location_query": query,
            "geocoding": geocoding,
            "forecast": forecast,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return normalized, raw_path
