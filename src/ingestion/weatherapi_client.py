from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from ..config import ROOT, settings

BASE_URL = "https://api.weatherapi.com/v1/forecast.json"
CURRENT_URL = "https://api.weatherapi.com/v1/current.json"
SEARCH_URL = "https://api.weatherapi.com/v1/search.json"


class WeatherAPIError(RuntimeError):
    pass


class WeatherAPIClient:
    """Fetch current conditions and a short forecast without exposing the API key."""

    def __init__(self, api_key: str | None = None, raw_dir: Path | None = None,
                 session: requests.Session | None = None):
        self.api_key = (api_key if api_key is not None else settings.weatherapi_key).strip()
        self.raw_dir = raw_dir or ROOT / "data" / "raw" / "weatherapi"
        self.session = session or requests.Session()

    def fetch_forecast(self, location: str, days: int = 3) -> tuple[dict, Path]:
        location = location.strip()
        if not self.api_key:
            raise WeatherAPIError("Set WEATHERAPI_KEY in the local .env file first")
        if not location:
            raise WeatherAPIError("Enter a city, place, or latitude,longitude")
        if not 1 <= days <= 3:
            raise ValueError("The configured weather plan provides up to 3 forecast days")
        try:
            response = self.session.get(BASE_URL, params={
                "key": self.api_key,
                "q": location,
                "days": days,
                "aqi": "no",
                "alerts": "no",
            }, timeout=(10, 30))
        except requests.RequestException as exc:
            raise WeatherAPIError(_connection_message(exc)) from None

        try:
            payload = response.json()
        except ValueError as exc:
            raise WeatherAPIError("The weather service returned an unreadable response") from exc
        if not isinstance(payload, dict):
            raise WeatherAPIError("The weather service returned an unexpected response format")
        if response.status_code >= 400 or "error" in payload:
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = error.get("message", "Request failed") if isinstance(error, dict) else "Request failed"
            message = str(message).replace(self.api_key, "[redacted]")
            if response.status_code in (401, 403):
                raise WeatherAPIError("The weather service rejected its key or plan. Check the local credentials and account access.")
            raise WeatherAPIError(f"Weather request failed (HTTP {response.status_code}): {message}")
        if not isinstance(payload, dict) or not payload.get("current") or not payload.get("forecast", {}).get("forecastday"):
            raise WeatherAPIError("The weather response is missing current conditions or forecast days")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        fetched_at = datetime.now(timezone.utc)
        timestamp = fetched_at.strftime("%Y%m%dT%H%M%S%fZ")
        raw_path = self.raw_dir / f"{timestamp}_forecast.json"
        # Keep the original location query and payload, never the request URL/key.
        raw_path.write_text(json.dumps({
            "source": "Live Weather Service",
            "endpoint": "/v1/forecast.json",
            "location_query": location,
            "fetched_at_utc": timestamp,
            "payload": payload,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload, raw_path

    def _get_json(self, url: str, location: str, **params):
        try:
            response = self.session.get(url, params={"key": self.api_key, "q": location, **params}, timeout=(10, 30))
            payload = response.json()
        except requests.RequestException as exc:
            raise WeatherAPIError(_connection_message(exc)) from None
        except ValueError as exc:
            raise WeatherAPIError("The weather service returned an unreadable response") from exc
        if not isinstance(payload, (dict, list)):
            raise WeatherAPIError("The weather service returned an unexpected response format")
        if response.status_code >= 400 or (isinstance(payload, dict) and "error" in payload):
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = error.get("message", "Request failed") if isinstance(error, dict) else "Request failed"
            message = str(message).replace(self.api_key, "[redacted]")
            if response.status_code in (401, 403):
                raise WeatherAPIError("The weather service rejected its key or plan. Check the local credentials and account access.")
            raise WeatherAPIError(f"Weather request failed (HTTP {response.status_code}): {message}")
        return payload

    def search_locations(self, query: str) -> list[dict]:
        query = query.strip()
        if not self.api_key:
            raise WeatherAPIError("Set WEATHERAPI_KEY in the local .env file first")
        if len(query) < 2:
            raise WeatherAPIError("Enter at least two characters to search for a place")
        results = self._get_json(SEARCH_URL, query)
        return results if isinstance(results, list) else []

    def fetch_location_weather(self, location: str, days: int = 3) -> tuple[dict, Path]:
        """Fetch the current endpoint and forecast endpoint for a selected place."""
        location = location.strip()
        if not self.api_key:
            raise WeatherAPIError("Set WEATHERAPI_KEY in the local .env file first")
        if not location:
            raise WeatherAPIError("Choose a location before loading its weather")
        if not 1 <= days <= 3:
            raise ValueError("The configured weather plan provides up to 3 forecast days")
        current_payload = self._get_json(CURRENT_URL, location, aqi="no")
        forecast_payload = self._get_json(BASE_URL, location, days=days, aqi="no", alerts="no")
        current = current_payload.get("current")
        forecast = forecast_payload.get("forecast", {}).get("forecastday", [])
        if not current or not forecast:
            raise WeatherAPIError("The weather response is missing current conditions or forecast days")
        forecast_payload["current"] = current

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        raw_path = self.raw_dir / f"{timestamp}_weather.json"
        raw_path.write_text(json.dumps({
            "source": "Live Weather Service",
            "endpoints": ["/v1/current.json", "/v1/forecast.json"],
            "location_query": location,
            "fetched_at_utc": timestamp,
            "current": current_payload,
            "forecast": forecast_payload,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return forecast_payload, raw_path


def _connection_message(exc: requests.RequestException) -> str:
    """Return actionable transport guidance without echoing the credential URL."""
    if isinstance(exc, requests.Timeout):
        return "The live weather service did not respond before the connection timed out. Check internet access and try again."
    if isinstance(exc, requests.ConnectionError):
        return "This app could not reach the live weather service. Check DNS, firewall, proxy, and internet access; a model forecast fallback is available."
    return "The live weather request could not be completed. Check internet access and try again."
