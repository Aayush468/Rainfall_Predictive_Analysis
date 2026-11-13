from __future__ import annotations

import pandas as pd

from .analysis import imd_daily_rain_category


def weatherapi_daily_frame(payload: dict) -> pd.DataFrame:
    location = payload.get("location", {}).get("name", "Selected location")
    rows = []
    for forecast in payload.get("forecast", {}).get("forecastday", []):
        day = forecast.get("day", {})
        amount = float(day.get("totalprecip_mm") or 0.0)
        rows.append({
            "date": pd.to_datetime(forecast.get("date")).date(),
            "location": location,
            "forecast_rain_mm": amount,
            "rain_chance_pct": float(day.get("daily_chance_of_rain") or 0),
            "temperature_high_c": day.get("maxtemp_c"),
            "temperature_low_c": day.get("mintemp_c"),
            "wind_max_kph": day.get("maxwind_kph"),
            "condition": day.get("condition", {}).get("text", "Unknown"),
            "rainfall_band": imd_daily_rain_category(amount),
        })
    return pd.DataFrame(rows)


def weatherapi_hourly_frame(payload: dict, selected_date) -> pd.DataFrame:
    rows = []
    for forecast in payload.get("forecast", {}).get("forecastday", []):
        if pd.to_datetime(forecast.get("date")).date() != selected_date:
            continue
        for hour in forecast.get("hour", []):
            rows.append({
                "time": pd.to_datetime(hour.get("time")),
                "temperature_c": hour.get("temp_c"),
                "precipitation_mm": hour.get("precip_mm"),
                "rain_chance_pct": hour.get("chance_of_rain"),
                "condition": hour.get("condition", {}).get("text", "Unknown"),
            })
    return pd.DataFrame(rows)


def weatherapi_next_hours_frame(payload: dict, count: int = 3) -> pd.DataFrame:
    """Return the next hourly bins using the provider's local timestamps."""
    rows = []
    for forecast in payload.get("forecast", {}).get("forecastday", []):
        for hour in forecast.get("hour", []):
            rows.append({
                "time": pd.to_datetime(hour.get("time"), errors="coerce"),
                "precipitation_mm": pd.to_numeric(hour.get("precip_mm"), errors="coerce"),
                "rain_chance_pct": pd.to_numeric(hour.get("chance_of_rain"), errors="coerce"),
                "temperature_c": pd.to_numeric(hour.get("temp_c"), errors="coerce"),
                "humidity_pct": pd.to_numeric(hour.get("humidity"), errors="coerce"),
                "dewpoint_c": pd.to_numeric(hour.get("dewpoint_c"), errors="coerce"),
                "condition": hour.get("condition", {}).get("text", "Unknown"),
            })
    frame = pd.DataFrame(rows).dropna(subset=["time"]).sort_values("time")
    if frame.empty:
        return frame
    current_time = pd.to_datetime(payload.get("current", {}).get("last_updated"), errors="coerce")
    if not pd.isna(current_time):
        future = frame[frame.time > current_time]
        if not future.empty:
            frame = future
    return frame.head(count).reset_index(drop=True)


def weatherapi_nowcast_narrative(payload: dict, next_hours: pd.DataFrame | None = None) -> str:
    location = payload.get("location", {}).get("name", "your area")
    current = payload.get("current", {})
    now_rain = float(current.get("precip_mm") or 0)
    hours = next_hours if next_hours is not None else weatherapi_next_hours_frame(payload)
    if hours.empty:
        return f"Current conditions in {location}: {current.get('condition', {}).get('text', 'unavailable')}. No near-term hourly forecast was returned."
    peak = hours.loc[hours.precipitation_mm.fillna(0).idxmax()]
    chance = 0 if pd.isna(peak.rain_chance_pct) else int(peak.rain_chance_pct)
    amount = 0 if pd.isna(peak.precipitation_mm) else float(peak.precipitation_mm)
    time_label = pd.Timestamp(peak.time).strftime("%I %p").lstrip("0")
    if now_rain >= 0.1:
        return f"Rain is being reported now in {location}. The next {len(hours)} hours show the heaviest shower around {time_label}: {amount:.1f} mm in that hour, with a {chance}% rain chance."
    if amount >= 2.5:
        intensity = "Heavy rain"
    elif amount >= 0.1:
        intensity = "Rain"
    else:
        intensity = "A shower is possible" if chance >= 50 else "No measurable rain is indicated"
    if amount < 0.1 and chance < 50:
        return f"{intensity} in {location} over the next {len(hours)} hours. Highest hourly rain chance: {chance}%."
    return f"{intensity} is possible around {time_label} in {location}: about {amount:.1f} mm in that hour, with a {chance}% rain chance."


def weatherapi_narrative(payload: dict) -> str:
    location = payload.get("location", {}).get("name", "Selected location")
    current = payload.get("current", {})
    condition = current.get("condition", {}).get("text", "conditions unavailable")
    temperature = current.get("temp_c")
    feels = current.get("feelslike_c")
    humidity = current.get("humidity")
    wind = current.get("wind_kph")
    current_sentence = (
        f"Current conditions in {location}: {condition.lower()}, {temperature}°C"
        + (f" (feels like {feels}°C)" if feels is not None else "")
        + (f", humidity {humidity}%" if humidity is not None else "")
        + (f", wind {wind} km/h." if wind is not None else ".")
    )

    daily = weatherapi_daily_frame(payload)
    if daily.empty:
        return current_sentence + " A short-range forecast was not returned."
    likely_rain = daily[daily.rain_chance_pct >= 50]
    peak = daily.loc[daily.forecast_rain_mm.idxmax()]
    peak_day = pd.Timestamp(peak.date).strftime("%A, %d %b")
    if likely_rain.empty:
        outlook = "No forecast day reaches a 50% chance of rain"
    else:
        likely_dates = ", ".join(pd.Timestamp(day).strftime("%a %d %b") for day in likely_rain.date)
        outlook = f"Rain chance is at least 50% on {likely_dates}"
    return (
        f"{current_sentence} For the next {len(daily)} days, {outlook}. "
        f"The highest daily amount is {peak.forecast_rain_mm:.1f} mm on {peak_day} "
        f"({peak.rain_chance_pct:.0f}% rain chance)."
    )
