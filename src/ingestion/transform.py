from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd


def records_from_payload(payload: Any) -> list[dict]:
    """Accept only common documented JSON array/envelope shapes."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "Data", "result", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if payload and all(not isinstance(value, (dict, list)) for value in payload.values()):
            return [payload]
    return []


def _key(row: dict, *names: str):
    lookup = {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in row.items()}
    for name in names:
        value = lookup.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if value not in (None, "", "NA", "ND"):
            return value
    return None


def _number(value):
    if value is None:
        return None
    try:
        result = float(str(value).strip().replace("%", ""))
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _date(value, formats: tuple[str, ...]) -> date | None:
    if not value:
        return None
    for fmt in formats:
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            pass
    return None


def transform_district_rainfall(payload: Any) -> pd.DataFrame:
    rows = []
    for row in records_from_payload(payload):
        rows.append({
            "imd_district_id": str(_key(row, "OBJ_ID", "Obj_id", "District ID") or "").strip(),
            "district_name": str(_key(row, "District") or "").strip().title(),
            "observation_date": _date(_key(row, "Date"), ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y")),
            "actual_rainfall_mm": _number(_key(row, "Daily Actual")),
            "normal_rainfall_mm": _number(_key(row, "Daily Normal")),
            "departure_percentage": _number(_key(row, "Daily Departure Per")),
            "rainfall_category": _key(row, "Daily Category"),
            "data_source": "IMD API: districtrainfall",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["imd_district_id", "observation_date", "data_source"])
        df = df[df["district_name"].ne("") & df["observation_date"].notna()]
        df.loc[df["actual_rainfall_mm"] < 0, "actual_rainfall_mm"] = None
    return df


def transform_state_rainfall(payload: Any) -> pd.DataFrame:
    rows = []
    for row in records_from_payload(payload):
        rows.append({
            "state_name": str(_key(row, "State") or "").strip().title(),
            "observation_date": _date(_key(row, "Date"), ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y")),
            "actual_rainfall_mm": _number(_key(row, "Daily Actual")),
            "normal_rainfall_mm": _number(_key(row, "Daily Normal")),
            "departure_percentage": _number(_key(row, "Daily Departure Per")),
            "rainfall_category": _key(row, "Daily Category"),
            "data_source": "IMD API: staterainfall",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["state_name"].ne("") & df["observation_date"].notna()].drop_duplicates(["state_name", "observation_date"])
        for column in ("actual_rainfall_mm", "normal_rainfall_mm"):
            df.loc[df[column] < 0, column] = None
    return df


def transform_district_forecast(payload: Any) -> pd.DataFrame:
    result = []
    for row in records_from_payload(payload):
        issue_date = _date(_key(row, "date_obs"), ("%Y-%m-%d", "%d-%m-%Y"))
        for day in range(1, 6):
            distribution = _key(row, f"day{day}_distribution")
            percentage = _key(row, f"day{day}_distribution_percentage")
            if distribution is None and percentage is None:
                continue
            forecast_date = issue_date + pd.Timedelta(days=day) if issue_date else None
            result.append({
                "imd_district_id": _key(row, "Obj_id", "OBJ_ID"),
                "district_name": str(_key(row, "District") or "").strip().title(),
                "state_name": str(_key(row, "State") or "").strip().title(),
                "forecast_date": forecast_date if forecast_date is not None else None,
                "forecast_day": day,
                "rainfall_distribution": distribution,
                "distribution_percentage": percentage,
                "data_source": "IMD API: state_district_rainfall_forecast",
            })
    return pd.DataFrame(result)


def transform_district_warnings(payload: Any) -> pd.DataFrame:
    result = []
    for row in records_from_payload(payload):
        warning_date = _date(_key(row, "Date"), ("%Y-%m-%d", "%d-%m-%Y"))
        for day in range(1, 6):
            description = _key(row, f"Day_{day}")
            level = _key(row, f"Day{day}_Color")
            if description is None and level is None:
                continue
            result.append({
                "imd_district_id": _key(row, "Obj_id", "OBJ_ID"),
                "district_name": str(_key(row, "District") or "").strip().title(),
                "warning_date": warning_date,
                "forecast_day": day,
                "warning_level": level,
                "warning_description": description,
                "data_source": "IMD API: districtwarning",
            })
    return pd.DataFrame(result)


def transform_current_weather(payload: Any) -> pd.DataFrame:
    """Normalize the documented current-weather fields (time is UTC)."""
    result = []
    for row in records_from_payload(payload):
        station_id = _key(row, "Station Id", "Station_ID", "Station Code")
        observation_date = _date(_key(row, "Date of Observation"), ("%Y-%m-%d", "%d-%m-%Y"))
        observation_time = _key(row, "Time of Observation")
        if station_id is None or observation_date is None or observation_time is None:
            continue
        parsed_time = None
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                parsed_time = datetime.strptime(str(observation_time), fmt).time()
                break
            except ValueError:
                pass
        if parsed_time is None:
            continue
        result.append({
            "station_id": str(station_id),
            "observation_datetime": datetime.combine(observation_date, parsed_time, tzinfo=timezone.utc),
            "temperature": _number(_key(row, "Temperature")),
            "humidity": _number(_key(row, "Humidity")),
            "wind_speed": _number(_key(row, "Wind Speed")),
            "wind_direction": _number(_key(row, "Wind Direction")),
            "rainfall_mm": _number(_key(row, "Last 24 hrs Rainfall")),
            "data_source": "IMD API: current_wx",
        })
    frame = pd.DataFrame(result).drop_duplicates(
        subset=["station_id", "observation_datetime", "data_source"]
    ) if result else pd.DataFrame()
    if not frame.empty:
        frame.loc[frame["rainfall_mm"] < 0, "rainfall_mm"] = None
    return frame
