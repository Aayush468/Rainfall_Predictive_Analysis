from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from ..database.models import (District, DistrictWarning, RainfallForecast,
                               RainfallObservation, State, StateRainfallObservation,
                               WeatherObservation)
from ..database.session import make_engine

logger = logging.getLogger(__name__)


def _clean(value):
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _state_id(connection, state_name: str) -> int:
    row = connection.execute(select(State.state_id).where(State.state_name.ilike(state_name))).first()
    if row:
        return row[0]
    result = connection.execute(insert(State).values(state_name=state_name).on_conflict_do_nothing(
        index_elements=[State.state_name]).returning(State.state_id)).first()
    if result:
        return result[0]
    return connection.execute(select(State.state_id).where(State.state_name.ilike(state_name))).scalar_one()


def _district_id(connection, imd_id, name: str, state_name: str | None = None) -> int | None:
    try:
        district_id = int(imd_id) if imd_id not in (None, "") else None
    except (TypeError, ValueError):
        district_id = None
    state_id = _state_id(connection, state_name) if state_name else None
    if district_id is not None and state_id is not None:
        connection.execute(insert(District).values(
            district_id=district_id, district_name=name, state_id=state_id
        ).on_conflict_do_update(index_elements=[District.district_id], set_={
            "district_name": name, "state_id": state_id,
        }))
        return district_id
    stmt = select(District.district_id).where(District.district_name.ilike(name))
    if state_id is not None:
        stmt = stmt.where(District.state_id == state_id)
    matches = connection.execute(stmt.limit(2)).scalars().all()
    return matches[0] if len(matches) == 1 else None


def load_state_rainfall(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    engine = make_engine()
    inserted = 0
    with engine.begin() as connection:
        for state_name, rows in frame.groupby("state_name"):
            state_id = _state_id(connection, state_name)
            records = [{
                "state_id": state_id, "observation_date": _clean(row.observation_date),
                "actual_rainfall_mm": _clean(row.actual_rainfall_mm),
                "normal_rainfall_mm": _clean(row.normal_rainfall_mm),
                "departure_percentage": _clean(row.departure_percentage),
                "rainfall_category": _clean(row.rainfall_category),
                "data_source": _clean(row.data_source),
            } for row in rows.itertuples(index=False)]
            result = connection.execute(insert(StateRainfallObservation).on_conflict_do_nothing(
                constraint="uq_state_rainfall_date_source"), records)
            inserted += result.rowcount or 0
    return inserted


def load_district_rainfall(frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty:
        return 0, 0
    engine = make_engine()
    inserted = rejected = 0
    with engine.begin() as connection:
        records = []
        for row in frame.itertuples(index=False):
            district_id = _district_id(connection, row.imd_district_id, row.district_name)
            if district_id is None:
                rejected += 1
                continue
            records.append({
                "district_id": district_id, "observation_date": _clean(row.observation_date),
                "actual_rainfall_mm": _clean(row.actual_rainfall_mm),
                "normal_rainfall_mm": _clean(row.normal_rainfall_mm),
                "departure_percentage": _clean(row.departure_percentage),
                "rainfall_category": _clean(row.rainfall_category), "data_source": _clean(row.data_source),
            })
        if records:
            result = connection.execute(insert(RainfallObservation).on_conflict_do_nothing(
                constraint="uq_rainfall_district_date_source"), records)
            inserted = result.rowcount or 0
    return inserted, rejected


def load_forecasts(frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty:
        return 0, 0
    engine = make_engine()
    inserted = rejected = 0
    with engine.begin() as connection:
        records = []
        for row in frame.itertuples(index=False):
            district_id = _district_id(connection, row.imd_district_id, row.district_name, row.state_name)
            if district_id is None or pd.isna(row.forecast_date):
                rejected += 1
                continue
            records.append({
                "district_id": district_id, "forecast_date": _clean(row.forecast_date),
                "forecast_day": _clean(row.forecast_day),
                "rainfall_distribution": _clean(row.rainfall_distribution),
                "distribution_percentage": _clean(row.distribution_percentage),
                "data_source": _clean(row.data_source),
            })
        if records:
            result = connection.execute(insert(RainfallForecast).on_conflict_do_nothing(
                constraint="uq_forecast_district_date_day_source"), records)
            inserted = result.rowcount or 0
    return inserted, rejected


def load_warnings(frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty:
        return 0, 0
    engine = make_engine()
    inserted = rejected = 0
    with engine.begin() as connection:
        records = []
        for row in frame.itertuples(index=False):
            district_id = _district_id(connection, row.imd_district_id, row.district_name)
            if district_id is None or pd.isna(row.warning_date):
                rejected += 1
                continue
            records.append({
                "district_id": district_id, "warning_date": row.warning_date,
                "forecast_day": _clean(row.forecast_day), "warning_level": _clean(row.warning_level),
                "warning_description": _clean(row.warning_description), "data_source": _clean(row.data_source),
            })
        if records:
            result = connection.execute(insert(DistrictWarning).on_conflict_do_nothing(
                constraint="uq_warning_district_date_day_source"), records)
            inserted = result.rowcount or 0
    return inserted, rejected


def load_current_weather(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    records = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    with make_engine().begin() as connection:
        result = connection.execute(insert(WeatherObservation).on_conflict_do_nothing(
            constraint="uq_weather_station_datetime_source"), records)
        return result.rowcount or 0
