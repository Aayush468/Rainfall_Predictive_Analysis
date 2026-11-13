from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (Date, DateTime, Float, ForeignKey, Index, Integer,
                        String, Text, UniqueConstraint, func)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class State(Base):
    __tablename__ = "states"
    state_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state_name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    state_code: Mapped[str | None] = mapped_column(String(20), unique=True)


class District(Base):
    __tablename__ = "districts"
    district_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    district_name: Mapped[str] = mapped_column(String(160), nullable=False)
    state_id: Mapped[int] = mapped_column(ForeignKey("states.state_id"), nullable=False)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("district_name", "state_id", name="uq_district_state"),)


class RainfallObservation(Base):
    __tablename__ = "rainfall_observations"
    observation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    district_id: Mapped[int] = mapped_column(ForeignKey("districts.district_id"), nullable=False)
    observation_date: Mapped[date] = mapped_column(Date, nullable=False)
    actual_rainfall_mm: Mapped[float | None] = mapped_column(Float)
    normal_rainfall_mm: Mapped[float | None] = mapped_column(Float)
    departure_percentage: Mapped[float | None] = mapped_column(Float)
    rainfall_category: Mapped[str | None] = mapped_column(String(40))
    data_source: Mapped[str] = mapped_column(String(240), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        UniqueConstraint("district_id", "observation_date", "data_source", name="uq_rainfall_district_date_source"),
        Index("ix_rainfall_district_date", "district_id", "observation_date"),
    )


class StateRainfallObservation(Base):
    __tablename__ = "state_rainfall_observations"
    observation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state_id: Mapped[int] = mapped_column(ForeignKey("states.state_id"), nullable=False)
    observation_date: Mapped[date] = mapped_column(Date, nullable=False)
    actual_rainfall_mm: Mapped[float | None] = mapped_column(Float)
    normal_rainfall_mm: Mapped[float | None] = mapped_column(Float)
    departure_percentage: Mapped[float | None] = mapped_column(Float)
    rainfall_category: Mapped[str | None] = mapped_column(String(40))
    data_source: Mapped[str] = mapped_column(String(240), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        UniqueConstraint("state_id", "observation_date", "data_source", name="uq_state_rainfall_date_source"),
        Index("ix_state_rainfall_state_date", "state_id", "observation_date"),
    )


class WeatherObservation(Base):
    __tablename__ = "weather_observations"
    weather_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    station_id: Mapped[str] = mapped_column(String(80), nullable=False)
    district_id: Mapped[int | None] = mapped_column(ForeignKey("districts.district_id"))
    observation_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    temperature: Mapped[float | None] = mapped_column(Float)
    humidity: Mapped[float | None] = mapped_column(Float)
    wind_speed: Mapped[float | None] = mapped_column(Float)
    wind_direction: Mapped[float | None] = mapped_column(Float)
    rainfall_mm: Mapped[float | None] = mapped_column(Float)
    data_source: Mapped[str] = mapped_column(String(240), nullable=False)
    __table_args__ = (
        UniqueConstraint("station_id", "observation_datetime", "data_source", name="uq_weather_station_datetime_source"),
        Index("ix_weather_district_datetime", "district_id", "observation_datetime"),
    )


class RainfallForecast(Base):
    __tablename__ = "rainfall_forecasts"
    forecast_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    district_id: Mapped[int] = mapped_column(ForeignKey("districts.district_id"), nullable=False)
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False)
    forecast_day: Mapped[int] = mapped_column(Integer, nullable=False)
    rainfall_distribution: Mapped[str | None] = mapped_column(String(100))
    distribution_percentage: Mapped[str | None] = mapped_column(String(100))
    data_source: Mapped[str] = mapped_column(String(240), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        UniqueConstraint("district_id", "forecast_date", "forecast_day", "data_source", name="uq_forecast_district_date_day_source"),
        Index("ix_forecast_district_date", "district_id", "forecast_date"),
    )


class DistrictWarning(Base):
    __tablename__ = "district_warnings"
    warning_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    district_id: Mapped[int] = mapped_column(ForeignKey("districts.district_id"), nullable=False)
    warning_date: Mapped[date] = mapped_column(Date, nullable=False)
    forecast_day: Mapped[int] = mapped_column(Integer, nullable=False)
    warning_level: Mapped[str | None] = mapped_column(String(40))
    warning_description: Mapped[str | None] = mapped_column(Text)
    data_source: Mapped[str] = mapped_column(String(240), nullable=False)
    __table_args__ = (
        UniqueConstraint("district_id", "warning_date", "forecast_day", "data_source", name="uq_warning_district_date_day_source"),
        Index("ix_warning_district_date", "district_id", "warning_date"),
    )


class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    district_id: Mapped[int | None] = mapped_column(ForeignKey("districts.district_id"))
    prediction_date: Mapped[date] = mapped_column(Date, nullable=False)
    predicted_rainfall_mm: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (Index("ix_prediction_district_date", "district_id", "prediction_date"),)


class GridMLPrediction(Base):
    __tablename__ = "grid_ml_predictions"
    grid_prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    grid_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    grid_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_date: Mapped[date] = mapped_column(Date, nullable=False)
    predicted_rainfall_mm: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        UniqueConstraint("grid_latitude", "grid_longitude", "prediction_date", "model_version", name="uq_grid_prediction_date_model"),
        Index("ix_grid_prediction_date", "prediction_date"),
    )


class RainfallGridObservation(Base):
    """IMD gridded rows remain separate from district/station observations."""
    __tablename__ = "rainfall_grid_observations"
    grid_observation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    grid_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    grid_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    observation_date: Mapped[date] = mapped_column(Date, nullable=False)
    rainfall_mm: Mapped[float | None] = mapped_column(Float)
    data_source: Mapped[str] = mapped_column(String(240), nullable=False)
    __table_args__ = (
        UniqueConstraint("grid_latitude", "grid_longitude", "observation_date", "data_source", name="uq_grid_cell_date_source"),
        Index("ix_grid_date", "observation_date"),
    )


class GridRainfallForecast(Base):
    """Numeric model forecasts; kept separate from IMD categorical outlooks."""
    __tablename__ = "grid_rainfall_forecasts"
    forecast_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    grid_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    grid_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False)
    rainfall_mm: Mapped[float | None] = mapped_column(Float)
    precipitation_probability_pct: Mapped[float | None] = mapped_column(Float)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    data_source: Mapped[str] = mapped_column(String(240), nullable=False)
    __table_args__ = (
        UniqueConstraint("grid_latitude", "grid_longitude", "forecast_date", "data_source", name="uq_grid_forecast_cell_date_source"),
        Index("ix_grid_forecast_date", "forecast_date"),
    )


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"
    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(240), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(240), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status_code: Mapped[int | None] = mapped_column(Integer)
    records_received: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
