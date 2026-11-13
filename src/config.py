from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, make_url

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    imd_api_base: str = _setting("IMD_API_BASE", "https://api.imd.gov.in/api/v1").rstrip("/")
    imd_auth_header: str = _setting("IMD_AUTH_HEADER")
    imd_auth_value: str = _setting("IMD_AUTH_VALUE")
    weatherapi_key: str = _setting("WEATHERAPI_KEY").strip()
    database_url_override: str = _setting("DATABASE_URL").strip()
    db_host: str = _setting("DB_HOST", "localhost")
    db_port: int = int(_setting("DB_PORT", "5432"))
    db_name: str = _setting("DB_NAME", "indian_rainfall_db")
    db_user: str = _setting("DB_USER")
    db_password: str = _setting("DB_PASSWORD")
    db_sslmode: str = _setting("DB_SSLMODE", "prefer")
    auto_create_schema: bool = _setting("AUTO_CREATE_SCHEMA", "false").lower() == "true"

    def database_url(self) -> URL:
        if self.database_url_override:
            url = make_url(self.database_url_override)
            driver = "postgresql+psycopg" if url.drivername in {"postgres", "postgresql"} else url.drivername
            return url.set(drivername=driver)
        if not self.db_user:
            raise RuntimeError("Set DB_USER in the local .env file before connecting to PostgreSQL")
        if not self.db_password:
            raise RuntimeError("Set DB_PASSWORD in the local .env file before connecting to PostgreSQL")
        return URL.create(
            "postgresql+psycopg", username=self.db_user, password=self.db_password,
            host=self.db_host, port=self.db_port, database=self.db_name,
            query={"sslmode": self.db_sslmode},
        )


settings = Settings()
