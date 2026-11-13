"""Create the configured PostgreSQL database (if absent) and its schema."""

import psycopg
from psycopg import sql

from ..config import settings
from .session import create_schema, make_engine


def main() -> None:
    if settings.database_url_override:
        create_schema(make_engine())
        print("Created/verified the rainfall project schema")
        return
    if not settings.db_user or not settings.db_password:
        raise SystemExit("Set DB_USER and DB_PASSWORD in your local .env file first.")
    with psycopg.connect(
        dbname="postgres", user=settings.db_user, password=settings.db_password,
        host=settings.db_host, port=settings.db_port, sslmode=settings.db_sslmode,
        autocommit=True,
    ) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (settings.db_name,)
        ).fetchone()
        if not exists:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(settings.db_name)))
            print(f"Created database {settings.db_name}")
        else:
            print(f"Database {settings.db_name} already exists")
    create_schema(make_engine())
    print("Created/verified the rainfall project schema")


if __name__ == "__main__":
    main()
