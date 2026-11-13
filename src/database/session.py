from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import settings
from .models import Base


def make_engine():
    return create_engine(settings.database_url(), pool_pre_ping=True, future=True)


def create_schema(engine=None) -> None:
    engine = engine or make_engine()
    Base.metadata.create_all(engine)


def make_session_factory(engine=None):
    engine = engine or make_engine()
    return sessionmaker(bind=engine, expire_on_commit=False)
