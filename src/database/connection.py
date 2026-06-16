from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine(database_url: str | None = None) -> Engine:
    global _engine
    if _engine is None:
        url = database_url or DATABASE_URL
        _engine = create_engine(url, echo=False)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()


def initialize_database(database_url: str | None = None) -> None:
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)


def get_db() -> Generator[Session, None, None]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()
