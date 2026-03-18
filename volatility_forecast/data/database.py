import os
from sqlalchemy import create_engine, Column, Integer, Float, Date, String
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv


Base = declarative_base()


class PriceVolumeData(Base):
    __tablename__ = "price_volume_data"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    ticker = Column(String, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adjOpen = Column(Float)
    adjHigh = Column(Float)
    adjLow = Column(Float)
    adjClose = Column(Float)
    adjVolume = Column(Float)


def get_database_local_path():
    load_dotenv()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.abspath(os.path.join(base_dir, os.pardir, "data"))

    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_local_database_url():
    load_dotenv()
    data_dir = get_database_local_path()
    data_base_name = os.getenv("LOCAL_DATABASE_NAME", "local_database.db")
    database_url = os.getenv(
        "DATABASE_URL", f"sqlite:///{os.path.join(data_dir, data_base_name)}"
    )

    return database_url


_engine = None
_Session = None
_SESSION_OVERRIDE = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(get_local_database_url())
        Base.metadata.create_all(_engine)
    return _engine


def get_session():
    global _Session
    if _SESSION_OVERRIDE is not None:
        return _SESSION_OVERRIDE
    if _Session is None:
        _Session = sessionmaker(bind=get_engine())
    return _Session()


def set_session_override(session):
    global _SESSION_OVERRIDE
    _SESSION_OVERRIDE = session


def clear_session_override():
    global _SESSION_OVERRIDE
    _SESSION_OVERRIDE = None


def is_session_override(session) -> bool:
    return _SESSION_OVERRIDE is not None and session is _SESSION_OVERRIDE
