import os
from sqlalchemy import create_engine, Column, Integer, Float, Date, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
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


def get_database_loca_path():
    load_dotenv()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.abspath(os.path.join(base_dir, os.pardir, "data"))

    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_loca_database_url():
    load_dotenv()
    data_dir = get_database_loca_path()
    data_base_name = os.getenv("LOCAL_DATABASE_NAME", "local_database.db")
    database_url = os.getenv(
        "DATABASE_URL", f"sqlite:///{os.path.join(data_dir, data_base_name)}"
    )

    return database_url


engine = create_engine(get_loca_database_url())
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def get_session():
    return Session()
