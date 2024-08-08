# model_management.py
import joblib
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class ModelMetadata(Base):
    __tablename__ = 'model_metadata'
    id = Column(Integer, primary_key=True)
    model_type = Column(String)
    version = Column(Integer)
    training_date = Column(DateTime)
    performance_metric = Column(Float)
    file_path = Column(String)

class ModelManager:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_model(self, model, model_type, performance_metric):
        session = self.Session()
        latest_version = session.query(ModelMetadata).filter_by(model_type=model_type).order_by(ModelMetadata.version.desc()).first()
        new_version = 1 if latest_version is None else latest_version.version + 1
        
        file_path = f"models/{model_type}_v{new_version}.joblib"
        joblib.dump(model, file_path)
        
        metadata = ModelMetadata(
            model_type=model_type,
            version=new_version,
            training_date=datetime.now(),
            performance_metric=performance_metric,
            file_path=file_path
        )
        session.add(metadata)
        session.commit()
        session.close()

    def load_latest_model(self, model_type):
        session = self.Session()
        latest_model = session.query(ModelMetadata).filter_by(model_type=model_type).order_by(ModelMetadata.version.desc()).first()
        session.close()
        if latest_model:
            return joblib.load(latest_model.file_path)
        return None

    def get_model_history(self, model_type):
        session = self.Session()
        history = session.query(ModelMetadata).filter_by(model_type=model_type).order_by(ModelMetadata.version).all()
        session.close()
        return history