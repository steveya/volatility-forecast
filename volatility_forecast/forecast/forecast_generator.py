# forecast_generation.py
import pandas as pd
from sqlalchemy import create_engine

class ForecastGenerator:
    def __init__(self, db_url, model_manager):
        self.engine = create_engine(db_url)
        self.model_manager = model_manager

    def generate_forecasts(self, data):
        forecasts = {}
        for model_type in ['ES', 'STES', 'XGBoost-STES']:
            model = self.model_manager.load_latest_model(model_type)
            if model:
                forecasts[model_type] = model.predict(data)[-1]
        return forecasts

    def store_forecasts(self, forecasts, date):
        df = pd.DataFrame(forecasts, index=[date])
        df.to_sql('forecasts', self.engine, if_exists='append')