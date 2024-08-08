# reporting.py
import pandas as pd
from sqlalchemy import create_engine

class ReportGenerator:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)

    def generate_daily_report(self, date):
        query = f"""
        SELECT f.*, a.squared_returns as actual_volatility
        FROM forecasts f
        JOIN actual_data a ON f.date = a.date
        WHERE f.date = '{date}'
        """
        data = pd.read_sql(query, self.engine)
        report = data.to_dict(orient='records')[0]
        for model in ['ES', 'STES', 'XGBoost-STES']:
            report[f'{model}_error'] = report['actual_volatility'] - report[model]
        return report

    def generate_comparison_report(self, model_type, old_version, new_version):
        query = f"""
        SELECT * FROM model_metadata
        WHERE model_type = '{model_type}'
        AND version IN ({old_version}, {new_version})
        """
        data = pd.read_sql(query, self.engine)
        return data.to_dict(orient='records')