# src/main.py

import yaml
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

from volatility_forecast.data.datamanager import DataManager
from models.es_model import ESModel
from models.stes_model import STESModel
from models.xgboost_stes_model import XGBoostSTESModel
from utils.model_management import ModelManager
from forecast.forecast_generation import ForecastGenerator
from evaluation.evaluation import Evaluator
from reporting.reporting import ReportGenerator
from utils.scheduler import Scheduler

def load_config():
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def initialize_components(config):
    db_url = config['database']['url']
    engine = create_engine(db_url)
    
    data_manager = DataManager(db_url)
    model_manager = ModelManager(db_url)
    forecast_generator = ForecastGenerator(db_url, model_manager)
    evaluator = Evaluator(data_manager)
    report_generator = ReportGenerator(db_url)
    
    return data_manager, model_manager, forecast_generator, evaluator, report_generator, engine

def fetch_and_preprocess_data(data_manager, config):
    all_data = {}
    for ticker in config['data']['tickers']:
        data = data_manager.fetch_data(ticker, config['data']['start_date'], config['data']['end_date'])
        processed_data = data_manager.preprocess_data(data)
        all_data[ticker] = processed_data
        data_manager.store_data(processed_data, f'{ticker}_data')
    return all_data

def train_and_persist_models(all_data, model_manager, evaluator, config):
    for ticker, data in all_data.items():
        X = data[['returns', 'abs_returns', 'squared_returns']]
        y = data['squared_returns']
        
        # Split the data
        split_index = int(len(X) * (1 - config['evaluation']['test_size']))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        for model_type in config['models']:
            if model_type == 'ES':
                model = ESModel()
            elif model_type.startswith('STES'):
                features = model_type.split('-')[1:]
                model = STESModel(features=features)
            elif model_type == 'XGBoost-STES':
                model = XGBoostSTESModel(**config['xgboost_params'])
            
            model.fit(X_train, y_train)
            performance = evaluator.evaluate_model(model, X_test, y_test)
            model_manager.save_model(model, f"{ticker}_{model_type}", performance)

def run_daily_process(data_manager, forecast_generator, evaluator, report_generator, config):
    for ticker in config['data']['tickers']:
        today = datetime.now().date()
        data = data_manager.get_latest_data(f'{ticker}_data', 365)  # Get last year's data
        forecasts = forecast_generator.generate_forecasts(data, ticker)
        forecast_generator.store_forecasts(forecasts, today, ticker)
        
        # Assuming you have actual data for today (you might need to adjust this)
        actual_data = data_manager.fetch_data(ticker, today, today)
        actual_volatility = actual_data['returns'].iloc[-1] ** 2
        
        report = report_generator.generate_daily_report(today, ticker, actual_volatility)
        print(f"Daily Report for {ticker} on {today}:", report)

def main():
    config = load_config()
    data_manager, model_manager, forecast_generator, evaluator, report_generator, engine = initialize_components(config)
    
    # Fetch and preprocess data
    all_data = fetch_and_preprocess_data(data_manager, config)
    
    # Train and persist initial models
    train_and_persist_models(all_data, model_manager, evaluator, config)
    
    # Run daily process once
    run_daily_process(data_manager, forecast_generator, evaluator, report_generator, config)
    
    # Setup and start scheduler
    scheduler = Scheduler(data_manager, model_manager, forecast_generator, evaluator, report_generator, config)
    scheduler.start()
    
    # Keep the script running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down...")
        scheduler.shutdown()

if __name__ == "__main__":
    main()