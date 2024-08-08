# scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

class Scheduler:
    def __init__(self, data_manager, model_manager, forecast_generator, evaluator, report_generator):
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.forecast_generator = forecast_generator
        self.evaluator = evaluator
        self.report_generator = report_generator
        self.scheduler = BackgroundScheduler()

    def daily_task(self):
        today = datetime.now().date()
        data = self.data_manager.fetch_data('SPY', today - timedelta(days=365), today)
        processed_data = self.data_manager.preprocess_data(data)
        self.data_manager.store_data(processed_data, 'spy_data')
        
        forecasts = self.forecast_generator.generate_forecasts(processed_data)
        self.forecast_generator.store_forecasts(forecasts, today)
        
        report = self.report_generator.generate_daily_report(today)
        print(f"Daily Report for {today}:", report)

    def recalibrate_models(self):
        data = self.data_manager.get_latest_data('spy_data', 1000)
        for model_type in ['ES', 'STES', 'XGBoost-STES']:
            old_model = self.model_manager.load_latest_model(model_type)
            new_model = self.train_new_model(model_type, data)
            comparison = self.evaluator.compare_models(new_model, old_model, data)
            if comparison['improvement'] > 0:
                self.model_manager.save_model(new_model, model_type, comparison['new_model_rmse'])
                print(f"New {model_type} model saved with RMSE: {comparison['new_model_rmse']}")
            else:
                print(f"No improvement for {model_type} model")

    def train_new_model(self, model_type, data):
        # Implement model training logic here
        pass

    def start(self):
        self.scheduler.add_job(self.daily_task, 'cron', hour=0, minute=5)  # Run at 00:05 every day
        self.scheduler.add_job(self.recalibrate_models, 'cron', day_of_week='sun')  # Run every Sunday
        self.scheduler.start()