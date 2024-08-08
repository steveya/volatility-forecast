# api.py
from flask import Flask, jsonify, request
from scheduler import Scheduler

app = Flask(__name__)
scheduler = Scheduler(data_manager, model_manager, forecast_generator, evaluator, report_generator)

@app.route('/forecast/<date>', methods=['GET'])
def get_forecast(date):
    forecasts = forecast_generator.get_forecasts(date)
    return jsonify(forecasts)

@app.route('/report/<date>', methods=['GET'])
def get_report(date):
    report = report_generator.generate_daily_report(date)
    return jsonify(report)

@app.route('/model/<model_type>/history', methods=['GET'])
def get_model_history(model_type):
    history = model_manager.get_model_history(model_type)
    return jsonify(history)

@app.route('/recalibrate', methods=['POST'])
def trigger_recalibration():
    scheduler.recalibrate_models()
    return jsonify({"message": "Recalibration triggered"})

if __name__ == '__main__':
    scheduler.start()
    app.run(debug=True)