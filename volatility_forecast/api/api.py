# api.py
from flask import Flask, jsonify, request
from volatility_forecast.api.utils.scheduler import Scheduler

app = Flask(__name__)

# These are initialized at startup via init_app().
data_manager = None
model_manager = None
forecast_generator = None
evaluator = None
report_generator = None
scheduler = None


def init_app(dm, mm, fg, ev, rg):
    """Wire up the module-level components before first request."""
    global data_manager, model_manager, forecast_generator, evaluator, report_generator, scheduler
    data_manager = dm
    model_manager = mm
    forecast_generator = fg
    evaluator = ev
    report_generator = rg
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