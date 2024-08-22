import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(data_provider, model, metrics, is_index, os_index, oos_index=None):
    y, train_sample, test_sample, y_pred, date = generate_model_forecasts(
        data_provider, model, is_index, os_index, oos_index
    )
    return (
        metrics(y_pred[test_sample], y[test_sample]),
        metrics(y_pred[train_sample], y[train_sample]),
        pd.Series(y_pred, index=date[1:]),
        pd.Series(y.flatten(), index=date[1:]),
    )


def generate_model_forecasts(data_provider, model, is_index, os_index, oos_index):
    X, y, returns, date = data_provider()

    train_sample = slice(is_index, os_index)
    test_sample = slice(os_index, oos_index)

    model.fit(X, y, returns, is_index, os_index)

    y_pred = model.predict(X, returns)
    return y, train_sample, test_sample, y_pred, date


def compare_models(new_model, old_model, test_data):
    new_rmse = evaluate_model(new_model, test_data)
    old_rmse = evaluate_model(old_model, test_data)
    return {
        "new_model_rmse": new_rmse,
        "old_model_rmse": old_rmse,
        "improvement": old_rmse - new_rmse,
    }
