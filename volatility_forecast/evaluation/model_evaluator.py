# evaluation.py
import numpy as np
from sklearn.metrics import mean_squared_error


class Evaluator:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def calculate_rmse(self, true_values, predictions):
        return np.sqrt(mean_squared_error(true_values, predictions))

    def evaluate_model(self, model, test_data):
        predictions = model.predict(test_data)
        true_values = test_data["squared_returns"]
        return self.calculate_rmse(true_values, predictions)

    def compare_models(self, new_model, old_model, test_data):
        new_rmse = self.evaluate_model(new_model, test_data)
        old_rmse = self.evaluate_model(old_model, test_data)
        return {
            "new_model_rmse": new_rmse,
            "old_model_rmse": old_rmse,
            "improvement": old_rmse - new_rmse,
        }
