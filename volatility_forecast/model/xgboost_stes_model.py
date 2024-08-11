import numpy as np
import xgboost as xgb
from functools import partial
from scipy.special import expit

from .base_model import BaseVolatilityModel

DEFAULT_XGBOOST_PARAMS = {
    "num_boost_round": 10,  # Number of boosting iterations
    "max_depth": 3,  # Maximum depth of the tree
    "learning_rate": 0.1,  # Learning rate
    "colsample_bytree": 0.8,  # Column subsampling at tree level
    "colsample_bylevel": 0.8,  # Column subsampling at level level
    "colsample_bynode": 0.8,  # Column subsampling at node level
    "reg_lambda": 1.0,  # L2 regularization
    "random_state": 42,  # Seed for reproducibility
}


class XGBoostSTESModel(BaseVolatilityModel):
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model = None

    def _stes_variance_objective(self, preds, dtrain, returns):
        """
        This function computes the gradient and hessian of the objective
        function: the mean variance forecast squared error. The gradient is computed
        using the recursive formula for the variance. The hessian is computed
        using the chain rule.

        Args:
            preds (np.ndarray): The predicted values of the model.
            dtrain (xgb.DMatrix): The training data.
            returns (np.ndarray): The returns time series.

        Returns:
            tuple: The gradient and hessian of the objective function.
        """
        labels = dtrain.get_label()
        alphas = expit(preds)
        returns2 = returns**2

        grads = np.zeros_like(preds)
        hesss = np.zeros_like(preds)
        varhs = np.zeros_like(preds)

        for t in range(len(alphas)):
            if t == 0:
                varhs[t] = returns2[t]
                d_alpha = 0
            else:
                varhs[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * varhs[t - 1]
                d_alpha = returns2[t] - varhs[t - 1]

            d_pred = -alphas[t] * (1 - alphas[t]) * d_alpha
            grads[t] = 2 * (labels[t] - varhs[t]) * d_pred
            hesss[t] = 2 * d_pred**2

        return grads, hesss

    def fit(self, X, y, returns, start_index, end_index):
        dtrain = xgb.DMatrix(
            X[start_index:end_index, :], label=y[start_index:end_index]
        )
        self.model = xgb.train(
            self.xgb_params,
            dtrain,
            obj=partial(
                self._stes_variance_objective, returns=returns[start_index:end_index]
            ),
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )
        return self

    def predict(self, X, returns):
        """
        This function generates predictions for the 1-step ahead variance
        from the features time series. Since the variance is recursively
        computed, X should be the full sample of features. Once computed,
        one can take the slice of indices of interest (val_index for example)
        to compute the metrics.

        Args:
            X (pd.DataFrame): The features time series.
            burnin_size (int): The number of samples to use to initialize
            the variance.

        Returns:
            np.ndarray: The 1-step ahead variance predictions.
        """
        alphas = expit(self.model.predict(xgb.DMatrix(X)))

        returns2 = returns**2
        var_pred = np.zeros_like(alphas)
        var_pred[0] = returns[0] ** 2
        for t in range(1, len(returns2)):
            var_pred[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * var_pred[t - 1]

        return var_pred

    def save(self, filename):
        self.model.save_model(filename + ".model")
        super().save(filename)

    @classmethod
    def load(cls, filename):
        model = super().load(filename)
        model.model = xgb.Booster()
        model.model.load_model(filename + ".model")
        return model
