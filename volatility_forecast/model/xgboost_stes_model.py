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
    "random_state": 42,  # Seed for reproducibility,
    "verbosity": 0,
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

    def fit(self, X, y, **kwargs):
        returns = kwargs.pop("returns", None)
        start_index = kwargs.pop("start_index", 0)
        end_index = kwargs.pop("end_index", len(X))

        Xn = (
            X.to_numpy(dtype=float)
            if hasattr(X, "to_numpy")
            else np.asarray(X, dtype=float)
        )
        yn = (
            y.to_numpy(dtype=float)
            if hasattr(y, "to_numpy")
            else np.asarray(y, dtype=float)
        )

        if returns is None:
            raise ValueError("XGBoostSTESModel.fit requires `returns=` to be provided")
        rn = (
            returns.to_numpy(dtype=float)
            if hasattr(returns, "to_numpy")
            else np.asarray(returns, dtype=float)
        )

        params = dict(self.xgb_params)
        num_boost_round = int(params.pop("num_boost_round", 10))
        # xgboost.train uses `seed`, not `random_state`.
        if "random_state" in params and "seed" not in params:
            params["seed"] = int(params.pop("random_state"))

        dtrain = xgb.DMatrix(
            Xn[start_index:end_index, :], label=yn[start_index:end_index]
        )
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            obj=partial(
                self._stes_variance_objective, returns=rn[start_index:end_index]
            ),
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )
        return self

    def predict(self, X, **kwargs):
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
        var_pred, _ = self.predict_with_alpha(X, **kwargs)
        return var_pred

    def predict_with_alpha(self, X, **kwargs):
        """
        This function generates predictions for the 1-step ahead variance
        and returns the alpha series as well.

        Args:
            X (pd.DataFrame): The features time series.

        Returns:
            tuple[np.ndarray, np.ndarray]: The 1-step ahead variance predictions and the alpha series.
        """
        returns = kwargs.pop("returns", None)

        Xn = (
            X.to_numpy(dtype=float)
            if hasattr(X, "to_numpy")
            else np.asarray(X, dtype=float)
        )
        if returns is None:
            raise ValueError(
                "XGBoostSTESModel.predict_with_alpha requires `returns=` to be provided"
            )
        rn = (
            returns.to_numpy(dtype=float)
            if hasattr(returns, "to_numpy")
            else np.asarray(returns, dtype=float)
        )

        alphas = expit(self.model.predict(xgb.DMatrix(Xn)))

        returns2 = rn**2
        var_pred = np.zeros_like(alphas)
        var_pred[0] = rn[0] ** 2
        for t in range(1, len(returns2)):
            var_pred[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * var_pred[t - 1]

        return var_pred, alphas

    # Scikit-learn interface
    def get_params(self, deep=True):
        """
        Return the parameters of the estimator.
        """
        return self.xgb_params

    def set_params(self, **params):
        for key, value in params.items():
            self.xgb_params[key] = value

        return self

    def save(self, filename):
        self.model.save_model(filename + ".model")
        super().save(filename)

    @classmethod
    def load(cls, filename):
        model = super().load(filename)
        model.model = xgb.Booster()
        model.model.load_model(filename + ".model")
        return model
