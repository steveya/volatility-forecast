import numpy as np
import xgboost as xgb
from scipy.special import expit
from base_model import BaseVolatilityModel


class XGBoostSTESModel(BaseVolatilityModel):
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model = None

    def _stes_variance_objective(self, preds, dtrain):
        labels = dtrain.get_label()
        alphas = expit(preds)
        returns2 = dtrain.get_feature_names(["returns^2"])

        grads = np.zeros_like(preds)
        hesss = np.zeros_like(preds)
        varhs = np.zeros_like(preds)

        for t in range(len(alphas)):
            if t == 0:
                lvar_f = np.mean(returns2[:500])
                varhs[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * lvar_f
                d_alpha = returns2[t] - lvar_f
            else:
                varhs[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * varhs[t - 1]
                d_alpha = returns2[t] - varhs[t - 1]

            d_pred = -alphas[t] * (1 - alphas[t]) * d_alpha
            grads[t] = 2 * (labels[t] - varhs[t]) * d_pred
            hesss[t] = 2 * d_pred**2

        return grads, hesss

    def fit(self, X, y, returns):
        dtrain = xgb.DMatrix(X, label=X["returns^2"])
        self.model = xgb.train(
            self.xgb_params,
            dtrain,
            obj=self._stes_variance_objective,
            evals=[(dtrain, "train")],
            verbose_eval=False,
        )
        return self

    def predict(self, X, returns):
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        alphas = expit(preds)
        returns2 = X["returns^2"].values
        n = len(returns2)
        sigma2 = np.zeros(n)
        sigma2[0] = returns2[0]
        for t in range(1, n):
            sigma2[t] = alphas[t] * returns2[t - 1] + (1 - alphas[t]) * sigma2[t - 1]
        return sigma2

    def save(self, filename):
        self.model.save_model(filename + ".model")
        super().save(filename)

    @classmethod
    def load(cls, filename):
        model = super().load(filename)
        model.model = xgb.Booster()
        model.model.load_model(filename + ".model")
        return model
