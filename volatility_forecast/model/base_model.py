# base_model.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import joblib


class BaseVolatilityModel(BaseEstimator, RegressorMixin, ABC):
    @property
    def model_name(self):
        pass

    @abstractmethod
    def fit(self, X, y, returns):
        pass

    @abstractmethod
    def predict(self, X, returns):
        pass

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)
