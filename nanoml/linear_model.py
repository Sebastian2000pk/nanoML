from .base import BaseEstimator
import numpy as np

class LinearRegression(BaseEstimator):
  def __init__(self):
    self.coef_: list[float] = None
    self.intercept_: float = None

  def fit(self, X: list, y: list):
    if X is None or y is None:
      raise ValueError("X and y cannot be None")

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] != y.shape[0]:
      raise ValueError("X and y must have the same length")

    # aplanado temporal para asegurar que sean 1D
    X = X.reshape(-1)
    y = y.flatten()

    x_mean = sum(X) / len(X)
    y_mean = sum(y) / len(y)

    numerator = sum((X - x_mean) * (y - y_mean))
    denominator = sum((X - x_mean)**2)
    
    self.coef_ = numerator/denominator
    self.intercept_ = y_mean - self.coef_*x_mean

  def predict(self, X: list[float]) -> list[float]:
    Y = self.intercept_ + self.coef_*X
    return Y.flatten()