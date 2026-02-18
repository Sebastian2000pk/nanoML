from .base import BaseEstimator

class LinearRegression(BaseEstimator):
  def __init__(self):
    self.b0: float = 0.0 
    self.b1: float = 0.0
    self.coef_: list[float] = None
    self.intercept_: float = None

  def fit(self, X: list, y: list):
    pass

  def predict(self, X: list[float]) -> list[float]:
    Y = self.b0 + self.b1*X
    return Y.flatten()