from .base import BaseEstimator

class LinearRegression(BaseEstimator):
  def __init__(self):
    self.b0 = 0
    self.b1 = 0
  
  def fit(self, X, y):
    pass

  def predict(self, X):
    Y = self.b0 + self.b1*X
    return Y