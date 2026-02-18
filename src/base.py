class BaseEstimator:
  def fit(self, X, y=None):
    raise NotImplementedError("fit method not implemented")
  
  def predict(self, X):
    raise NotImplementedError("predict method not implemented")