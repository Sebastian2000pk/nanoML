from nanoml.linear_model import LinearRegression
import numpy as np
import pytest


simple_linear_data = (np.array([[1], [2], [3], [4], [5]]), np.array([2, 4, 6, 8, 10]))

class TestLinearRegression:
  def test_initialization(self):
    model = LinearRegression()
    assert model.coef_ is None
    assert model.intercept_ is None

  def test_fit_with_valid_data(self):
    X, y = simple_linear_data
    model = LinearRegression()
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None

  def test_predict_output_shape(self):
    X, y = simple_linear_data
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape

  def test_fit_with_invalid_data(self):
    model = LinearRegression()
    with pytest.raises(ValueError):
      model.fit(None, None)

  def test_fit_predict(self):
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    expected_coef = np.array([2.0])
    expected_intercept = 0.0

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    assert np.allclose(predictions, y), f"Expected {y}, but got {predictions}"
    assert np.allclose(model.coef_, expected_coef), "Coeficiente incorrecto"
    assert np.allclose(model.intercept_, expected_intercept), "Intersección incorrecta" 
    assert model.coef_.shape == (1,), f"Forma incorrecta: {model.coef_.shape}"