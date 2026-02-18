from nanoml.linear_model import LinearRegression
import numpy as np

class TestLinearRegression:
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