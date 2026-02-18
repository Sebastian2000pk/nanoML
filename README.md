# nanoML 🤖

Una librería de Machine Learning minimalista construida desde cero para aprender y entender profundamente cómo funcionan los algoritmos de ML. El objetivo final es evolucionar hacia la construcción de un LLM desde los fundamentos.

## Visión del Proyecto

**nanoML** es un proyecto educativo diseñado para:

- 📚 Comprender cada concepto de Machine Learning desde sus bases matemáticas
- 🧠 Implementar algoritmos clásicos sin dependencias externas (excepto NumPy)
- 🎯 Construir gradualmente hacia modelos más complejos
- 🚀 Eventualmente crear un LLM (Large Language Model) completo desde cero

Este enfoque contrasta con usar librerías como scikit-learn o TensorFlow directamente, permitiéndote ver exactamente qué sucede en cada paso.

## Roadmap 🗺️

### Fase 1: Regresión (En Progreso)
- [x] Regresión Lineal Simple
- [ ] Regresión Lineal Multivariable
- [ ] Regresión Polinómica
- [ ] Regularización (Ridge, Lasso)

### Fase 2: Clasificación
- [ ] Regresión Logística
- [ ] Árboles de Decisión
- [ ] Random Forest
- [ ] SVM (Support Vector Machines)

### Fase 3: Clustering
- [ ] K-Means
- [ ] DBSCAN
- [ ] Hierarchical Clustering

### Fase 4: Redes Neuronales
- [ ] Perceptrón Simple
- [ ] Red Neuronal Feedforward
- [ ] Backpropagation desde cero
- [ ] Convolutional Neural Networks (CNN)
- [ ] Recurrent Neural Networks (RNN)

### Fase 5: Procesamiento de Lenguaje Natural
- [ ] Tokenización
- [ ] Word Embeddings
- [ ] Transformers
- [ ] Attention Mechanism
- [ ] LLM Foundation Model

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Sebastian2000pk/nanoML.git
cd nanoML

# Crear un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements-dev.txt
```

## Requisitos

- Python 3.8+
- NumPy
- Pytest (para testing)

## Uso Actual

### Regresión Lineal Simple

```python
from nanoml.linear_model import LinearRegression
import numpy as np

# Datos de ejemplo
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X, y)

# Realizar predicciones
predictions = model.predict(X)
print(f"Coeficiente: {model.coef_}")
print(f"Intercepción: {model.intercept_}")
print(f"Predicciones: {predictions}")
```

## Testing

```bash
# Ejecutar todos los tests
pytest

# Con cobertura
pytest --cov=nanoml tests/

# Modo verbose
pytest -v
```

## Estructura del Proyecto

```
nanoML/
├── nanoml/
│   ├── __init__.py
│   ├── base.py                 # Clase base para estimadores
│   ├── linear_model.py         # Modelos de regresión
│   └── ...más módulos...
├── tests/
│   ├── __init__.py
│   ├── test_linear_regression.py
│   └── ...más tests...
├── README.md
├── requirements-dev.txt
└── pytest.ini
```

## Principios de Desarrollo

1. **Claridad sobre Optimización**: El código debe ser comprensible, no necesariamente el más rápido
2. **Documentación Matemática**: Cada algoritmo incluye su derivación matemática
3. **Testing Exhaustivo**: Cobertura alta de tests para cada implementación
4. **Sin Magia Negra**: Cada línea debe ser comprensible y explicable
5. **Implementación Educativa**: Comentarios explicativos en code crítico

## Contribuciones y Aprendizaje

Este es un proyecto personal de aprendizaje. El código está escrito para ser:

- Didáctico
- Modular
- Extensible
- Bien documentado

## Conceptos Clave a Aprender

📖 Por cada algoritmo, documentamos:
- La intuición matemática detrás
- La derivación paso a paso
- Ejemplos de uso
- Limitaciones y casos de borde
- Comparación con implementaciones estándar

## Próximos Pasos

1. Completar regresión lineal multivariable
2. Implementar validación cruzada (cross-validation)
3. Agregar métricas de evaluación (MSE, R², etc.)
4. Construir pipeline básico de procesamiento
5. Iniciar con redes neuronales simples

## Licencia

MIT

## Autor

Desarrollado como proyecto de aprendizaje en ML Foundational.

---

**"El mejor profesor de Machine Learning es construirlo desde cero"**
