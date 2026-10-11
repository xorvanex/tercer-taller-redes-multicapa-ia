# 9. CONCLUSIONES

---

1. **Efectividad del Perceptrón Multicapa frente a la No Linealidad Ambiental:**  
   El desarrollo experimental demostró la aplicabilidad práctica del Teorema Universal de Aproximación en problemas complejos de pronóstico ambiental. A diferencia del Perceptrón Simple —cuya frontera de decisión lineal resulta incapaz de modelar la interacción atmosférica multidimensional—, las redes neuronales multicapa implementadas aprendieron con éxito el mapeo no lineal entre variables meteorológicas, índices temporales y retardos autorregresivos, logrando predecir la concentración de PM2.5 con 24 horas de antelación.

2. **Ventaja Empírica de la Topología Propia en Embudo (10:16:8:1):**  
   La arquitectura diseñada con dos capas ocultas en estructura de compresión progresiva superó de manera concluyente al modelo de referencia de una sola capa (10:10:1). El modelo propio logró una reducción del **8,75% en el Error Cuadrático Medio de prueba** (de 8.529,83 a 7.782,96) y disminuyó el RMSE en **4,14 μg/m³** (alcanzando 88,22 μg/m³ frente a 92,36 μg/m³ de la referencia). La expansión inicial a 16 neuronas facilitó el desacoplamiento de relaciones no lineales, mientras que la compresión subsiguiente a 8 neuronas actuó como un filtro de abstracción que optimizó la capacidad predictiva.

3. **Inmunidad al Sobreajuste mediante Arquitectura Controlada:**  
   A pesar de incrementar la cantidad de parámetros libres de 121 a 321 (un factor de 2,65×), el modelo propio no presentó síntomas de sobreajuste (*overfitting*). La paridad casi exacta entre el MSE normalizado de entrenamiento (0,031615) y el MSE de prueba (0,031509), sumada a la estabilidad descendente de la curva de prueba a lo largo de las 500 épocas, valida que la topología en embudo cumplió una función regularizadora implícita, preservando la capacidad de generalización sobre observaciones nunca vistas.

4. **Trascendencia del Rigor Metodológico en Series Temporales:**  
   La partición estrictamente cronológica por corte de índice (80% pasado / 20% futuro) y el ajuste de los parámetros de normalización Min-Max exclusivamente sobre el conjunto de entrenamiento demostraron ser indispensables para garantizar la validez científica del trabajo. La adopción de estos principios eliminó por completo la fuga de información temporal (*data leakage*), asegurando que las métricas obtenidas reflejen el desempeño operativo real del modelo en un escenario de monitoreo continuo.

5. **Alineación Física y Jerarquía de Variables:**  
   El comportamiento del modelo confirmó los principios del transporte de contaminantes atmosféricos. Los retardos autorregresivos inmediatos (`pm25_t` a `pm25_t-3`) establecieron la inercia fundamental de la serie, mientras que la velocidad acumulada del viento (`Iws`) operó como el principal agente dispersor inhibitorio y el punto de rocío (`DEWP`) como modulador de acumulación por humedad.

6. **Eficiencia de la Implementación Vectorizada:**  
   La construcción del algoritmo mediante operaciones tensoriales vectorizadas en NumPy acreditó una alta eficiencia computacional: el entrenamiento completo de 500 épocas sobre 33.384 observaciones se ejecutó en 6,79 segundos para el modelo de referencia y en 21,13 segundos para el modelo propio, demostrando que el descenso de gradiente por lote completo (*full-batch*) es plenamente viable y numéricamente estable para este volumen de datos.

---

---

# 10. REFERENCIAS BIBLIOGRÁFICAS

---

* Cybenko, G. (1989). *Approximation by superpositions of a sigmoidal function*. Mathematics of Control, Signals and Systems, 2(4), 303-314. https://doi.org/10.1007/BF02551274
* Hornik, K. (1991). *Approximation capabilities of multilayer feedforward networks*. Neural Networks, 4(2), 251-257. https://doi.org/10.1016/0893-6080(91)90009-T
* Liang, X., Zou, T., Guo, B., Zhang, S., Zhang, H., Sheng, S., Chen, P., & Chen, S. X. (2015). *Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating*. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 471(2182), 20150257. https://doi.org/10.1098/rspa.2015.0257
* Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. The MIT Press, Cambridge, MA.
* Organización Mundial de la Salud (OMS). (2021). *Directrices mundiales de la OMS sobre la calidad del aire: partículas en suspensión (PM2.5 y PM10), ozono, dióxido de nitrógeno, dióxido de azufre y monóxido de carbono*. Ginebra: OMS.
* Rosenblatt, F. (1958). *The perceptron: A probabilistic model for information storage and organization in the brain*. Psychological Review, 65(6), 386-408. https://doi.org/10.1037/h0042519
* Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature, 323(6088), 533-536. https://doi.org/10.1038/323533a0

---

---

# 11. ANEXOS

---

## 11.1. Anexo A. Código Fuente Completo del Script de Entrenamiento

A continuación se presenta la implementación modular del modelo `MultilayerPerceptron` (`src/mlp.py`):

```python
# -*- coding: utf-8 -*-
"""
Implementacion vectorizada del Perceptron Multicapa (MLP)
con Backpropagation analitico y funciones tansig / purelin.
"""

import numpy as np


class MultilayerPerceptron:
    def __init__(self, topology, lr=0.01, seed=42):
        self.topology = topology
        self.lr = lr
        self.num_layers = len(topology) - 1

        np.random.seed(seed)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            n_in = self.topology[i]
            n_out = self.topology[i + 1]
            w = np.random.uniform(-0.5, 0.5, size=(n_in, n_out))
            b = np.random.uniform(-0.5, 0.5, size=(1, n_out))
            self.weights.append(w)
            self.biases.append(b)

        self.history = {"epoch": [], "train_mse": [], "test_mse": []}

    def forward(self, X):
        activations = [X]
        net_inputs = []
        current_a = X
        for i in range(self.num_layers):
            z = np.dot(current_a, self.weights[i]) + self.biases[i]
            net_inputs.append(z)
            current_a = np.tanh(z) if i < self.num_layers - 1 else z
            activations.append(current_a)
        return activations, net_inputs

    def backward(self, activations, y_true):
        n_samples = y_true.shape[0]
        y_pred = activations[-1]
        grad_w = [None] * self.num_layers
        grad_b = [None] * self.num_layers

        # Gradiente en la salida (purelin con MSE)
        delta = -(2.0 / n_samples) * (y_true - y_pred)
        grad_w[-1] = np.dot(activations[-2].T, delta)
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Retropropagacion a capas ocultas (tansig)
        current_delta = delta
        for i in range(self.num_layers - 2, -1, -1):
            w_next = self.weights[i + 1]
            a_curr = activations[i + 1]
            current_delta = np.dot(current_delta, w_next.T) * (1.0 - a_curr ** 2)
            grad_w[i] = np.dot(activations[i].T, current_delta)
            grad_b[i] = np.sum(current_delta, axis=0, keepdims=True)

        return grad_w, grad_b

    def train_epoch(self, X_train, y_train):
        activations, _ = self.forward(X_train)
        grad_w, grad_b = self.backward(activations, y_train)
        for i in range(self.num_layers):
            self.weights[i] -= self.lr * grad_w[i]
            self.biases[i] -= self.lr * grad_b[i]
        y_pred = activations[-1]
        return float(np.mean((y_train - y_pred) ** 2))

    def fit(self, X_train, y_train, epochs=500, X_test=None, y_test=None):
        for ep in range(1, epochs + 1):
            tr_mse = self.train_epoch(X_train, y_train)
            te_mse = None
            if X_test is not None and y_test is not None:
                te_pred = self.predict(X_test)
                te_mse = float(np.mean((y_test - te_pred) ** 2))
            self.history["epoch"].append(ep)
            self.history["train_mse"].append(tr_mse)
            self.history["test_mse"].append(te_mse)
        return self.history

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def evaluate(self, X, y, scaler_y=None):
        y_pred_norm = self.predict(X)
        mse_norm = float(np.mean((y - y_pred_norm) ** 2))
        rmse_norm = float(np.sqrt(mse_norm))
        res = {"mse_norm": mse_norm, "rmse_norm": rmse_norm, "y_pred_norm": y_pred_norm}
        if scaler_y is not None:
            y_real = scaler_y.inverse_transform(y)
            y_pred_real = scaler_y.inverse_transform(y_pred_norm)
            res["mse_real"] = float(np.mean((y_real - y_pred_real) ** 2))
            res["rmse_real"] = float(np.sqrt(res["mse_real"]))
            res["y_real"] = y_real
            res["y_pred_real"] = y_pred_real
        return res
```

---

## 11.2. Anexo B. Configuración de Hiperparámetros y Parámetros Finales Calibrados

| Parámetro / Condición | Configuración Modelo de Referencia | Configuración Modelo Propio |
| :--- | :---: | :---: |
| Topología Completa | $10 \to 10 \to 1$ | $10 \to 16 \to 8 \to 1$ |
| Número de Capas Ocultas | 1 | 2 |
| Tasa de Aprendizaje ($\eta$) | 0,01 | 0,01 |
| Número Total de Épocas | 500 | 500 |
| Modo de Descenso de Gradiente | Lote Completo (*Full-Batch*) | Lote Completo (*Full-Batch*) |
| Tamaño de Lote ($N$) | 33.384 muestras | 33.384 muestras |
| Distribución de Inicialización | Uniforme $\mathcal{U}(-0,5; \, 0,5)$ | Uniforme $\mathcal{U}(-0,5; \, 0,5)$ |
| Semilla Pseudoaleatoria | 42 | 42 |
| Función en Capa Oculta 1 | `tansig` | `tansig` |
| Función en Capa Oculta 2 | — | `tansig` |
| Función en Capa de Salida | `purelin` | `purelin` |
| Cantidad de Pesos Libres | 110 | 296 |
| Cantidad de Sesgos Libres | 11 | 25 |
| **Total Coeficientes Sinápticos** | **121** | **321** |

---

## 11.3. Anexo C. Registro Numérico Extendido de la Evolución del Error

A continuación se registra la evolución cuantitativa del Error Cuadrático Medio normalizado (MSE) extraído en épocas de control representativas:

| Época ($k$) | Train MSE (Ref 10:10:1) | Test MSE (Ref 10:10:1) | Train MSE (Prop 10:16:8:1) | Test MSE (Prop 10:16:8:1) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0,579984 | 0,446699 | 0,591240 | 0,453180 |
| 25 | 0,051280 | 0,046920 | 0,047320 | 0,043810 |
| 50 | 0,044300 | 0,040770 | 0,039840 | 0,037250 |
| 100 | 0,039912 | 0,037410 | 0,035820 | 0,034410 |
| 200 | 0,037210 | 0,035720 | 0,033410 | 0,032740 |
| 300 | 0,036180 | 0,034980 | 0,032390 | 0,031980 |
| 400 | 0,035710 | 0,034710 | 0,031880 | 0,031670 |
| **500 (Final)** | **0,035462** | **0,034532** | **0,031615** | **0,031509** |
