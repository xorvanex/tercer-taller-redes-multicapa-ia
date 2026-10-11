# -*- coding: utf-8 -*-
"""
Modulo de Red Neuronal Multicapa (MLP)
Implementacion vectorizada en NumPy con optimizacion por Descenso de Gradiente,
funciones de activacion tansig en capas ocultas y purelin en capa de salida.
"""

import numpy as np


class MultilayerPerceptron:
    """
    Red Neuronal Multicapa para regresion continua.

    Atributos:
        topology (list): Lista de enteros con el numero de neuronas por capa.
        lr (float): Tasa de aprendizaje eta.
        weights (list): Matrices de pesos por capa.
        biases (list): Vectores de sesgo por capa.
        history (dict): Registro de la perdida MSE por epoca para entrenamiento y prueba.
    """

    def __init__(self, topology, lr=0.01, seed=42):
        """
        Inicializa las matrices de pesos y vectores de sesgo de la red.

        Args:
            topology (list): Dimensiones de las capas [n_in, n_h1, ..., n_out].
            lr (float): Tasa de aprendizaje (default: 0.01).
            seed (int): Semilla para garantizar reproducibilidad (default: 42).
        """
        self.topology = topology
        self.lr = lr
        self.seed = seed
        self.num_layers = len(topology) - 1

        # Inicializacion pseudoaleatoria uniforme en [-0.5, 0.5]
        np.random.seed(self.seed)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            n_in = self.topology[i]
            n_out = self.topology[i + 1]
            w = np.random.uniform(-0.5, 0.5, size=(n_in, n_out))
            b = np.random.uniform(-0.5, 0.5, size=(1, n_out))
            self.weights.append(w)
            self.biases.append(b)

        self.history = {
            "epoch": [],
            "train_mse": [],
            "test_mse": []
        }

    @staticmethod
    def _tansig(z):
        """Funcion de activacion tangente hiperbolica."""
        return np.tanh(z)

    @staticmethod
    def _tansig_derivative(a):
        """Derivada analitica de tansig en terminos de su salida a = tanh(z)."""
        return 1.0 - (a ** 2)

    @staticmethod
    def _purelin(z):
        """Funcion de activacion lineal o identidad."""
        return z

    @staticmethod
    def _purelin_derivative(a):
        """Derivada analitica de purelin."""
        return np.ones_like(a)

    def forward(self, X):
        """
        Propagacion hacia adelante a traves de todas las capas.

        Args:
            X (np.ndarray): Tensor de entrada con dimensiones (N, n_in).

        Returns:
            tuple: (activaciones, net_inputs) listas con los tensores por capa.
        """
        activations = [X]
        net_inputs = []

        current_a = X
        for i in range(self.num_layers):
            z = np.dot(current_a, self.weights[i]) + self.biases[i]
            net_inputs.append(z)

            # Capas ocultas usan tansig, capa de salida usa purelin
            if i < self.num_layers - 1:
                current_a = self._tansig(z)
            else:
                current_a = self._purelin(z)

            activations.append(current_a)

        return activations, net_inputs

    def backward(self, activations, y_true):
        """
        Retropropagacion del error y computo de gradientes mediante regla de la cadena.

        Args:
            activations (list): Lista de activaciones por capa calculadas en el forward pass.
            y_true (np.ndarray): Valores objetivo reales de dimension (N, 1).

        Returns:
            tuple: (grad_weights, grad_biases) listas con los gradientes por capa.
        """
        n_samples = y_true.shape[0]
        y_pred = activations[-1]

        grad_w = [None] * self.num_layers
        grad_b = [None] * self.num_layers

        # Gradiente en la capa de salida (error cuadratico medio con purelin)
        # dJ/dz_out = -2/N * (y_true - y_pred) * f'(z_out)
        error = y_true - y_pred
        delta = -(2.0 / n_samples) * error * self._purelin_derivative(y_pred)

        grad_w[-1] = np.dot(activations[-2].T, delta)
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Retropropagacion a traves de capas ocultas
        current_delta = delta
        for i in range(self.num_layers - 2, -1, -1):
            w_next = self.weights[i + 1]
            a_curr = activations[i + 1]
            # delta_l = (delta_{l+1} * W_{l+1}^T) .* tansig'(a_curr)
            current_delta = np.dot(current_delta, w_next.T) * self._tansig_derivative(a_curr)

            grad_w[i] = np.dot(activations[i].T, current_delta)
            grad_b[i] = np.sum(current_delta, axis=0, keepdims=True)

        return grad_w, grad_b

    def train_epoch(self, X_train, y_train):
        """
        Ejecuta una epoca de optimizacion por lote completo (full-batch).

        Args:
            X_train (np.ndarray): Tensor de entrenamiento normalizado.
            y_train (np.ndarray): Vector objetivo normalizado.

        Returns:
            float: Error cuadratico medio (MSE) de la epoca sobre train.
        """
        activations, _ = self.forward(X_train)
        grad_w, grad_b = self.backward(activations, y_train)

        # Actualizacion de pesos y sesgos por descenso del gradiente
        for i in range(self.num_layers):
            self.weights[i] -= self.lr * grad_w[i]
            self.biases[i] -= self.lr * grad_b[i]

        y_pred = activations[-1]
        train_mse = float(np.mean((y_train - y_pred) ** 2))
        return train_mse

    def fit(self, X_train, y_train, epochs=500, X_test=None, y_test=None, callback=None):
        """
        Ciclo completo de entrenamiento supervisado.

        Args:
            X_train (np.ndarray): Entradas de entrenamiento.
            y_train (np.ndarray): Salidas esperadas de entrenamiento.
            epochs (int): Cantidad de epocas de entrenamiento.
            X_test (np.ndarray, opcional): Entradas de prueba para monitoreo.
            y_test (np.ndarray, opcional): Salidas esperadas de prueba.
            callback (callable, opcional): Funcion invocada por epoca con firma callback(epoch, tr_mse, te_mse).
        """
        for ep in range(1, epochs + 1):
            tr_mse = self.train_epoch(X_train, y_train)

            te_mse = None
            if X_test is not None and y_test is not None:
                te_pred = self.predict(X_test)
                te_mse = float(np.mean((y_test - te_pred) ** 2))

            self.history["epoch"].append(ep)
            self.history["train_mse"].append(tr_mse)
            self.history["test_mse"].append(te_mse)

            if callback is not None:
                callback(ep, tr_mse, te_mse)

        return self.history

    def predict(self, X):
        """
        Genera predicciones propagando entradas por la red.

        Args:
            X (np.ndarray): Tensor de entradas.

        Returns:
            np.ndarray: Salida continua predicha.
        """
        activations, _ = self.forward(X)
        return activations[-1]

    def evaluate(self, X, y, scaler_y=None):
        """
        Calcula metricas de desempeno (MSE y RMSE), con opcion de desnormalizar.

        Args:
            X (np.ndarray): Entradas normalizadas.
            y (np.ndarray): Salidas objetivo normalizadas.
            scaler_y (MinMaxSymScaler, opcional): Escalador para metricas en escala original.

        Returns:
            dict: Metricas MSE y RMSE en escala normalizada y escala real.
        """
        y_pred_norm = self.predict(X)
        mse_norm = float(np.mean((y - y_pred_norm) ** 2))
        rmse_norm = float(np.sqrt(mse_norm))

        results = {
            "mse_norm": mse_norm,
            "rmse_norm": rmse_norm,
            "y_pred_norm": y_pred_norm
        }

        if scaler_y is not None:
            y_real = scaler_y.inverse_transform(y)
            y_pred_real = scaler_y.inverse_transform(y_pred_norm)

            mse_real = float(np.mean((y_real - y_pred_real) ** 2))
            rmse_real = float(np.sqrt(mse_real))

            results["mse_real"] = mse_real
            results["rmse_real"] = rmse_real
            results["y_real"] = y_real
            results["y_pred_real"] = y_pred_real

        return results

    def count_parameters(self):
        """
        Calcula analiticamente la cantidad de parametros libres de la red.

        Returns:
            dict: Desglose capa por capa y total general de parametros entrenables.
        """
        layer_details = []
        total_params = 0

        for i in range(self.num_layers):
            n_in = self.topology[i]
            n_out = self.topology[i + 1]
            n_weights = n_in * n_out
            n_biases = n_out
            subtotal = n_weights + n_biases
            total_params += subtotal

            layer_name = f"Capa Oculta {i + 1}" if i < self.num_layers - 1 else "Capa de Salida"
            layer_details.append({
                "layer": layer_name,
                "n_in": n_in,
                "n_out": n_out,
                "weights": n_weights,
                "biases": n_biases,
                "subtotal": subtotal
            })

        return {
            "layers": layer_details,
            "total": total_params
        }
