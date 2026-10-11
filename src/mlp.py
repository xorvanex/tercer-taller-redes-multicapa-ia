# -*- coding: utf-8 -*-
"""
Modulo de Red Neuronal Multicapa (MLP)
Implementacion vectorizada en NumPy optimizada con BLAS (@),
retropropagacion analitica del error (Backpropagation),
funciones de activacion tansig en capas ocultas y purelin en capa de salida,
con soporte de tasa de aprendizaje (η) y termino de momento (α).
"""

import numpy as np


class MultilayerPerceptron:
    """
    Red Neuronal Multicapa (MLP) para aproximacion y regresion de funciones continuas.

    Atributos:
        topology (list): Dimensiones de las capas [n_in, n_h1, ..., n_out].
        lr (float): Tasa de aprendizaje η (eta).
        momentum (float): Coeficiente de momento α (alpha).
        seed (int): Semilla pseudoaleatoria para reproducibilidad.
        weights (list): Matrices de pesos sinapticos W^(l) por cada capa l.
        biases (list): Vectores de sesgo o polarizacion b^(l) por cada capa l.
        v_weights (list): Velocidades de momento para pesos sinapticos.
        v_biases (list): Velocidades de momento para sesgos.
        history (dict): Registro del Error Cuadratico Medio (EMC / MSE) por epoca.
    """

    def __init__(self, topology, lr=0.01, momentum=0.0, seed=42):
        """
        Inicializa las matrices de pesos sinapticos y sesgos de la red.

        Args:
            topology (list): Dimensiones de las capas [n_in, n_h1, ..., n_out].
            lr (float): Tasa de aprendizaje η (default: 0.01).
            momentum (float): Coeficiente de momento α (default: 0.0).
            seed (int): Semilla para garantizar reproducibilidad exacta (default: 42).
        """
        self.topology = list(topology)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.seed = int(seed)
        self.num_layers = len(topology) - 1

        # Inicializacion pseudoaleatoria uniforme U(-0.5, 0.5)
        np.random.seed(self.seed)
        self.weights = []
        self.biases = []
        self.v_weights = []
        self.v_biases = []

        for i in range(self.num_layers):
            n_in = self.topology[i]
            n_out = self.topology[i + 1]
            w = np.random.uniform(-0.5, 0.5, size=(n_in, n_out))
            b = np.random.uniform(-0.5, 0.5, size=(1, n_out))
            self.weights.append(w)
            self.biases.append(b)
            self.v_weights.append(np.zeros_like(w))
            self.v_biases.append(np.zeros_like(b))

        self.history = {
            "epoch": [],
            "train_mse": [],
            "test_mse": []
        }

    @staticmethod
    def _tansig(z):
        """
        Funcion de activacion sigmoidal bipolar / tangente hiperbolica:
        tansig(z) = (2 / (1 + exp(-2z))) - 1 = tanh(z)
        """
        return np.tanh(z)

    @staticmethod
    def _tansig_derivative(a):
        """
        Derivada analitica de tansig respecto a su activacion a:
        f'(z) = 1 - (tansig(z))^2 = 1 - a^2
        """
        return 1.0 - (a ** 2)

    @staticmethod
    def _purelin(z):
        """
        Funcion de activacion lineal o identidad para la capa de salida:
        purelin(z) = z
        """
        return z

    @staticmethod
    def _purelin_derivative(a):
        """Derivada analitica de purelin: f'(z) = 1."""
        return np.ones_like(a)

    def forward(self, X):
        """
        Propagacion hacia adelante (Forward Pass) vectorizada de alta velocidad.

        Args:
            X (np.ndarray): Tensor de entrada de dimension (N, n_in).

        Returns:
            tuple: (activaciones, net_inputs) tensores por cada capa de la red.
        """
        activations = [X]
        net_inputs = []

        current_a = X
        for i in range(self.num_layers):
            # Computo lineal z = a * W + b optimizado con BLAS
            z = current_a @ self.weights[i] + self.biases[i]
            net_inputs.append(z)

            # Capas intermedias aplican tansig; capa de salida aplica purelin
            if i < self.num_layers - 1:
                current_a = self._tansig(z)
            else:
                current_a = self._purelin(z)

            activations.append(current_a)

        return activations, net_inputs

    def backward(self, activations, y_true):
        """
        Retropropagacion del error (Backward Pass) computando gradientes analiticos exactos.

        Args:
            activations (list): Lista de tensores de activacion por capa del forward pass.
            y_true (np.ndarray): Valores objetivo reales de dimension (N, 1).

        Returns:
            tuple: (grad_w, grad_b) listas de matrices y vectores de gradientes por capa.
        """
        n_samples = y_true.shape[0]
        y_pred = activations[-1]

        grad_w = [None] * self.num_layers
        grad_b = [None] * self.num_layers

        # Gradiente en la capa de salida L (Error Cuadratico Medio con purelin)
        # delta^(L) = -2/N * (y_true - y_pred) * purelin'(y_pred)
        error = y_true - y_pred
        delta = -(2.0 / n_samples) * error

        grad_w[-1] = activations[-2].T @ delta
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Retropropagacion a traves de capas ocultas (tansig)
        current_delta = delta
        for i in range(self.num_layers - 2, -1, -1):
            w_next = self.weights[i + 1]
            a_curr = activations[i + 1]
            # delta^(l) = (delta^(l+1) @ W^(l+1).T) * (1 - (a^(l))^2)
            current_delta = (current_delta @ w_next.T) * self._tansig_derivative(a_curr)

            grad_w[i] = activations[i].T @ current_delta
            grad_b[i] = np.sum(current_delta, axis=0, keepdims=True)

        return grad_w, grad_b

    def train_epoch(self, X_train, y_train):
        """
        Ejecuta una epoca de optimizacion por lote completo con soporte opcional de momento (α).

        Args:
            X_train (np.ndarray): Tensor de entrada de entrenamiento normalizado.
            y_train (np.ndarray): Vector objetivo normalizado.

        Returns:
            float: Error Cuadratico Medio (EMC / MSE) de entrenamiento en la epoca.
        """
        activations, _ = self.forward(X_train)
        grad_w, grad_b = self.backward(activations, y_train)

        # Regla de actualizacion con momento:
        # V = alpha * V + eta * grad
        # W = W - V
        for i in range(self.num_layers):
            if self.momentum > 0.0:
                self.v_weights[i] = self.momentum * self.v_weights[i] + self.lr * grad_w[i]
                self.v_biases[i] = self.momentum * self.v_biases[i] + self.lr * grad_b[i]
                self.weights[i] -= self.v_weights[i]
                self.biases[i] -= self.v_biases[i]
            else:
                self.weights[i] -= self.lr * grad_w[i]
                self.biases[i] -= self.lr * grad_b[i]

        y_pred = activations[-1]
        train_mse = float(np.mean((y_train - y_pred) ** 2))
        return train_mse

    def fit(self, X_train, y_train, epochs=500, X_test=None, y_test=None, callback=None):
        """
        Ciclo integral de entrenamiento supervisado con monitoreo en tiempo real.

        Args:
            X_train (np.ndarray): Entradas de entrenamiento.
            y_train (np.ndarray): Salidas objetivo de entrenamiento.
            epochs (int): Cantidad de epocas de entrenamiento.
            X_test (np.ndarray, opcional): Entradas de prueba para monitoreo.
            y_test (np.ndarray, opcional): Salidas esperadas de prueba.
            callback (callable, opcional): Funcion callback(ep, tr_mse, te_mse) por epoca.

        Returns:
            dict: Historial de epocas y errores cuadraticos medios.
        """
        has_test = X_test is not None and y_test is not None

        for ep in range(1, epochs + 1):
            tr_mse = self.train_epoch(X_train, y_train)

            te_mse = None
            if has_test:
                # Inferencia rapida sin registrar activaciones intermedias
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
        Propaga entradas por la red para generar predicciones continuas.

        Args:
            X (np.ndarray): Tensor de entradas.

        Returns:
            np.ndarray: Salida continua predicha.
        """
        current_a = X
        for i in range(self.num_layers):
            z = current_a @ self.weights[i] + self.biases[i]
            if i < self.num_layers - 1:
                current_a = self._tansig(z)
            else:
                current_a = self._purelin(z)
        return current_a

    def evaluate(self, X, y, scaler_y=None):
        """
        Calcula metricas de desempeno (EMC / MSE y REMC / RMSE), con conversion a escala real.

        Args:
            X (np.ndarray): Entradas normalizadas.
            y (np.ndarray): Salidas objetivo normalizadas.
            scaler_y (MinMaxSymScaler, opcional): Escalador para desnormalizar a µg/m³.

        Returns:
            dict: Metricas MSE y RMSE en escala normalizada y escala real fisica.
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
            dict: Desglose capa por capa y total general de coeficientes sinapticos entrenables.
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
