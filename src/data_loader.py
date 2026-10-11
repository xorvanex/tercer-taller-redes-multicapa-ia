# -*- coding: utf-8 -*-
"""
Modulo de Carga y Preprocesamiento de Datos
Gestiona la lectura del dataset de Beijing PM2.5, la particion temporal
y la normalizacion al rango [-1, 1] sin fuga de informacion (data leakage).
"""

import os
import numpy as np
import pandas as pd


class MinMaxSymScaler:
    """
    Escalador Min-Max simetrico al rango [-1, 1].
    Mapea linealmente los datos de entrada preservando la simetria respecto al origen.
    """

    def __init__(self):
        """Inicializa los vectores de minimos y maximos."""
        self.min_val = None
        self.max_val = None

    def fit(self, data):
        """
        Calcula los valores minimos y maximos exclusivamente sobre el conjunto de entrenamiento.

        Args:
            data (np.ndarray): Tensor de caracteristicas de entrenamiento.
        """
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        # Evitar division por cero si alguna columna tiene rango constante
        range_val = self.max_val - self.min_val
        range_val[range_val == 0.0] = 1.0
        self.range_val = range_val
        return self

    def transform(self, data):
        """
        Transforma los datos al intervalo [-1, 1] usando los parametros calibrados.

        Args:
            data (np.ndarray): Tensor de caracteristicas a escalar.

        Returns:
            np.ndarray: Datos normalizados en [-1, 1].
        """
        return 2.0 * ((data - self.min_val) / self.range_val) - 1.0

    def fit_transform(self, data):
        """Ajusta y transforma en una sola llamada."""
        return self.fit(data).transform(data)

    def inverse_transform(self, norm_data):
        """
        Aplica la transformacion inversa para regresar a las unidades fisicas originales.

        Args:
            norm_data (np.ndarray): Datos en la escala [-1, 1].

        Returns:
            np.ndarray: Datos en su escala original.
        """
        return ((norm_data + 1.0) / 2.0) * self.range_val + self.min_val


def load_and_preprocess_dataset(csv_path, train_ratio=0.8):
    """
    Carga el archivo CSV, realiza particion cronologica y normaliza en [-1, 1].

    Args:
        csv_path (str): Ruta al archivo dataset_pm25.csv.
        train_ratio (float): Proporcion de datos para entrenamiento (por defecto 0.8).

    Returns:
        dict: Diccionario estructurado con subconjuntos, escaladores y nombres de variables.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El archivo especificado no existe: {csv_path}")

    df = pd.read_csv(csv_path)

    # Separar entradas y variable objetivo
    feature_cols = [
        "TEMP", "PRES", "DEWP", "Iws", "hour", "day",
        "pm25_t", "pm25_t-1", "pm25_t-2", "pm25_t-3"
    ]
    target_col = "pm25_t+24"

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[[target_col]].to_numpy(dtype=np.float64)

    total_samples = len(df)
    n_train = int(total_samples * train_ratio)

    # Particion estrictamente cronologica (sin reordenamiento aleatorio)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    # Normalizacion de caracteristicas (fit solo en train)
    scaler_X = MinMaxSymScaler()
    scaler_X.fit(X_train)
    X_train_norm = scaler_X.transform(X_train)
    X_test_norm = scaler_X.transform(X_test)

    # Normalizacion de la variable objetivo (fit solo en train)
    scaler_y = MinMaxSymScaler()
    scaler_y.fit(y_train)
    y_train_norm = scaler_y.transform(y_train)
    y_test_norm = scaler_y.transform(y_test)

    return {
        "X_train_raw": X_train,
        "y_train_raw": y_train,
        "X_test_raw": X_test,
        "y_test_raw": y_test,
        "X_train_norm": X_train_norm,
        "y_train_norm": y_train_norm,
        "X_test_norm": X_test_norm,
        "y_test_norm": y_test_norm,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "feature_names": feature_cols,
        "target_name": target_col,
        "n_train": n_train,
        "n_test": total_samples - n_train
    }
