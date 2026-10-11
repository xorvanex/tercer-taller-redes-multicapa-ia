# -*- coding: utf-8 -*-
"""
Punto de Entrada Principal (Main)
Permite la ejecucion del proyecto en modo Interfaz Grafica (GUI) o en modo
Consola (CLI) con exportacion automatica de graficas y metricas
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Importar modulos internos
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_loader import load_and_preprocess_dataset
from mlp import MultilayerPerceptron
from app_gui import launch_gui


def train_and_export_figures(dataset_path, output_img_dir, epochs=500, lr=0.01):
    """
    Ejecuta el entrenamiento por consola de ambas arquitecturas y guarda las figuras oficiales.

    Args:
        dataset_path (str): Ruta al archivo CSV con los datos.
        output_img_dir (str): Directorio destino para exportar las graficas generadas.
        epochs (int): Cantidad de epocas de entrenamiento.
        lr (float): Tasa de aprendizaje.
    """
    matplotlib.use("Agg")
    os.makedirs(output_img_dir, exist_ok=True)

    print("================================================================================")
    print("PROYECTO: REDES NEURONALES MULTICAPA (MLP) - CALIDAD DEL AIRE (PM2.5)")
    print("Universidad de Cartagena | Inteligencia Artificial (Segundo Corte)")
    print("Dago David Palmera Navarro")
    print("================================================================================")

    # 1. Carga y particion de datos
    print("\n[1/4] Cargando dataset y aplicando normalizacion [-1, 1] sin data leakage...")
    data = load_and_preprocess_dataset(dataset_path)
    print(f"Total registros: {data['n_train'] + data['n_test']:,}")
    print(f"  - Conjunto de Entrenamiento (80%): {data['n_train']:,} muestras")
    print(f"  - Conjunto de Prueba (20%):        {data['n_test']:,} muestras")

    # 2. Entrenamiento de Modelo 1: Referencia (10:10:1)
    print("\n[2/4] Entrenando Modelo 1: Arquitectura de Referencia (10:10:1)...")
    mlp_ref = MultilayerPerceptron(topology=[10, 10, 1], lr=lr, seed=42)
    p_ref = mlp_ref.count_parameters()
    print(f"  - Parametros entrenables: {p_ref['total']}")

    t0 = time.time()
    mlp_ref.fit(
        data["X_train_norm"], data["y_train_norm"],
        epochs=epochs,
        X_test=data["X_test_norm"], y_test=data["y_test_norm"]
    )
    t_ref = time.time() - t0
    ev_ref = mlp_ref.evaluate(data["X_test_norm"], data["y_test_norm"], scaler_y=data["scaler_y"])
    print(f"  - Tiempo de entrenamiento: {t_ref:.2f} s")
    print(f"  - Test MSE (real):  {ev_ref['mse_real']:.2f}")
    print(f"  - Test RMSE (real): {ev_ref['rmse_real']:.2f} ug/m3")

    # 3. Entrenamiento de Modelo 2: Propia Propuesta (10:16:8:1)
    print("\n[3/4] Entrenando Modelo 2: Arquitectura Propia Propuesta (10:16:8:1)...")
    mlp_prop = MultilayerPerceptron(topology=[10, 16, 8, 1], lr=lr, seed=42)
    p_prop = mlp_prop.count_parameters()
    print(f"  - Parametros entrenables: {p_prop['total']}")

    t0 = time.time()
    mlp_prop.fit(
        data["X_train_norm"], data["y_train_norm"],
        epochs=epochs,
        X_test=data["X_test_norm"], y_test=data["y_test_norm"]
    )
    t_prop = time.time() - t0
    ev_prop = mlp_prop.evaluate(data["X_test_norm"], data["y_test_norm"], scaler_y=data["scaler_y"])
    print(f"  - Tiempo de entrenamiento: {t_prop:.2f} s")
    print(f"  - Test MSE (real):  {ev_prop['mse_real']:.2f}")
    print(f"  - Test RMSE (real): {ev_prop['rmse_real']:.2f} ug/m3")

    # 4. Generacion y exportacion de figuras comparativas
    print("\n[4/4] Exportando graficas de alta resolucion a docs/img/...")

    # Figura A: Curvas de Aprendizaje Comparativas
    plt.figure(figsize=(10, 5.5), dpi=300)
    ep_axis = mlp_ref.history["epoch"]
    plt.plot(ep_axis, mlp_ref.history["train_mse"], label="Referencia (10:10:1) - Train MSE", color="#2E75B6", lw=1.8)
    plt.plot(ep_axis, mlp_ref.history["test_mse"], label="Referencia (10:10:1) - Test MSE", color="#2E75B6", lw=1.8, linestyle="--")
    plt.plot(ep_axis, mlp_prop.history["train_mse"], label="Propia (10:16:8:1) - Train MSE", color="#C00000", lw=1.8)
    plt.plot(ep_axis, mlp_prop.history["test_mse"], label="Propia (10:16:8:1) - Test MSE", color="#C00000", lw=1.8, linestyle="--")
    plt.title("Curvas de Aprendizaje Comparativas (MSE normalizado vs. Epocas)", fontsize=12, weight="bold", color="#1F4E79")
    plt.xlabel("Epocas de Entrenamiento", fontsize=10, weight="bold")
    plt.ylabel("Error Cuadratico Medio (MSE normalizado)", fontsize=10, weight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    fig_a_path = os.path.join(output_img_dir, "Curvas_Aprendizaje_Comparativas.png")
    plt.savefig(fig_a_path, bbox_inches="tight")
    plt.close()
    print(f"  - Guardada: {fig_a_path}")

    # Figura B: Comparacion Temporal Real vs Predicha (Primeras 200 horas de Test)
    window = 200
    hours = np.arange(1, window + 1)
    y_real_win = ev_ref["y_real"][:window].flatten()
    y_ref_win = ev_ref["y_pred_real"][:window].flatten()
    y_prop_win = ev_prop["y_pred_real"][:window].flatten()

    plt.figure(figsize=(12, 5.5), dpi=300)
    plt.plot(hours, y_real_win, label="Concentracion Real de PM2.5", color="#1F4E79", lw=2.0)
    plt.plot(hours, y_ref_win, label="Prediccion Referencia (10:10:1)", color="#2E75B6", lw=1.5, linestyle="--")
    plt.plot(hours, y_prop_win, label="Prediccion Propia (10:16:8:1)", color="#C00000", lw=1.5, linestyle="-.")
    plt.title(f"Concentracion Real frente a Predicha en Escala Original (Muestra de {window} Horas en Test)", fontsize=12, weight="bold", color="#1F4E79")
    plt.xlabel("Horas de Prueba Consecutivas", fontsize=10, weight="bold")
    plt.ylabel("Concentracion de PM2.5 (ug/m3)", fontsize=10, weight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    fig_b_path = os.path.join(output_img_dir, "Comparativa_Prediccion_Real_vs_Predicha.png")
    plt.savefig(fig_b_path, bbox_inches="tight")
    plt.close()
    print(f"  - Guardada: {fig_b_path}")

    # Figura C: Grafica de Dispersion (Scatter Real vs Predicho para el modelo propio)
    plt.figure(figsize=(6.5, 6), dpi=300)
    y_real_all = ev_prop["y_real"].flatten()
    y_prop_all = ev_prop["y_pred_real"].flatten()
    plt.scatter(y_real_all, y_prop_all, alpha=0.25, color="#1F4E79", s=12, edgecolors="none")
    max_val = max(np.max(y_real_all), np.max(y_prop_all))
    plt.plot([0, max_val], [0, max_val], color="#C00000", lw=1.8, linestyle="--", label="Prediccion Perfecta (y = x)")
    plt.title("Dispersion: Concentracion Real vs. Predicha (Modelo Propio 10:16:8:1)", fontsize=11, weight="bold", color="#1F4E79")
    plt.xlabel("PM2.5 Real (ug/m3)", fontsize=10, weight="bold")
    plt.ylabel("PM2.5 Predicho (ug/m3)", fontsize=10, weight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    fig_c_path = os.path.join(output_img_dir, "Dispersion_Real_vs_Predicho_Propio.png")
    plt.savefig(fig_c_path, bbox_inches="tight")
    plt.close()
    print(f"  - Guardada: {fig_c_path}")

    # Tabla Resumen Consolidada
    print("\n================================================================================")
    print("TABLA RESUMEN DE RESULTADOS CUANTITATIVOS (CONJUNTO DE PRUEBA: 8.346 MUESTRAS)")
    print("================================================================================")
    tr_mse_ref_str = f"{mlp_ref.history['train_mse'][-1]:.6f}"
    tr_mse_prop_str = f"{mlp_prop.history['train_mse'][-1]:.6f}"
    te_mse_norm_ref_str = f"{ev_ref['mse_norm']:.6f}"
    te_mse_norm_prop_str = f"{ev_prop['mse_norm']:.6f}"
    te_mse_real_ref_str = f"{ev_ref['mse_real']:.2f}"
    te_mse_real_prop_str = f"{ev_prop['mse_real']:.2f}"
    te_rmse_real_ref_str = f"{ev_ref['rmse_real']:.2f} ug/m3"
    te_rmse_real_prop_str = f"{ev_prop['rmse_real']:.2f} ug/m3"
    t_ref_str = f"{t_ref:.2f} s"
    t_prop_str = f"{t_prop:.2f} s"

    print(f"{'Metrica':<32} | {'Modelo Referencia (10:10:1)':<28} | {'Modelo Propio (10:16:8:1)':<26}")
    print("-" * 92)
    print(f"{'Topologia':<32} | {'10:10:1':<28} | {'10:16:8:1':<26}")
    print(f"{'Parametros Entrenables':<32} | {p_ref['total']:<28} | {p_prop['total']:<26}")
    print(f"{'Tiempo de Entrenamiento (500 ep)':<32} | {t_ref_str:<28} | {t_prop_str:<26}")
    print(f"{'MSE Train (Normalizado)':<32} | {tr_mse_ref_str:<28} | {tr_mse_prop_str:<26}")
    print(f"{'MSE Test (Normalizado)':<32} | {te_mse_norm_ref_str:<28} | {te_mse_norm_prop_str:<26}")
    print(f"{'MSE Test (Escala Real)':<32} | {te_mse_real_ref_str:<28} | {te_mse_real_prop_str:<26}")
    print(f"{'RMSE Test (Escala Real ug/m3)':<32} | {te_rmse_real_ref_str:<28} | {te_rmse_real_prop_str:<26}")
    print("================================================================================")


def main():
    """Analiza argumentos de linea de comandos y despacha el modo de ejecucion."""
    parser = argparse.ArgumentParser(
        description="Sistema de Redes Neuronales Multicapa (MLP) - Prediccion de Calidad del Aire PM2.5."
    )
    parser.add_argument("--cli", action="store_true", help="Ejecutar entrenamiento por consola y exportar graficas.")
    parser.add_argument("--gui", action="store_true", help="Iniciar aplicacion de escritorio con interfaz grafica.")
    parser.add_argument("--epochs", type=int, default=500, help="Numero de epocas de entrenamiento (default: 500).")
    parser.add_argument("--lr", type=float, default=0.01, help="Tasa de aprendizaje eta (default: 0.01).")

    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(project_root, "data", "datasets", "dataset_pm25.csv")
    output_img_dir = os.path.join(project_root, "docs", "img")

    if args.cli:
        train_and_export_figures(dataset_path, output_img_dir, epochs=args.epochs, lr=args.lr)
    else:
        # Por defecto inicia la interfaz grafica
        launch_gui()


if __name__ == "__main__":
    main()
