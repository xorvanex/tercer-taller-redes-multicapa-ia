# -*- coding: utf-8 -*-
"""
Modulo de Interfaz Grafica de Usuario (GUI)
Aplicacion de escritorio interactiva con Tkinter y Matplotlib para el diseno,
visualizacion topologica en grafo, entrenamiento y simulacion de redes MLP.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Matplotlib para integracion interactiva en Tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Importar modulos del proyecto
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_loader import load_and_preprocess_dataset
from mlp import MultilayerPerceptron


class MlpAppGUI:
    """
    Aplicacion interactiva para el analisis y evaluacion comparativa de redes MLP.
    Permite visualizar la red como un grafo topologico, monitorizar el entrenamiento,
    inspeccionar metricas en tiempo real y simular predicciones individuales.
    """

    def __init__(self, root):
        """
        Inicializa la ventana principal y los paneles del sistema.

        Args:
            root (tk.Tk): Instancia de la ventana raiz de Tkinter.
        """
        self.root = root
        self.root.title("Sistema de Redes Neuronales Multicapa (MLP) - Prediccion PM2.5")
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)

        # Estado del sistema
        self.data = None
        self.model_ref = None
        self.model_prop = None
        self.is_training = False

        # Configurar estilos visuales
        self._setup_styles()

        # Construir interfaz
        self._build_header()
        self._build_notebook()

        # Cargar dataset en segundo plano
        self._load_data_thread()

    def _setup_styles(self):
        """Configura la paleta de colores y estilos tipograficos."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.color_primary = "#1F4E79"
        self.color_secondary = "#2E75B6"
        self.color_bg = "#F2F4F7"
        self.color_card = "#FFFFFF"

        self.style.configure(".", background=self.color_bg, font=("Segoe UI", 10))
        self.style.configure("TNotebook", background=self.color_bg)
        self.style.configure("TNotebook.Tab", padding=[16, 8], font=("Segoe UI", 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", self.color_primary)], foreground=[("selected", "#FFFFFF")])
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=self.color_primary)
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), foreground="#FFFFFF", background=self.color_primary)
        self.style.map("Action.TButton", background=[("active", self.color_secondary)])

    def _build_header(self):
        """Construye la cabecera institucional y de estado del sistema."""
        header_frame = tk.Frame(self.root, bg=self.color_primary, height=65)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_frame,
            text="REDES NEURONALES MULTICAPA (MLP) - PREDICCION DE CALIDAD DEL AIRE (PM2.5 A 24H)",
            font=("Segoe UI", 12, "bold"),
            fg="#FFFFFF",
            bg=self.color_primary
        )
        title_lbl.pack(anchor="w", padx=20, pady=(8, 2))

        sub_lbl = tk.Label(
            header_frame,
            text="Universidad de Cartagena | Dago David Palmera Navarro - Julian David Camargo Padilla",
            font=("Segoe UI", 9),
            fg="#D9E1F2",
            bg=self.color_primary
        )
        sub_lbl.pack(anchor="w", padx=20, pady=(0, 8))

    def _build_notebook(self):
        """Construye el contenedor de pestanas principales."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Pestana 1: Visualizador Topologico en Grafo
        self.tab_graph = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_graph, text="Grafo Topologico")
        self._build_tab_graph()

        # Pestana 2: Curvas de Aprendizaje
        self.tab_train = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="Curvas de Aprendizaje")
        self._build_tab_train()

        # Pestana 3: Evaluacion y Comparativa
        self.tab_eval = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_eval, text="Evaluacion y Metricas")
        self._build_tab_eval()

        # Pestana 4: Simulador Interactivo
        self.tab_sim = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sim, text="Simulador Individual")
        self._build_tab_sim()

    # -------------------------------------------------------------------------
    # Pestana 1: Grafo Topologico
    # -------------------------------------------------------------------------
    def _build_tab_graph(self):
        """Construye la pestana con el renderizado interactivo de la red neuronal como grafo."""
        control_frame = tk.Frame(self.tab_graph, bg=self.color_card, bd=1, relief=tk.SOLID)
        control_frame.pack(fill=tk.X, padx=10, pady=8)

        lbl = tk.Label(control_frame, text="Seleccionar Modelo para Visualizar Grafo:", font=("Segoe UI", 10, "bold"), bg=self.color_card)
        lbl.pack(side=tk.LEFT, padx=12, pady=10)

        self.graph_model_var = tk.StringVar(value="Referencia (10:10:1)")
        combo = ttk.Combobox(
            control_frame,
            textvariable=self.graph_model_var,
            values=["Referencia (10:10:1)", "Propia Propuesta (10:16:8:1)"],
            state="readonly",
            width=28
        )
        combo.pack(side=tk.LEFT, padx=8)
        combo.bind("<<ComboboxSelected>>", lambda e: self._draw_network_graph())

        btn_draw = ttk.Button(control_frame, text="Actualizar Grafo", command=self._draw_network_graph)
        btn_draw.pack(side=tk.LEFT, padx=8)

        self.graph_info_lbl = tk.Label(control_frame, text="", font=("Segoe UI", 9, "italic"), bg=self.color_card, fg="#1F4E79")
        self.graph_info_lbl.pack(side=tk.RIGHT, padx=15)

        # Lienzo Matplotlib para el grafo
        plot_frame = tk.Frame(self.tab_graph, bg=self.color_card)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig_graph = Figure(figsize=(9, 5.5), dpi=100)
        self.ax_graph = self.fig_graph.add_subplot(111)
        self.canvas_graph = FigureCanvasTkAgg(self.fig_graph, master=plot_frame)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(plot_frame, bg=self.color_bg)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_graph = NavigationToolbar2Tk(self.canvas_graph, toolbar_frame)
        self.toolbar_graph.update()

        self._draw_network_graph()

    def _draw_network_graph(self):
        """Dibuja las capas, nodos y aristas de la red seleccionada como grafo conexo."""
        self.ax_graph.clear()
        selected = self.graph_model_var.get()

        if "Referencia" in selected:
            layers = [10, 10, 1]
            layer_names = ["Entrada (10)", "Oculta 1 (10, tansig)", "Salida (1, purelin)"]
            total_params = 121
            title = "Grafo Topologico - Arquitectura de Referencia (10:10:1)"
        else:
            layers = [10, 16, 8, 1]
            layer_names = ["Entrada (10)", "Oculta 1 (16, tansig)", "Oculta 2 (8, tansig)", "Salida (1, purelin)"]
            total_params = 321
            title = "Grafo Topologico - Arquitectura Propia en Embudo (10:16:8:1)"

        self.graph_info_lbl.config(text=f"Total Parametros Entrenables: {total_params}")

        # Coordenadas de los nodos por capa
        n_layers = len(layers)
        max_nodes = max(layers)
        x_coords = np.linspace(0.1, 0.9, n_layers)

        node_positions = []
        for l_idx, n_nodes in enumerate(layers):
            y_coords = np.linspace(0.1, 0.9, n_nodes) if n_nodes > 1 else np.array([0.5])
            layer_pos = [(x_coords[l_idx], y) for y in y_coords]
            node_positions.append(layer_pos)

        # Dibujar aristas sinapticas (conexiones entre capas consecutivas)
        for l_idx in range(n_layers - 1):
            src_nodes = node_positions[l_idx]
            dst_nodes = node_positions[l_idx + 1]
            for (x1, y1) in src_nodes:
                for (x2, y2) in dst_nodes:
                    self.ax_graph.plot([x1, x2], [y1, y2], color="#BDC3C7", lw=0.4, alpha=0.5, zorder=1)

        # Dibujar nodos de cada capa
        feature_labels = ["TEMP", "PRES", "DEWP", "Iws", "hour", "day", "pm25_t", "t-1", "t-2", "t-3"]
        for l_idx, (layer_pos, n_nodes) in enumerate(zip(node_positions, layers)):
            color = "#2E75B6" if l_idx == 0 else ("#1F4E79" if l_idx < n_layers - 1 else "#C00000")
            for n_idx, (x, y) in enumerate(layer_pos):
                circle = plt.Circle((x, y), 0.02, color=color, ec="#2C3E50", lw=1.2, zorder=3)
                self.ax_graph.add_patch(circle)

                # Etiquetas de entrada y salida
                if l_idx == 0 and n_idx < len(feature_labels):
                    self.ax_graph.text(x - 0.035, y, feature_labels[n_idx], fontsize=7, ha="right", va="center", weight="bold")
                elif l_idx == n_layers - 1:
                    self.ax_graph.text(x + 0.035, y, "pm25_t+24", fontsize=8, ha="left", va="center", weight="bold", color="#C00000")

            # Titulo de la capa en la parte superior
            self.ax_graph.text(x_coords[l_idx], 0.97, layer_names[l_idx], fontsize=8, ha="center", weight="bold", color="#1F4E79")

        self.ax_graph.set_xlim(0.0, 1.0)
        self.ax_graph.set_ylim(0.05, 1.02)
        self.ax_graph.axis("off")
        self.ax_graph.set_title(title, fontsize=11, weight="bold", color="#1F4E79", pad=12)

        self.canvas_graph.draw()

    # -------------------------------------------------------------------------
    # Pestana 2: Curvas de Aprendizaje
    # -------------------------------------------------------------------------
    def _build_tab_train(self):
        """Construye el panel de control y graficacion del entrenamiento en tiempo real."""
        top_frame = tk.Frame(self.tab_train, bg=self.color_card, bd=1, relief=tk.SOLID)
        top_frame.pack(fill=tk.X, padx=10, pady=8)

        # Hiperparametros
        tk.Label(top_frame, text="Epocas:", font=("Segoe UI", 9, "bold"), bg=self.color_card).grid(row=0, column=0, padx=6, pady=6)
        self.epochs_entry = ttk.Entry(top_frame, width=8)
        self.epochs_entry.insert(0, "500")
        self.epochs_entry.grid(row=0, column=1, padx=4)

        tk.Label(top_frame, text="Tasa de Aprendizaje (eta):", font=("Segoe UI", 9, "bold"), bg=self.color_card).grid(row=0, column=2, padx=6)
        self.lr_entry = ttk.Entry(top_frame, width=8)
        self.lr_entry.insert(0, "0.01")
        self.lr_entry.grid(row=0, column=3, padx=4)

        self.btn_train = ttk.Button(top_frame, text="Entrenar Ambas Redes", command=self._start_training)
        self.btn_train.grid(row=0, column=4, padx=16)

        self.train_status_lbl = tk.Label(top_frame, text="Esperando datos...", font=("Segoe UI", 9, "italic"), bg=self.color_card)
        self.train_status_lbl.grid(row=0, column=5, padx=8)

        # Progreso
        self.progress_bar = ttk.Progressbar(top_frame, orient="horizontal", mode="determinate", length=220)
        self.progress_bar.grid(row=0, column=6, padx=10)

        # Grafica interactiva de curvas de perdida
        plot_frame = tk.Frame(self.tab_train, bg=self.color_card)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig_train = Figure(figsize=(9, 5), dpi=100)
        self.ax_train = self.fig_train.add_subplot(111)
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=plot_frame)
        self.canvas_train.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(plot_frame, bg=self.color_bg)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_train = NavigationToolbar2Tk(self.canvas_train, toolbar_frame)
        self.toolbar_train.update()

    def _start_training(self):
        """Inicia el entrenamiento en un hilo secundario para mantener la interfaz receptiva."""
        if self.is_training:
            messagebox.showwarning("Proceso Activo", "Ya existe un entrenamiento en ejecucion.")
            return

        if self.data is None:
            messagebox.showerror("Error", "Los datos aun no han sido cargados.")
            return

        try:
            epochs = int(self.epochs_entry.get())
            lr = float(self.lr_entry.get())
        except ValueError:
            messagebox.showerror("Parametros Invalidos", "Ingrese valores numericos validos para epocas y tasa de aprendizaje.")
            return

        self.is_training = True
        self.btn_train.config(state=tk.DISABLED)
        self.progress_bar["value"] = 0
        self.train_status_lbl.config(text="Entrenando modelos...")

        thread = threading.Thread(target=self._run_training_worker, args=(epochs, lr), daemon=True)
        thread.start()

    def _run_training_worker(self, epochs, lr):
        """Hilo de ejecucion para el entrenamiento de ambas arquitecturas."""
        # Modelo 1: Referencia (10:10:1)
        self.model_ref = MultilayerPerceptron(topology=[10, 10, 1], lr=lr, seed=42)
        # Modelo 2: Propia (10:16:8:1)
        self.model_prop = MultilayerPerceptron(topology=[10, 16, 8, 1], lr=lr, seed=42)

        X_tr = self.data["X_train_norm"]
        y_tr = self.data["y_train_norm"]
        X_te = self.data["X_test_norm"]
        y_te = self.data["y_test_norm"]

        for ep in range(1, epochs + 1):
            tr_mse_ref = self.model_ref.train_epoch(X_tr, y_tr)
            tr_mse_prop = self.model_prop.train_epoch(X_tr, y_tr)

            te_pred_ref = self.model_ref.predict(X_te)
            te_pred_prop = self.model_prop.predict(X_te)

            te_mse_ref = float(np.mean((y_te - te_pred_ref) ** 2))
            te_mse_prop = float(np.mean((y_te - te_pred_prop) ** 2))

            self.model_ref.history["epoch"].append(ep)
            self.model_ref.history["train_mse"].append(tr_mse_ref)
            self.model_ref.history["test_mse"].append(te_mse_ref)

            self.model_prop.history["epoch"].append(ep)
            self.model_prop.history["train_mse"].append(tr_mse_prop)
            self.model_prop.history["test_mse"].append(te_mse_prop)

            if ep % 25 == 0 or ep == epochs:
                prog = int((ep / epochs) * 100)
                self.root.after(0, self._update_train_progress, ep, epochs, prog)

        self.root.after(0, self._finish_training)

    def _update_train_progress(self, ep, epochs, prog):
        """Actualiza el progreso en la interfaz."""
        self.progress_bar["value"] = prog
        self.train_status_lbl.config(text=f"Epoca {ep}/{epochs} ({prog}%)")

    def _finish_training(self):
        """Finaliza el entrenamiento y actualiza las graficas y tablas de evaluacion."""
        self.is_training = False
        self.btn_train.config(state=tk.NORMAL)
        self.train_status_lbl.config(text="Entrenamiento completado exitosamente.")

        # Graficar curvas de aprendizaje
        self._plot_learning_curves()
        # Actualizar pestana de evaluacion
        self._update_eval_tab()

        messagebox.showinfo("Entrenamiento Finalizado", "Ambas arquitecturas han sido entrenadas exitosamente.")

    def _plot_learning_curves(self):
        """Grafica la evolucion comparativa del MSE sobre train y test para ambos modelos."""
        self.ax_train.clear()

        ep_ref = self.model_ref.history["epoch"]
        tr_ref = self.model_ref.history["train_mse"]
        te_ref = self.model_ref.history["test_mse"]

        tr_prop = self.model_prop.history["train_mse"]
        te_prop = self.model_prop.history["test_mse"]

        self.ax_train.plot(ep_ref, tr_ref, label="Referencia (Train MSE)", color="#2E75B6", lw=1.6)
        self.ax_train.plot(ep_ref, te_ref, label="Referencia (Test MSE)", color="#2E75B6", lw=1.6, linestyle="--")

        self.ax_train.plot(ep_ref, tr_prop, label="Propia (Train MSE)", color="#C00000", lw=1.6)
        self.ax_train.plot(ep_ref, te_prop, label="Propia (Test MSE)", color="#C00000", lw=1.6, linestyle="--")

        self.ax_train.set_title("Curvas de Aprendizaje Comparativas (MSE vs Epocas)", fontsize=11, weight="bold", color="#1F4E79")
        self.ax_train.set_xlabel("Epoca de Entrenamiento", fontsize=9, weight="bold")
        self.ax_train.set_ylabel("Error Cuadratico Medio (MSE normalizado)", fontsize=9, weight="bold")
        self.ax_train.grid(True, linestyle=":", alpha=0.6)
        self.ax_train.legend(loc="upper right", fontsize=8)

        self.canvas_train.draw()

    # -------------------------------------------------------------------------
    # Pestana 3: Evaluacion y Metricas
    # -------------------------------------------------------------------------
    def _build_tab_eval(self):
        """Construye el panel de metricas cuantitativas y grafica de real vs predicho."""
        table_frame = tk.Frame(self.tab_eval, bg=self.color_card, bd=1, relief=tk.SOLID)
        table_frame.pack(fill=tk.X, padx=10, pady=8)

        cols = ("Modelo", "Topologia", "Parametros", "Train MSE (norm)", "Test MSE (norm)", "Test MSE (real)", "Test RMSE (ug/m3)")
        self.eval_tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=3)
        for c in cols:
            self.eval_tree.heading(c, text=c)
            self.eval_tree.column(c, anchor="center", width=145)

        self.eval_tree.pack(fill=tk.X, padx=8, pady=8)

        # Grafica de dispersion y serie temporal Real vs Predicho
        plot_frame = tk.Frame(self.tab_eval, bg=self.color_card)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig_eval = Figure(figsize=(9, 4.8), dpi=100)
        self.ax_eval = self.fig_eval.add_subplot(111)
        self.canvas_eval = FigureCanvasTkAgg(self.fig_eval, master=plot_frame)
        self.canvas_eval.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(plot_frame, bg=self.color_bg)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar_eval = NavigationToolbar2Tk(self.canvas_eval, toolbar_frame)
        self.toolbar_eval.update()

    def _update_eval_tab(self):
        """Calcula y proyecta las metricas de evaluacion en la tabla y la grafica."""
        if self.model_ref is None or self.model_prop is None:
            return

        for item in self.eval_tree.get_children():
            self.eval_tree.delete(item)

        X_te = self.data["X_test_norm"]
        y_te = self.data["y_test_norm"]
        scaler_y = self.data["scaler_y"]

        ev_ref = self.model_ref.evaluate(X_te, y_te, scaler_y=scaler_y)
        ev_prop = self.model_prop.evaluate(X_te, y_te, scaler_y=scaler_y)

        tr_mse_ref = self.model_ref.history["train_mse"][-1]
        tr_mse_prop = self.model_prop.history["train_mse"][-1]

        self.eval_tree.insert("", tk.END, values=(
            "Referencia", "10:10:1", "121",
            f"{tr_mse_ref:.6f}", f"{ev_ref['mse_norm']:.6f}",
            f"{ev_ref['mse_real']:.2f}", f"{ev_ref['rmse_real']:.2f}"
        ))

        self.eval_tree.insert("", tk.END, values=(
            "Propia Propuesta", "10:16:8:1", "321",
            f"{tr_mse_prop:.6f}", f"{ev_prop['mse_norm']:.6f}",
            f"{ev_prop['mse_real']:.2f}", f"{ev_prop['rmse_real']:.2f}"
        ))

        # Graficar una ventana representativa de 200 horas consecutivas de prueba
        self.ax_eval.clear()
        window = 200
        y_real_slice = ev_ref["y_real"][:window].flatten()
        y_pred_ref_slice = ev_ref["y_pred_real"][:window].flatten()
        y_pred_prop_slice = ev_prop["y_pred_real"][:window].flatten()
        hours = np.arange(1, window + 1)

        self.ax_eval.plot(hours, y_real_slice, label="Concentracion Real (PM2.5)", color="#1F4E79", lw=1.8)
        self.ax_eval.plot(hours, y_pred_ref_slice, label="Prediccion Ref (10:10:1)", color="#2E75B6", lw=1.4, linestyle="--")
        self.ax_eval.plot(hours, y_pred_prop_slice, label="Prediccion Propia (10:16:8:1)", color="#C00000", lw=1.4, linestyle="-.")

        self.ax_eval.set_title(f"Concentracion Real frente a Predicha (Muestra de {window} horas en Test)", fontsize=11, weight="bold", color="#1F4E79")
        self.ax_eval.set_xlabel("Horas de Prueba Consecutivas", fontsize=9, weight="bold")
        self.ax_eval.set_ylabel("Concentracion de PM2.5 (ug/m3)", fontsize=9, weight="bold")
        self.ax_eval.grid(True, linestyle=":", alpha=0.6)
        self.ax_eval.legend(loc="upper right", fontsize=8)

        self.canvas_eval.draw()

    # -------------------------------------------------------------------------
    # Pestana 4: Simulador Individual
    # -------------------------------------------------------------------------
    def _build_tab_sim(self):
        """Construye un panel para ingresar valores de prueba y simular la prediccion continua."""
        main_frame = tk.Frame(self.tab_sim, bg=self.color_card, bd=1, relief=tk.SOLID)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        title = tk.Label(main_frame, text="Simulador de Prediccion PM2.5 a 24 Horas", font=("Segoe UI", 12, "bold"), fg=self.color_primary, bg=self.color_card)
        title.pack(anchor="w", padx=20, pady=(15, 10))

        inputs_grid = tk.Frame(main_frame, bg=self.color_card)
        inputs_grid.pack(fill=tk.X, padx=20, pady=5)

        self.sim_entries = {}
        feature_defaults = [
            ("TEMP", "-5.0", "Temperatura (C)"),
            ("PRES", "1022.0", "Presion Atmosferica (hPa)"),
            ("DEWP", "-7.0", "Punto de Rocio (C)"),
            ("Iws", "5.36", "Viento Acumulado (m/s)"),
            ("hour", "3", "Hora del dia (0-23)"),
            ("day", "2", "Dia del mes (1-31)"),
            ("pm25_t", "181.0", "PM2.5 Actual (t)"),
            ("pm25_t-1", "159.0", "PM2.5 Retardo 1 (t-1)"),
            ("pm25_t-2", "148.0", "PM2.5 Retardo 2 (t-2)"),
            ("pm25_t-3", "129.0", "PM2.5 Retardo 3 (t-3)")
        ]

        for idx, (feat, default_val, desc) in enumerate(feature_defaults):
            row = idx // 2
            col = (idx % 2) * 2

            lbl = tk.Label(inputs_grid, text=f"{feat} ({desc}):", font=("Segoe UI", 9, "bold"), bg=self.color_card)
            lbl.grid(row=row, column=col, sticky="w", padx=8, pady=4)

            entry = ttk.Entry(inputs_grid, width=12)
            entry.insert(0, default_val)
            entry.grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)
            self.sim_entries[feat] = entry

        btn_sim = ttk.Button(main_frame, text="Calcular Prediccion a 24 Horas", command=self._execute_simulation)
        btn_sim.pack(anchor="w", padx=20, pady=15)

        # Panel de resultados de prediccion
        self.sim_res_frame = tk.Frame(main_frame, bg="#EBF1F5", bd=1, relief=tk.SOLID)
        self.sim_res_frame.pack(fill=tk.X, padx=20, pady=10)

        self.sim_res_lbl = tk.Label(
            self.sim_res_frame,
            text="Ingrese los parametros de entrada y haga clic en 'Calcular Prediccion'.",
            font=("Segoe UI", 10),
            bg="#EBF1F5",
            fg="#1F4E79",
            justify=tk.LEFT
        )
        self.sim_res_lbl.pack(anchor="w", padx=15, pady=12)

    def _execute_simulation(self):
        """Ejecuta la inferencia manual para el vector ingresado."""
        if self.model_ref is None or self.model_prop is None:
            messagebox.showwarning("Modelos no entrenados", "Primero entrene los modelos en la pestana 'Curvas de Aprendizaje'.")
            return

        try:
            vec_raw = []
            for feat in ["TEMP", "PRES", "DEWP", "Iws", "hour", "day", "pm25_t", "pm25_t-1", "pm25_t-2", "pm25_t-3"]:
                val = float(self.sim_entries[feat].get())
                vec_raw.append(val)
        except ValueError:
            messagebox.showerror("Error de Formato", "Asegurese de que todas las variables contengan valores numericos validos.")
            return

        X_input_raw = np.array([vec_raw], dtype=np.float64)
        X_input_norm = self.data["scaler_X"].transform(X_input_raw)

        # Prediccion de ambos modelos
        pred_norm_ref = self.model_ref.predict(X_input_norm)
        pred_norm_prop = self.model_prop.predict(X_input_norm)

        pred_real_ref = float(self.data["scaler_y"].inverse_transform(pred_norm_ref)[0, 0])
        pred_real_prop = float(self.data["scaler_y"].inverse_transform(pred_norm_prop)[0, 0])

        res_text = (
            f"Resultados de Prediccion (PM2.5 a 24 horas hacia adelante):\n"
            f"  - Modelo 1 (Referencia 10:10:1):      {pred_real_ref:.2f} ug/m3\n"
            f"  - Modelo 2 (Propia Propuesta 10:16:8:1): {pred_real_prop:.2f} ug/m3\n"
            f"  - Nivel de Calidad del Aire Estimado:  {self._classify_aqi(pred_real_prop)}"
        )
        self.sim_res_lbl.config(text=res_text)

    @staticmethod
    def _classify_aqi(pm25_val):
        """Clasifica el nivel de riesgo de salud segun la escala estandar de PM2.5."""
        if pm25_val <= 50:
            return "Bueno (0 - 50 ug/m3)"
        elif pm25_val <= 100:
            return "Moderado (51 - 100 ug/m3)"
        elif pm25_val <= 150:
            return "Danino para grupos sensibles (101 - 150 ug/m3)"
        elif pm25_val <= 200:
            return "Danino para la salud (151 - 200 ug/m3)"
        elif pm25_val <= 300:
            return "Muy danino para la salud (201 - 300 ug/m3)"
        else:
            return "Peligroso (> 300 ug/m3)"

    # -------------------------------------------------------------------------
    # Carga de Datos en Segundo Plano
    # -------------------------------------------------------------------------
    def _load_data_thread(self):
        """Inicia la carga de datos sin bloquear el arranque de la GUI."""
        thread = threading.Thread(target=self._load_data_worker, daemon=True)
        thread.start()

    def _load_data_worker(self):
        """Carga el dataset y actualiza el estado."""
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "datasets", "dataset_pm25.csv"))
        if os.path.exists(csv_path):
            self.data = load_and_preprocess_dataset(csv_path)
            self.root.after(0, lambda: self.train_status_lbl.config(text="Datos cargados y listos para entrenar."))


def launch_gui():
    """Punto de entrada de la interfaz grafica."""
    root = tk.Tk()
    app = MlpAppGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
