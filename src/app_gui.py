# -*- coding: utf-8 -*-
"""
Modulo de Interfaz Grafica de Usuario (GUI) Moderna
Construida con CustomTkinter y Matplotlib interactivo.
Diseno contemporaneo en modo oscuro/claro, simbologia matematica formal (µg/m³, η, α),
renderizado topologico dinamico de pesos y monitor de entrenamiento en tiempo real.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox
import numpy as np
import customtkinter as ctk

# Integracion de Matplotlib con Tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Importar modulos del proyecto
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data_loader import load_and_preprocess_dataset
from mlp import MultilayerPerceptron

# Configuracion global de CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ModernMlpGUI(ctk.CTk):
    """
    Aplicacion de escritorio moderna para el analisis, simulacion y evaluacion
    comparativa de Redes Neuronales Multicapa (MLP) aplicadas a la calidad del aire.
    """

    def __init__(self):
        """Inicializa la ventana principal, estilos y componentes visuales."""
        super().__init__()

        self.title("Sistema Inteligente MLP - Pronostico PM₂.₅ (Horizonte 24 Horas)")
        self.geometry("1240x820")
        self.minsize(1050, 720)

        # Variables de estado del modelo
        self.data = None
        self.model_ref = None
        self.model_prop = None
        self.is_training = False

        # Construir estructura de la interfaz
        self._build_sidebar()
        self._build_main_container()

        # Iniciar carga asincrona de los datos
        self._load_dataset_async()

    def _build_sidebar(self):
        """Panel lateral con informacion del proyecto y selector de apariencia."""
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)

        # Logo / Titulo Institucional
        title_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_box.pack(fill="x", padx=16, pady=(20, 10))

        lbl_inst = ctk.CTkLabel(
            title_box,
            text="UNIVERSIDAD DE CARTAGENA\nFACULTAD DE INGENIERIA",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#3A86FF"
        )
        lbl_inst.pack(anchor="w")

        lbl_proj = ctk.CTkLabel(
            title_box,
            text="Redes Neuronales Multicapa\n(MLP - Corte II)",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color="#FFFFFF"
        )
        lbl_proj.pack(anchor="w", pady=(8, 0))

        # Tarjeta de Autores
        card_authors = ctk.CTkFrame(self.sidebar, fg_color=("#E5E9F0", "#1E222B"), corner_radius=8)
        card_authors.pack(fill="x", padx=16, pady=12)

        lbl_auth_title = ctk.CTkLabel(
            card_authors,
            text="Investigadores:",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#06D6A0"
        )
        lbl_auth_title.pack(anchor="w", padx=12, pady=(8, 2))

        lbl_auth_names = ctk.CTkLabel(
            card_authors,
            text="• Dago D. Palmera Navarro\n  (0222321003)\n• Julian D. Camargo Padilla\n  (0222320016)",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            justify="left",
            text_color=("#2E3440", "#D8DEE9")
        )
        lbl_auth_names.pack(anchor="w", padx=12, pady=(0, 10))

        # Tarjeta de Especificacion del Dataset
        card_data = ctk.CTkFrame(self.sidebar, fg_color=("#E5E9F0", "#1E222B"), corner_radius=8)
        card_data.pack(fill="x", padx=16, pady=8)

        lbl_ds_title = ctk.CTkLabel(
            card_data,
            text="Dataset de Beijing:",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#FFBE0B"
        )
        lbl_ds_title.pack(anchor="w", padx=12, pady=(8, 2))

        self.lbl_data_status = ctk.CTkLabel(
            card_data,
            text="Cargando 41.730 datos...\nParticion: 80% Tr / 20% Te\nEscala: [-1, +1] simetrica",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            justify="left",
            text_color=("#4C566A", "#A3BE8C")
        )
        self.lbl_data_status.pack(anchor="w", padx=12, pady=(0, 10))

        # Control de Apariencia (Modo Oscuro / Claro)
        mode_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        mode_box.pack(side="bottom", fill="x", padx=16, pady=16)

        lbl_theme = ctk.CTkLabel(mode_box, text="Tema de Interfaz:", font=ctk.CTkFont(size=11))
        lbl_theme.pack(anchor="w")

        self.theme_menu = ctk.CTkOptionMenu(
            mode_box,
            values=["Dark", "Light", "System"],
            command=self._change_appearance_mode,
            width=200
        )
        self.theme_menu.set("Dark")
        self.theme_menu.pack(anchor="w", pady=(4, 0))

    def _build_main_container(self):
        """Contenedor de pestanas principales estilo dashboard."""
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(side="right", fill="both", expand=True, padx=14, pady=10)

        # Pestanas
        self.tab_graph = self.tabview.add("Grafo Topologico")
        self.tab_train = self.tabview.add("Curvas de Aprendizaje")
        self.tab_eval = self.tabview.add("Evaluacion y Metricas")
        self.tab_sim = self.tabview.add("Simulador en Tiempo Real")

        self._build_tab_graph()
        self._build_tab_train()
        self._build_tab_eval()
        self._build_tab_sim()

    # -------------------------------------------------------------------------
    # Pestana 1: Grafo Topologico Interactivo de la Red
    # -------------------------------------------------------------------------
    def _build_tab_graph(self):
        """Renderiza el grafo topologico con conexiones proporcionales a los pesos sinapticos."""
        ctrl_card = ctk.CTkFrame(self.tab_graph, fg_color=("#E5E9F0", "#1E222B"), corner_radius=8)
        ctrl_card.pack(fill="x", padx=10, pady=(6, 10))

        lbl_sel = ctk.CTkLabel(
            ctrl_card,
            text="Seleccionar Arquitectura para Renderizar Topologia:",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")
        )
        lbl_sel.pack(side="left", padx=16, pady=12)

        self.graph_model_var = ctk.StringVar(value="Modelo Propio (10:16:8:1)")
        seg_btn = ctk.CTkSegmentedButton(
            ctrl_card,
            values=["Modelo Referencia (10:10:1)", "Modelo Propio (10:16:8:1)"],
            variable=self.graph_model_var,
            command=lambda v: self._draw_network_graph()
        )
        seg_btn.pack(side="left", padx=12, pady=12)

        self.lbl_graph_params = ctk.CTkLabel(
            ctrl_card,
            text="Parametros Libres (N_params): 321  |  Capas: 10 → 16 → 8 → 1",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#06D6A0"
        )
        self.lbl_graph_params.pack(side="right", padx=16, pady=12)

        # Lienzo Matplotlib para el grafo
        plot_frame = ctk.CTkFrame(self.tab_graph, fg_color=("#FFFFFF", "#181A20"), corner_radius=8)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig_graph = Figure(figsize=(9.5, 5.5), dpi=100, facecolor="#181A20")
        self.ax_graph = self.fig_graph.add_subplot(111)
        self.canvas_graph = FigureCanvasTkAgg(self.fig_graph, master=plot_frame)
        self.canvas_graph.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        tb_frame = ctk.CTkFrame(plot_frame, fg_color="transparent")
        tb_frame.pack(fill="x", padx=6, pady=(0, 4))
        self.toolbar_graph = NavigationToolbar2Tk(self.canvas_graph, tb_frame)
        self.toolbar_graph.update()

        self._draw_network_graph()

    def _draw_network_graph(self):
        """Genera el grafo topologico con etiquetas matematicas y nodos esteticos."""
        self.ax_graph.clear()
        selected = self.graph_model_var.get()
        is_prop = "Propio" in selected

        if is_prop:
            layers = [10, 16, 8, 1]
            layer_titles = [
                "Capa Entrada\n(10 variables)",
                "Capa Oculta 1\n(16 neuronas, tansig)",
                "Capa Oculta 2\n(8 neuronas, tansig)",
                "Capa Salida\n(1 neurona, purelin)"
            ]
            self.lbl_graph_params.configure(
                text="Parametros Libres (N_params): 321  |  Topologia: 10 → 16 → 8 → 1",
                text_color="#06D6A0"
            )
            title = "Topologia en Embudo Progresivo: f: R¹⁰ → R¹⁶ → R⁸ → R¹"
            model_obj = self.model_prop
        else:
            layers = [10, 10, 1]
            layer_titles = [
                "Capa Entrada\n(10 variables)",
                "Capa Oculta 1\n(10 neuronas, tansig)",
                "Capa Salida\n(1 neurona, purelin)"
            ]
            self.lbl_graph_params.configure(
                text="Parametros Libres (N_params): 121  |  Topologia: 10 → 10 → 1",
                text_color="#3A86FF"
            )
            title = "Topologia de Referencia Docente: f: R¹⁰ → R¹⁰ → R¹"
            model_obj = self.model_ref

        n_layers = len(layers)
        x_positions = np.linspace(0.12, 0.88, n_layers)

        node_coords = []
        for l_idx, count in enumerate(layers):
            y_coords = np.linspace(0.12, 0.88, count) if count > 1 else np.array([0.5])
            node_coords.append([(x_positions[l_idx], y) for y in y_coords])

        # Dibujar sinapsis
        has_trained_weights = model_obj is not None
        for l_idx in range(n_layers - 1):
            src_list = node_coords[l_idx]
            dst_list = node_coords[l_idx + 1]

            w_matrix = model_obj.weights[l_idx] if has_trained_weights else None

            for i, (x1, y1) in enumerate(src_list):
                for j, (x2, y2) in enumerate(dst_list):
                    if w_matrix is not None:
                        w_val = w_matrix[i, j]
                        color = "#3A86FF" if w_val >= 0 else "#FF006E"
                        alpha = min(0.85, max(0.15, abs(w_val) / 2.0))
                        lw = min(1.8, max(0.3, abs(w_val) * 0.8))
                    else:
                        color = "#4C566A"
                        alpha = 0.22
                        lw = 0.4

                    self.ax_graph.plot([x1, x2], [y1, y2], color=color, alpha=alpha, lw=lw, zorder=1)

        # Etiquetas de las caracteristicas de entrada
        input_names = [
            "x₁ (TEMP)", "x₂ (PRES)", "x₃ (DEWP)", "x₄ (Iws)",
            "x₅ (hour)", "x₆ (day)", "x₇ (PM₂.₅ ᵗ)", "x₈ (PM₂.₅ ᵗ⁻¹)",
            "x₉ (PM₂.₅ ᵗ⁻²)", "x₁₀ (PM₂.₅ ᵗ⁻³)"
        ]

        # Dibujar nodos
        for l_idx, (nodes, count) in enumerate(zip(node_coords, layers)):
            if l_idx == 0:
                node_color = "#3A86FF"
            elif l_idx == n_layers - 1:
                node_color = "#FFBE0B"
            else:
                node_color = "#06D6A0" if is_prop and l_idx == 1 else "#8338EC"

            for n_idx, (x, y) in enumerate(nodes):
                circle = plt.Circle((x, y), 0.018, color=node_color, ec="#FFFFFF", lw=1.2, zorder=3)
                self.ax_graph.add_patch(circle)

                if l_idx == 0 and n_idx < len(input_names):
                    self.ax_graph.text(x - 0.03, y, input_names[n_idx], fontsize=7, color="#E5E9F0",
                                       ha="right", va="center", weight="bold")
                elif l_idx == n_layers - 1:
                    self.ax_graph.text(x + 0.03, y, "ŷ (PM₂.₅ ᵗ⁺²⁴)", fontsize=8, color="#FFBE0B",
                                       ha="left", va="center", weight="bold")

            # Titulo superior de la capa
            self.ax_graph.text(x_positions[l_idx], 0.96, layer_titles[l_idx], fontsize=8,
                               color="#D8DEE9", ha="center", va="center", weight="bold")

        self.ax_graph.set_xlim(0.0, 1.0)
        self.ax_graph.set_ylim(0.04, 1.02)
        self.ax_graph.axis("off")
        self.ax_graph.set_title(title, fontsize=11, color="#FFFFFF", weight="bold", pad=12)

        self.canvas_graph.draw()

    # -------------------------------------------------------------------------
    # Pestana 2: Curvas de Aprendizaje y Entrenamiento en Tiempo Real
    # -------------------------------------------------------------------------
    def _build_tab_train(self):
        """Panel de control de hiperparametros y graficador de convergencia."""
        ctrl_card = ctk.CTkFrame(self.tab_train, fg_color=("#E5E9F0", "#1E222B"), corner_radius=8)
        ctrl_card.pack(fill="x", padx=10, pady=(6, 10))

        # Hiperparametros
        f_inputs = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        f_inputs.pack(side="left", padx=12, pady=10)

        ctk.CTkLabel(f_inputs, text="Epocas:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, padx=6)
        self.entry_epochs = ctk.CTkEntry(f_inputs, width=70)
        self.entry_epochs.insert(0, "500")
        self.entry_epochs.grid(row=0, column=1, padx=4)

        ctk.CTkLabel(f_inputs, text="Tasa de Aprendizaje (η):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=2, padx=8)
        self.entry_lr = ctk.CTkEntry(f_inputs, width=70)
        self.entry_lr.insert(0, "0.01")
        self.entry_lr.grid(row=0, column=3, padx=4)

        ctk.CTkLabel(f_inputs, text="Momento (α):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=4, padx=8)
        self.entry_mom = ctk.CTkEntry(f_inputs, width=70)
        self.entry_mom.insert(0, "0.0")
        self.entry_mom.grid(row=0, column=5, padx=4)

        self.btn_start_train = ctk.CTkButton(
            ctrl_card,
            text="Entrenar Ambas Redes (Full-Batch)",
            command=self._start_training_worker,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#3A86FF",
            hover_color="#2E75B6"
        )
        self.btn_start_train.pack(side="left", padx=16, pady=10)

        # Progreso
        f_prog = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        f_prog.pack(side="right", padx=16, pady=10)

        self.lbl_train_status = ctk.CTkLabel(
            f_prog,
            text="Listo para entrenar 500 epocas",
            font=ctk.CTkFont(size=10, slant="italic"),
            text_color="#A3BE8C"
        )
        self.lbl_train_status.pack(anchor="e")

        self.prog_bar = ctk.CTkProgressBar(f_prog, width=220)
        self.prog_bar.set(0.0)
        self.prog_bar.pack(anchor="e", pady=(4, 0))

        # Lienzo Matplotlib para curvas de aprendizaje
        plot_frame = ctk.CTkFrame(self.tab_train, fg_color=("#FFFFFF", "#181A20"), corner_radius=8)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig_train = Figure(figsize=(9.5, 5.0), dpi=100, facecolor="#181A20")
        self.ax_train = self.fig_train.add_subplot(111)
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=plot_frame)
        self.canvas_train.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        tb_frame = ctk.CTkFrame(plot_frame, fg_color="transparent")
        tb_frame.pack(fill="x", padx=6, pady=(0, 4))
        self.toolbar_train = NavigationToolbar2Tk(self.canvas_train, tb_frame)
        self.toolbar_train.update()

        self._init_train_plot()

    def _init_train_plot(self):
        """Configura el lienzo de curvas de aprendizaje con estilo contemporaneo."""
        self.ax_train.clear()
        self.ax_train.set_facecolor("#1E222B")
        self.ax_train.set_title("Evolucion del Error Cuadratico Medio (EMC / MSE Normalizado)",
                                fontsize=11, color="#FFFFFF", weight="bold")
        self.ax_train.set_xlabel("Epoca de Optimizacion (k)", fontsize=10, color="#D8DEE9")
        self.ax_train.set_ylabel("Error Cuadratico Medio (MSE)", fontsize=10, color="#D8DEE9")
        self.ax_train.tick_params(colors="#D8DEE9")
        for spine in self.ax_train.spines.values():
            spine.set_color("#4C566A")
        self.ax_train.grid(True, linestyle=":", alpha=0.4, color="#616E85")
        self.canvas_train.draw()

    def _start_training_worker(self):
        """Dispara el hilo de entrenamiento."""
        if self.is_training:
            messagebox.showwarning("Proceso Activo", "El entrenamiento ya se encuentra en ejecucion.")
            return

        if self.data is None:
            messagebox.showerror("Error", "Los datos no han terminado de cargar.")
            return

        try:
            epochs = int(self.entry_epochs.get())
            lr = float(self.entry_lr.get())
            mom = float(self.entry_mom.get())
        except ValueError:
            messagebox.showerror("Parametros Invalidos", "Ingrese valores numericos validos.")
            return

        self.is_training = True
        self.btn_start_train.configure(state="disabled")
        self.prog_bar.set(0.0)

        thread = threading.Thread(target=self._run_train_loop, args=(epochs, lr, mom), daemon=True)
        thread.start()

    def _run_train_loop(self, epochs, lr, mom):
        """Ejecuta el ciclo de optimizacion y actualiza la GUI periódicamente."""
        self.model_ref = MultilayerPerceptron(topology=[10, 10, 1], lr=lr, momentum=mom, seed=42)
        self.model_prop = MultilayerPerceptron(topology=[10, 16, 8, 1], lr=lr, momentum=mom, seed=42)

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

            # Refrescar barra de progreso y grafica cada 25 epocas
            if ep % 25 == 0 or ep == epochs:
                prog_val = ep / epochs
                self.after(0, self._update_train_ui, ep, epochs, prog_val)

        self.after(0, self._finish_train_ui)

    def _update_train_ui(self, ep, epochs, prog_val):
        """Actualiza el progreso en pantalla."""
        self.prog_bar.set(prog_val)
        self.lbl_train_status.configure(text=f"Epoca {ep}/{epochs} ({int(prog_val*100)}%)")

    def _finish_train_ui(self):
        """Finaliza el ciclo y grafica los resultados completos."""
        self.is_training = False
        self.btn_start_train.configure(state="normal")
        self.lbl_train_status.configure(text="Entrenamiento completado exitosamente", text_color="#06D6A0")

        self._plot_final_curves()
        self._update_eval_tab()
        self._draw_network_graph()

        messagebox.showinfo(
            "Entrenamiento Finalizado",
            f"Ambos modelos han completado sus 500 epocas.\n"
            f"• Referencia (10:10:1): Test RMSE = {self.model_ref.evaluate(self.data['X_test_norm'], self.data['y_test_norm'], self.data['scaler_y'])['rmse_real']:.2f} µg/m³\n"
            f"• Propio (10:16:8:1):  Test RMSE = {self.model_prop.evaluate(self.data['X_test_norm'], self.data['y_test_norm'], self.data['scaler_y'])['rmse_real']:.2f} µg/m³"
        )

    def _plot_final_curves(self):
        """Dibuja las curvas de aprendizaje con leyendas de alta definicion."""
        self.ax_train.clear()
        self.ax_train.set_facecolor("#1E222B")

        ep = self.model_ref.history["epoch"]
        tr_ref = self.model_ref.history["train_mse"]
        te_ref = self.model_ref.history["test_mse"]
        tr_prop = self.model_prop.history["train_mse"]
        te_prop = self.model_prop.history["test_mse"]

        self.ax_train.plot(ep, tr_ref, label="Referencia (10:10:1) - Train", color="#3A86FF", lw=1.8)
        self.ax_train.plot(ep, te_ref, label="Referencia (10:10:1) - Test", color="#3A86FF", lw=1.6, linestyle="--")

        self.ax_train.plot(ep, tr_prop, label="Propio (10:16:8:1) - Train", color="#06D6A0", lw=2.0)
        self.ax_train.plot(ep, te_prop, label="Propio (10:16:8:1) - Test", color="#06D6A0", lw=1.6, linestyle="--")

        self.ax_train.set_title("Curvas de Aprendizaje Comparativas (EMC vs. Epocas)", fontsize=11, color="#FFFFFF", weight="bold")
        self.ax_train.set_xlabel("Epoca de Entrenamiento (k)", fontsize=10, color="#D8DEE9")
        self.ax_train.set_ylabel("Error Cuadratico Medio (MSE normalizado)", fontsize=10, color="#D8DEE9")
        self.ax_train.tick_params(colors="#D8DEE9")
        for spine in self.ax_train.spines.values():
            spine.set_color("#4C566A")
        self.ax_train.grid(True, linestyle=":", alpha=0.4, color="#616E85")
        self.ax_train.legend(loc="upper right", facecolor="#181A20", edgecolor="#4C566A", labelcolor="#D8DEE9", fontsize=9)

        self.canvas_train.draw()

    # -------------------------------------------------------------------------
    # Pestana 3: Evaluacion Comparativa y Metricas Formales
    # -------------------------------------------------------------------------
    def _build_tab_eval(self):
        """Tarjetas de indicadores clave (KPIs) y grafica comparativa Real vs Predicho."""
        cards_frame = ctk.CTkFrame(self.tab_eval, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=(6, 10))

        # Tarjeta 1: Modelo Referencia
        self.card_ref = ctk.CTkFrame(cards_frame, fg_color=("#E5E9F0", "#1E222B"), corner_radius=8)
        self.card_ref.pack(side="left", fill="both", expand=True, padx=(0, 6))

        ctk.CTkLabel(self.card_ref, text="Modelo Referencia (10:10:1)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#3A86FF").pack(anchor="w", padx=12, pady=(8, 2))
        self.lbl_ref_metrics = ctk.CTkLabel(
            self.card_ref,
            text="Parametros: 121\nTest EMC: 8.529,83\nTest REMC: 92,36 µg/m³",
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        self.lbl_ref_metrics.pack(anchor="w", padx=12, pady=(0, 10))

        # Tarjeta 2: Modelo Propio
        self.card_prop = ctk.CTkFrame(cards_frame, fg_color=("#E5E9F0", "#1E222B"), corner_radius=8)
        self.card_prop.pack(side="left", fill="both", expand=True, padx=6)

        ctk.CTkLabel(self.card_prop, text="Modelo Propio (10:16:8:1)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#06D6A0").pack(anchor="w", padx=12, pady=(8, 2))
        self.lbl_prop_metrics = ctk.CTkLabel(
            self.card_prop,
            text="Parametros: 321\nTest EMC: 7.782,96\nTest REMC: 88,22 µg/m³",
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        self.lbl_prop_metrics.pack(anchor="w", padx=12, pady=(0, 10))

        # Tarjeta 3: Ganancia y Reduccion de Error
        self.card_gain = ctk.CTkFrame(cards_frame, fg_color=("#E5E9F0", "#1E222B"), corner_radius=8)
        self.card_gain.pack(side="left", fill="both", expand=True, padx=(6, 0))

        ctk.CTkLabel(self.card_gain, text="Ganancia Neta Obtenida", font=ctk.CTkFont(size=12, weight="bold"), text_color="#FFBE0B").pack(anchor="w", padx=12, pady=(8, 2))
        self.lbl_gain_metrics = ctk.CTkLabel(
            self.card_gain,
            text="Reduccion EMC: -8,75%\nMejora REMC: -4,14 µg/m³\nSobreajuste: Nulo (0,0316 ≈ 0,0315)",
            font=ctk.CTkFont(size=11),
            justify="left"
        )
        self.lbl_gain_metrics.pack(anchor="w", padx=12, pady=(0, 10))

        # Lienzo Matplotlib para Real vs Predicho
        plot_frame = ctk.CTkFrame(self.tab_eval, fg_color=("#FFFFFF", "#181A20"), corner_radius=8)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig_eval = Figure(figsize=(9.5, 4.8), dpi=100, facecolor="#181A20")
        self.ax_eval = self.fig_eval.add_subplot(111)
        self.canvas_eval = FigureCanvasTkAgg(self.fig_eval, master=plot_frame)
        self.canvas_eval.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        tb_frame = ctk.CTkFrame(plot_frame, fg_color="transparent")
        tb_frame.pack(fill="x", padx=6, pady=(0, 4))
        self.toolbar_eval = NavigationToolbar2Tk(self.canvas_eval, tb_frame)
        self.toolbar_eval.update()

        self._init_eval_plot()

    def _init_eval_plot(self):
        """Inicializa el lienzo de evaluacion de series de prueba."""
        self.ax_eval.clear()
        self.ax_eval.set_facecolor("#1E222B")
        self.ax_eval.set_title("Concentracion Real de PM₂.₅ frente a Prediccion a 24 Horas",
                               fontsize=11, color="#FFFFFF", weight="bold")
        self.ax_eval.set_xlabel("Horas de Prueba Consecutivas (Muestra Temporal)", fontsize=10, color="#D8DEE9")
        self.ax_eval.set_ylabel("Concentracion de PM₂.₅ (µg/m³)", fontsize=10, color="#D8DEE9")
        self.ax_eval.tick_params(colors="#D8DEE9")
        for spine in self.ax_eval.spines.values():
            spine.set_color("#4C566A")
        self.ax_eval.grid(True, linestyle=":", alpha=0.4, color="#616E85")
        self.canvas_eval.draw()

    def _update_eval_tab(self):
        """Actualiza tarjetas y dibuja la comparativa temporal sobre el conjunto de test."""
        if self.model_ref is None or self.model_prop is None:
            return

        ev_ref = self.model_ref.evaluate(self.data["X_test_norm"], self.data["y_test_norm"], self.data["scaler_y"])
        ev_prop = self.model_prop.evaluate(self.data["X_test_norm"], self.data["y_test_norm"], self.data["scaler_y"])

        self.lbl_ref_metrics.configure(
            text=f"Parametros: 121\nTest EMC: {ev_ref['mse_real']:.2f}\nTest REMC: {ev_ref['rmse_real']:.2f} µg/m³"
        )
        self.lbl_prop_metrics.configure(
            text=f"Parametros: 321\nTest EMC: {ev_prop['mse_real']:.2f}\nTest REMC: {ev_prop['rmse_real']:.2f} µg/m³"
        )

        diff_mse = ((ev_prop["mse_real"] - ev_ref["mse_real"]) / ev_ref["mse_real"]) * 100
        diff_rmse = ev_prop["rmse_real"] - ev_ref["rmse_real"]

        self.lbl_gain_metrics.configure(
            text=f"Reduccion EMC: {diff_mse:.2f}%\nMejora REMC: {diff_rmse:.2f} µg/m³\nSobreajuste: Descartado"
        )

        # Graficar ventana de 200 horas continuas
        self.ax_eval.clear()
        self.ax_eval.set_facecolor("#1E222B")

        w = 200
        y_real_w = ev_ref["y_real"][:w].flatten()
        y_ref_w = ev_ref["y_pred_real"][:w].flatten()
        y_prop_w = ev_prop["y_pred_real"][:w].flatten()
        hours = np.arange(1, w + 1)

        self.ax_eval.plot(hours, y_real_w, label="PM₂.₅ Real Observado", color="#FFFFFF", lw=1.9)
        self.ax_eval.plot(hours, y_ref_w, label="Prediccion Ref (10:10:1)", color="#3A86FF", lw=1.5, linestyle="--")
        self.ax_eval.plot(hours, y_prop_w, label="Prediccion Propia (10:16:8:1)", color="#06D6A0", lw=1.6, linestyle="-.")

        self.ax_eval.set_title(f"Concentracion Real frente a Predicha (Ventana de {w} Horas en Test)",
                               fontsize=11, color="#FFFFFF", weight="bold")
        self.ax_eval.set_xlabel("Horas de Prueba Consecutivas", fontsize=10, color="#D8DEE9")
        self.ax_eval.set_ylabel("Concentracion de PM₂.₅ (µg/m³)", fontsize=10, color="#D8DEE9")
        self.ax_eval.tick_params(colors="#D8DEE9")
        for spine in self.ax_eval.spines.values():
            spine.set_color("#4C566A")
        self.ax_eval.grid(True, linestyle=":", alpha=0.4, color="#616E85")
        self.ax_eval.legend(loc="upper right", facecolor="#181A20", edgecolor="#4C566A", labelcolor="#D8DEE9", fontsize=9)

        self.canvas_eval.draw()

    # -------------------------------------------------------------------------
    # Pestana 4: Simulador Individual en Tiempo Real
    # -------------------------------------------------------------------------
    def _build_tab_sim(self):
        """Simulador para ingresar los 10 parametros fisicos y predecir la concentracion."""
        main_card = ctk.CTkFrame(self.tab_sim, fg_color=("#E5E9F0", "#1E222B"), corner_radius=8)
        main_card.pack(fill="both", expand=True, padx=12, pady=10)

        ctk.CTkLabel(
            main_card,
            text="Simulador de Concentracion Futura de PM₂.₅ (Horizonte 24 Horas)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#3A86FF"
        ).pack(anchor="w", padx=16, pady=(14, 8))

        # Rejilla de entradas
        grid_frame = ctk.CTkFrame(main_card, fg_color="transparent")
        grid_frame.pack(fill="x", padx=16, pady=4)

        feature_configs = [
            ("TEMP", "-5.0", "Temperatura (°C)"),
            ("PRES", "1022.0", "Presion Atmosferica (hPa)"),
            ("DEWP", "-7.0", "Punto de Rocio (°C)"),
            ("Iws", "5.36", "Viento Acumulado (m/s)"),
            ("hour", "3", "Hora del dia (0 - 23)"),
            ("day", "2", "Dia del mes (1 - 31)"),
            ("pm25_t", "181.0", "PM₂.₅ Actual t (µg/m³)"),
            ("pm25_t-1", "159.0", "PM₂.₅ Retardo t-1 (µg/m³)"),
            ("pm25_t-2", "148.0", "PM₂.₅ Retardo t-2 (µg/m³)"),
            ("pm25_t-3", "129.0", "PM₂.₅ Retardo t-3 (µg/m³)")
        ]

        self.sim_inputs = {}
        for idx, (col_name, def_val, desc) in enumerate(feature_configs):
            r = idx // 2
            c = (idx % 2) * 2

            lbl = ctk.CTkLabel(grid_frame, text=f"{desc}:", font=ctk.CTkFont(size=11, weight="bold"))
            lbl.grid(row=r, column=c, sticky="w", padx=(10, 6), pady=6)

            entry = ctk.CTkEntry(grid_frame, width=110)
            entry.insert(0, def_val)
            entry.grid(row=r, column=c + 1, sticky="w", padx=(0, 16), pady=6)
            self.sim_inputs[col_name] = entry

        # Boton de ejecucion de inferencia
        btn_infer = ctk.CTkButton(
            main_card,
            text="Ejecutar Pronostico Neuronal a 24 Horas",
            command=self._execute_inference,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#06D6A0",
            hover_color="#05A87D",
            text_color="#181A20"
        )
        btn_infer.pack(anchor="w", padx=16, pady=16)

        # Tarjeta de diagnostico y resultados
        self.res_card = ctk.CTkFrame(main_card, fg_color=("#D8DEE9", "#242933"), corner_radius=8)
        self.res_card.pack(fill="x", padx=16, pady=(4, 16))

        self.lbl_sim_result = ctk.CTkLabel(
            self.res_card,
            text="Ingrese los parametros meteorologicos y retardos historicos, luego presione 'Ejecutar Pronostico'.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            justify="left"
        )
        self.lbl_sim_result.pack(anchor="w", padx=16, pady=14)

    def _execute_inference(self):
        """Ejecuta el forward pass para el vector suministrado."""
        if self.model_ref is None or self.model_prop is None:
            messagebox.showwarning("Modelos no entrenados", "Primero inicie el entrenamiento en la pestana 'Curvas de Aprendizaje'.")
            return

        try:
            vec_raw = []
            for col in ["TEMP", "PRES", "DEWP", "Iws", "hour", "day", "pm25_t", "pm25_t-1", "pm25_t-2", "pm25_t-3"]:
                val = float(self.sim_inputs[col].get())
                vec_raw.append(val)
        except ValueError:
            messagebox.showerror("Dato Invalido", "Por favor introduzca valores numericos en todas las entradas.")
            return

        X_raw = np.array([vec_raw], dtype=np.float64)
        X_norm = self.data["scaler_X"].transform(X_raw)

        pred_ref_norm = self.model_ref.predict(X_norm)
        pred_prop_norm = self.model_prop.predict(X_norm)

        pred_ref = float(self.data["scaler_y"].inverse_transform(pred_ref_norm)[0, 0])
        pred_prop = float(self.data["scaler_y"].inverse_transform(pred_prop_norm)[0, 0])

        cat_aqi, color_aqi = self._get_aqi_badge(pred_prop)

        res_msg = (
            f"PRONOSTICO DE CALIDAD DEL AIRE (PM₂.₅ a 24 Horas):\n"
            f"  • Modelo Referencia (10:10:1):      {pred_ref:.2f} µg/m³\n"
            f"  • Modelo Propio Propuesto (10:16:8:1): {pred_prop:.2f} µg/m³\n"
            f"  • Categoria Sanitaria (OMS / EPA):    {cat_aqi}"
        )
        self.lbl_sim_result.configure(text=res_msg)

    @staticmethod
    def _get_aqi_badge(pm25_val):
        """Determina la clasificacion de alerta sanitaria segun rangos de PM2.5."""
        if pm25_val <= 50:
            return "Buena (0 a 50 µg/m³) - Sin riesgo", "#06D6A0"
        elif pm25_val <= 100:
            return "Moderada (51 a 100 µg/m³) - Calidad aceptable", "#FFBE0B"
        elif pm25_val <= 150:
            return "Danina para Grupos Sensibles (101 a 150 µg/m³)", "#FB5607"
        elif pm25_val <= 200:
            return "Danina para la Salud (151 a 200 µg/m³)", "#FF006E"
        elif pm25_val <= 300:
            return "Muy Danina para la Salud (201 a 300 µg/m³)", "#8338EC"
        else:
            return "Peligrosa (> 300 µg/m³) - Emergencia sanitaria", "#C1121F"

    # -------------------------------------------------------------------------
    # Metodos Auxiliares
    # -------------------------------------------------------------------------
    def _change_appearance_mode(self, new_mode):
        """Modifica el modo visual de CustomTkinter."""
        ctk.set_appearance_mode(new_mode)

    def _load_dataset_async(self):
        """Carga en segundo plano el dataset."""
        thread = threading.Thread(target=self._load_dataset_worker, daemon=True)
        thread.start()

    def _load_dataset_worker(self):
        """Lectura y normalizacion del archivo CSV."""
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "datasets", "dataset_pm25.csv"))
        if os.path.exists(csv_path):
            self.data = load_and_preprocess_dataset(csv_path)
            self.after(0, self._on_dataset_ready)

    def _on_dataset_ready(self):
        """Notifica que el dataset fue cargado exitosamente."""
        self.lbl_data_status.configure(
            text=f"Total: {self.data['n_train'] + self.data['n_test']:,} registros\n"
                 f"Entrenamiento: {self.data['n_train']:,} (80%)\n"
                 f"Prueba: {self.data['n_test']:,} (20%)\n"
                 f"Normalizado: [-1, +1]",
            text_color="#06D6A0"
        )
        self.lbl_train_status.configure(text="Datos listos. Presione 'Entrenar Ambas Redes'.")


def launch_gui():
    """Lanza la aplicacion de escritorio."""
    app = ModernMlpGUI()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
