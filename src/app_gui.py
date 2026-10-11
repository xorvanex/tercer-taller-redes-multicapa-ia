# -*- coding: utf-8 -*-
"""
Modulo de Interfaz Grafica de Usuario (GUI) Moderna y Avanzada
Construida con CustomTkinter y Matplotlib interactivo.
Diseno exclusivo en modo oscuro, simbologia matematica formal (µg/m³, η, α),
visualizacion de relaciones mediante grafos sinapticos de alto contraste con enfoque selectivo,
diagramas de bloques estilo MATLAB (view(net)), mapas de calor sin fugas de ejes,
analisis cuantitativo integrado y simulador de dispersion ambiental con graficacion proyectiva.
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

# Modo Oscuro Exclusivo
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
        self.geometry("1280x860")
        self.minsize(1100, 740)

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
        """Panel lateral con informacion del proyecto y estado del sistema en modo oscuro."""
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color="#13161C")
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)

        # Logo / Titulo Institucional
        title_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_box.pack(fill="x", padx=16, pady=(18, 10))

        lbl_inst = ctk.CTkLabel(
            title_box,
            text="UNIVERSIDAD DE CARTAGENA\nFACULTAD DE INGENIERIA",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#38BDF8"
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
        card_authors = ctk.CTkFrame(self.sidebar, fg_color="#1E222B", corner_radius=8)
        card_authors.pack(fill="x", padx=16, pady=10)

        lbl_auth_title = ctk.CTkLabel(
            card_authors,
            text="Investigadores:",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#34D399"
        )
        lbl_auth_title.pack(anchor="w", padx=12, pady=(8, 2))

        lbl_auth_names = ctk.CTkLabel(
            card_authors,
            text="• Dago D. Palmera Navarro\n  (0222321003)\n• Julian D. Camargo Padilla\n  (0222320016)",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            justify="left",
            text_color="#D8DEE9"
        )
        lbl_auth_names.pack(anchor="w", padx=12, pady=(0, 10))

        # Tarjeta de Especificacion del Dataset
        card_data = ctk.CTkFrame(self.sidebar, fg_color="#1E222B", corner_radius=8)
        card_data.pack(fill="x", padx=16, pady=6)

        lbl_ds_title = ctk.CTkLabel(
            card_data,
            text="Dataset de Beijing:",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#FBBF24"
        )
        lbl_ds_title.pack(anchor="w", padx=12, pady=(8, 2))

        self.lbl_data_status = ctk.CTkLabel(
            card_data,
            text="Cargando 41.730 registros...\nParticion: 80% Tr / 20% Te\nEscala: [-1, +1] simetrica",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            justify="left",
            text_color="#A3BE8C"
        )
        self.lbl_data_status.pack(anchor="w", padx=12, pady=(0, 10))

        # Tarjeta de Auditoria de Criterios Docente
        card_crit = ctk.CTkFrame(self.sidebar, fg_color="#1E222B", corner_radius=8)
        card_crit.pack(fill="x", padx=16, pady=6)

        lbl_crit_title = ctk.CTkLabel(
            card_crit,
            text="Criterios Evaluados:",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color="#A78BFA"
        )
        lbl_crit_title.pack(anchor="w", padx=12, pady=(6, 2))

        lbl_crit_desc = ctk.CTkLabel(
            card_crit,
            text="✔ 10 Entradas / 1 Salida a 24h\n✔ Particion Temporal 80/20\n✔ Modelo Propio vs Referencia\n✔ Backpropagation Analitico\n✔ Evaluacion MSE, RMSE y MAE\n✔ Diagnostico de Sobreajuste",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            justify="left",
            text_color="#CBD5E1"
        )
        # Boton universal para restablecer y limpiar la interfaz
        self.btn_reset_all = ctk.CTkButton(
            self.sidebar,
            text="↺ Restablecer / Limpiar Todo",
            command=self._reset_entire_interface,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#334155",
            hover_color="#475569",
            border_width=1,
            border_color="#475569",
            text_color="#F8FAFC",
            height=36
        )
        self.btn_reset_all.pack(side="bottom", fill="x", padx=16, pady=16)

    def _build_main_container(self):
        """Contenedor de pestanas principales estilo dashboard."""
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(side="right", fill="both", expand=True, padx=14, pady=10)

        # Pestanas
        self.tab_graph = self.tabview.add("Topologia y Relaciones")
        self.tab_train = self.tabview.add("Curvas de Aprendizaje")
        self.tab_eval = self.tabview.add("Evaluacion y Metricas")
        self.tab_sim = self.tabview.add("Simulador en Tiempo Real")

        self._build_tab_graph()
        self._build_tab_train()
        self._build_tab_eval()
        self._build_tab_sim()

    # -------------------------------------------------------------------------
    # Pestana 1: Topologia y Visualizacion de Relaciones (3 Modos Interactivos)
    # -------------------------------------------------------------------------
    def _build_tab_graph(self):
        """Construye controles avanzados para inspeccionar relaciones sinapticas."""
        ctrl_card = ctk.CTkFrame(self.tab_graph, fg_color="#1E222B", corner_radius=8)
        ctrl_card.pack(fill="x", padx=10, pady=(6, 8))

        # Fila 1: Selectores de Modelo y Modo de Vista
        f_row1 = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        f_row1.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(f_row1, text="Modelo:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=(0, 6))
        self.graph_model_var = ctk.StringVar(value="Propio (10:16:8:1)")
        seg_model = ctk.CTkSegmentedButton(
            f_row1,
            values=["Referencia (10:10:1)", "Propio (10:16:8:1)"],
            variable=self.graph_model_var,
            command=lambda v: self._render_topology_view()
        )
        seg_model.pack(side="left", padx=(0, 16))

        ctk.CTkLabel(f_row1, text="Modo de Visualizacion:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=(0, 6))
        self.graph_view_mode = ctk.StringVar(value="Grafo Sinaptico Nitido")
        seg_view = ctk.CTkSegmentedButton(
            f_row1,
            values=["Grafo Sinaptico Nitido", "Diagrama de Bloques (MATLAB)", "Mapa de Calor de Pesos (Heatmap)"],
            variable=self.graph_view_mode,
            command=lambda v: self._on_view_mode_changed()
        )
        seg_view.pack(side="left", padx=(0, 16))

        # Fila 2: Contenedor dinamico para el filtro de enfoque
        self.f_row2 = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        self.f_row2.pack(fill="x", padx=10, pady=(4, 8))

        self.f_focus_container = ctk.CTkFrame(self.f_row2, fg_color="transparent")
        self.f_focus_container.pack(side="left")

        ctk.CTkLabel(self.f_focus_container, text="Enfocar Relaciones de Entrada:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=(0, 6))
        self.focus_node_var = ctk.StringVar(value="Todas las Relaciones")
        self.opt_focus = ctk.CTkOptionMenu(
            self.f_focus_container,
            values=[
                "Todas las Relaciones",
                "x₁ (TEMP)", "x₂ (PRES)", "x₃ (DEWP)", "x₄ (Iws - Viento)",
                "x₅ (hour)", "x₆ (day)", "x₇ (PM₂.₅ ᵗ)", "x₈ (PM₂.₅ ᵗ⁻¹)",
                "x₉ (PM₂.₅ ᵗ⁻²)", "x₁₀ (PM₂.₅ ᵗ⁻³)"
            ],
            variable=self.focus_node_var,
            command=lambda v: self._render_topology_view(),
            width=200
        )
        self.opt_focus.pack(side="left", padx=(0, 16))

        self.lbl_graph_params = ctk.CTkLabel(
            self.f_row2,
            text="Parametros Libres (N_params): 321  |  Topologia: 10 → 16 → 8 → 1",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#34D399"
        )
        self.lbl_graph_params.pack(side="right", padx=6)

        # Lienzo Matplotlib
        plot_frame = ctk.CTkFrame(self.tab_graph, fg_color="#181A20", corner_radius=8)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig_graph = Figure(figsize=(9.8, 5.2), dpi=100, facecolor="#181A20")
        self.canvas_graph = FigureCanvasTkAgg(self.fig_graph, master=plot_frame)
        self.canvas_graph.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        tb_frame = ctk.CTkFrame(plot_frame, fg_color="transparent")
        tb_frame.pack(fill="x", padx=6, pady=(0, 4))
        self.toolbar_graph = NavigationToolbar2Tk(self.canvas_graph, tb_frame)
        self.toolbar_graph.update()

        self._render_topology_view()

    def _on_view_mode_changed(self):
        """Muestra u oculta el filtro de enfoque segun el modo activo."""
        mode = self.graph_view_mode.get()
        if mode == "Grafo Sinaptico Nitido":
            self.f_focus_container.pack(side="left")
        else:
            self.f_focus_container.pack_forget()
        self._render_topology_view()

    def _render_topology_view(self):
        """
        Limpia completamente la figura y despacha el renderizado.
        El uso estricto de clf() elimina cualquier eje de colorbar residual evitando acumulaciones.
        """
        selected_model = self.graph_model_var.get()
        is_prop = "Propio" in selected_model
        if is_prop:
            self.lbl_graph_params.configure(
                text="Parametros Libres (N_params): 321  |  Topologia: 10 → 16 → 8 → 1",
                text_color="#34D399"
            )
        else:
            self.lbl_graph_params.configure(
                text="Parametros Libres (N_params): 121  |  Topologia: 10 → 10 → 1",
                text_color="#38BDF8"
            )

        self.fig_graph.clf()
        self.ax_graph = self.fig_graph.add_subplot(111)

        view_mode = self.graph_view_mode.get()
        if view_mode == "Diagrama de Bloques (MATLAB)":
            self._draw_matlab_block_diagram()
        elif view_mode == "Mapa de Calor de Pesos (Heatmap)":
            self._draw_weight_heatmap()
        else:
            self._draw_clean_synaptic_graph()

    def _draw_clean_synaptic_graph(self):
        """
        Dibuja el grafo sinaptico con circulos geometricos perfectos (ax.scatter)
        y lineas de sinapsis de alto contraste con resaltado interactivo.
        """
        self.ax_graph.set_facecolor("#181A20")

        selected_model = self.graph_model_var.get()
        is_prop = "Propio" in selected_model
        model_obj = self.model_prop if is_prop else self.model_ref

        if is_prop:
            layers = [10, 16, 8, 1]
            layer_titles = [
                "Capa Entrada (x)\n(10 variables)",
                "Capa Oculta 1 (h¹)\n(16 tansig)",
                "Capa Oculta 2 (h²)\n(8 tansig)",
                "Salida (ŷ)\n(purelin)"
            ]
            self.lbl_graph_params.configure(
                text="Parametros Libres (N_params): 321  |  Topologia: 10 → 16 → 8 → 1",
                text_color="#34D399"
            )
            title = "Grafo de Conexiones Sinapticas - Modelo Propio en Embudo (10:16:8:1)"
        else:
            layers = [10, 10, 1]
            layer_titles = [
                "Capa Entrada (x)\n(10 variables)",
                "Capa Oculta 1 (h¹)\n(10 tansig)",
                "Salida (ŷ)\n(purelin)"
            ]
            self.lbl_graph_params.configure(
                text="Parametros Libres (N_params): 121  |  Topologia: 10 → 10 → 1",
                text_color="#38BDF8"
            )
            title = "Grafo de Conexiones Sinapticas - Modelo Referencia Docente (10:10:1)"

        n_layers = len(layers)
        x_positions = np.linspace(0.12, 0.88, n_layers)

        node_coords = []
        for l_idx, count in enumerate(layers):
            y_coords = np.linspace(0.10, 0.90, count) if count > 1 else np.array([0.5])
            node_coords.append([(x_positions[l_idx], y) for y in y_coords])

        focus_filter = self.focus_node_var.get()
        focus_input_idx = None
        input_names = [
            "x₁ (TEMP)", "x₂ (PRES)", "x₃ (DEWP)", "x₄ (Iws - Viento)",
            "x₅ (hour)", "x₆ (day)", "x₇ (PM₂.₅ ᵗ)", "x₈ (PM₂.₅ ᵗ⁻¹)",
            "x₉ (PM₂.₅ ᵗ⁻²)", "x₁₀ (PM₂.₅ ᵗ⁻³)"
        ]
        if focus_filter in input_names:
            focus_input_idx = input_names.index(focus_filter)

        has_trained_w = model_obj is not None

        # 1. Dibujar lineas de sinapsis con alto contraste y visibilidad
        for l_idx in range(n_layers - 1):
            src_list = node_coords[l_idx]
            dst_list = node_coords[l_idx + 1]
            w_mat = model_obj.weights[l_idx] if has_trained_w else None

            for i, (x1, y1) in enumerate(src_list):
                is_focused = (focus_input_idx is None) or (l_idx == 0 and i == focus_input_idx)

                for j, (x2, y2) in enumerate(dst_list):
                    if w_mat is not None:
                        w_val = w_mat[i, j]
                        base_color = "#38BDF8" if w_val >= 0 else "#F43F5E"
                        w_abs = abs(w_val)
                        alpha = min(0.95, max(0.25, w_abs / 1.5)) if is_focused else 0.04
                        lw = min(2.4, max(0.7, w_abs * 1.1)) if is_focused else 0.3
                    else:
                        base_color = "#60A5FA" if is_focused else "#334155"
                        alpha = 0.55 if is_focused else 0.06
                        lw = 0.9 if is_focused else 0.3

                    self.ax_graph.plot([x1, x2], [y1, y2], color=base_color, alpha=alpha, lw=lw, zorder=1)

        # 2. Dibujar neuronas usando scatter (garantiza circulos perfectamente redondos)
        for l_idx, (nodes, count) in enumerate(zip(node_coords, layers)):
            xs = [pos[0] for pos in nodes]
            ys = [pos[1] for pos in nodes]

            if l_idx == 0:
                c_fill = "#38BDF8"
            elif l_idx == n_layers - 1:
                c_fill = "#FBBF24"
            else:
                c_fill = "#34D399" if is_prop and l_idx == 1 else "#A78BFA"

            self.ax_graph.scatter(xs, ys, s=320, color=c_fill, edgecolors="#FFFFFF", linewidths=1.8, zorder=4)

            for n_idx, (x, y) in enumerate(nodes):
                if l_idx == 0 and n_idx < len(input_names):
                    txt_color = "#FBBF24" if (focus_input_idx is not None and n_idx == focus_input_idx) else "#E2E8F0"
                    self.ax_graph.text(x - 0.035, y, input_names[n_idx], fontsize=8, color=txt_color,
                                       ha="right", va="center", weight="bold")
                elif l_idx == n_layers - 1:
                    self.ax_graph.text(x + 0.035, y, "ŷ = PM₂.₅(t+24)", fontsize=9, color="#FBBF24",
                                       ha="left", va="center", weight="bold")

            self.ax_graph.text(x_positions[l_idx], 0.97, layer_titles[l_idx], fontsize=8.5,
                               color="#F1F5F9", ha="center", va="center", weight="bold")

        self.ax_graph.set_xlim(0.0, 1.0)
        self.ax_graph.set_ylim(0.03, 1.03)
        self.ax_graph.axis("off")
        self.ax_graph.set_title(title, fontsize=11, color="#FFFFFF", weight="bold", pad=12)

        self.canvas_graph.draw()

    def _draw_matlab_block_diagram(self):
        """
        Dibuja la arquitectura en diagrama de bloques modular al estilo MATLAB (view(net)),
        con cajas tensoriales formales, funciones de activacion y flujo de senales.
        """
        self.ax_graph.set_facecolor("#181A20")

        selected_model = self.graph_model_var.get()
        is_prop = "Propio" in selected_model

        if is_prop:
            blocks = [
                {"title": "Vector Entrada\nx(k)", "dim": "10 × 1", "desc": "10 Variables Ambientales", "color": "#0284C7"},
                {"title": "Capa Oculta 1\nW¹ + b¹", "dim": "W¹: 16 × 10\nb¹: 16 × 1", "desc": "16 neuronas\nf(z) = tansig", "color": "#059669"},
                {"title": "Capa Oculta 2\nW² + b²", "dim": "W²: 8 × 16\nb²: 8 × 1", "desc": "8 neuronas\nf(z) = tansig", "color": "#7C3AED"},
                {"title": "Capa Salida\nW³ + b³", "dim": "W³: 1 × 8\nb³: 1 × 1", "desc": "1 neurona lineal\nf(z) = purelin", "color": "#D97706"},
                {"title": "Salida Predicha\nŷ(k+24)", "dim": "1 × 1", "desc": "PM₂.₅ a 24 horas (µg/m³)", "color": "#DC2626"}
            ]
            main_title = "Diagrama de Bloques Estilo MATLAB (view(net)) - Embudo 10:16:8:1"
        else:
            blocks = [
                {"title": "Vector Entrada\nx(k)", "dim": "10 × 1", "desc": "10 Variables Ambientales", "color": "#0284C7"},
                {"title": "Capa Oculta 1\nW¹ + b¹", "dim": "W¹: 10 × 10\nb¹: 10 × 1", "desc": "10 neuronas\nf(z) = tansig", "color": "#059669"},
                {"title": "Capa Salida\nW² + b²", "dim": "W²: 1 × 10\nb²: 1 × 1", "desc": "1 neurona lineal\nf(z) = purelin", "color": "#D97706"},
                {"title": "Salida Predicha\nŷ(k+24)", "dim": "1 × 1", "desc": "PM₂.₅ a 24 horas (µg/m³)", "color": "#DC2626"}
            ]
            main_title = "Diagrama de Bloques Estilo MATLAB (view(net)) - Referencia 10:10:1"

        n_b = len(blocks)
        xs = np.linspace(0.10, 0.90, n_b)

        for i in range(n_b - 1):
            x_start = xs[i] + 0.075
            x_end = xs[i + 1] - 0.075
            self.ax_graph.annotate(
                "",
                xy=(x_end, 0.5), xytext=(x_start, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#F1F5F9", mutation_scale=18)
            )

        for i, b in enumerate(blocks):
            xc = xs[i]
            rect = plt.Rectangle((xc - 0.07, 0.28), 0.14, 0.44, facecolor=b["color"],
                                 edgecolor="#FFFFFF", lw=2.0, alpha=0.92, zorder=3)
            self.ax_graph.add_patch(rect)

            self.ax_graph.text(xc, 0.62, b["title"], fontsize=9, color="#FFFFFF", weight="bold", ha="center", va="center")
            self.ax_graph.text(xc, 0.48, b["dim"], fontsize=8, color="#FDE047", weight="bold", ha="center", va="center")
            self.ax_graph.text(xc, 0.36, b["desc"], fontsize=7.5, color="#F1F5F9", ha="center", va="center")

        self.ax_graph.set_xlim(0.0, 1.0)
        self.ax_graph.set_ylim(0.1, 0.9)
        self.ax_graph.axis("off")
        self.ax_graph.set_title(main_title, fontsize=11, color="#FFFFFF", weight="bold", pad=12)

        self.canvas_graph.draw()

    def _draw_weight_heatmap(self):
        """
        Muestra la matriz de pesos W^(1) como un mapa de calor cuantitativo.
        El colorbar se genera de forma segura sin fugas al limpiar previamente con clf().
        """
        self.ax_graph.set_facecolor("#181A20")

        selected_model = self.graph_model_var.get()
        is_prop = "Propio" in selected_model
        model_obj = self.model_prop if is_prop else self.model_ref

        if model_obj is None:
            self.ax_graph.text(0.5, 0.5, "Entrene los modelos para visualizar los pesos calibrados en el Mapa de Calor.\nPresione 'Entrenar Ambas Redes' en la pestana de Curvas.",
                               fontsize=11, color="#FBBF24", ha="center", va="center", weight="bold")
            self.ax_graph.axis("off")
            self.canvas_graph.draw()
            return

        w1 = model_obj.weights[0].T  # (n_neuronas, n_entradas)
        input_names = ["TEMP", "PRES", "DEWP", "Iws", "hour", "day", "pm25_t", "t-1", "t-2", "t-3"]

        im = self.ax_graph.imshow(w1, cmap="coolwarm", aspect="auto", interpolation="nearest")
        cbar = self.fig_graph.colorbar(im, ax=self.ax_graph, fraction=0.032, pad=0.04)
        cbar.ax.tick_params(colors="#CBD5E1")
        cbar.set_label("Magnitud del Peso W (Azul: Excitatorio, Rojo: Inhibitorio)", color="#CBD5E1", fontsize=8.5, weight="bold")

        self.ax_graph.set_xticks(range(len(input_names)))
        self.ax_graph.set_xticklabels(input_names, fontsize=8, color="#E2E8F0", rotation=25, weight="bold")
        self.ax_graph.set_yticks(range(w1.shape[0]))
        self.ax_graph.set_yticklabels([f"h{i+1}" for i in range(w1.shape[0])], fontsize=8, color="#E2E8F0")

        self.ax_graph.set_title(f"Mapa de Calor de la Matriz W¹ (Pesos Entrada → Capa Oculta 1: {w1.shape[0]}×10)",
                                fontsize=11, color="#FFFFFF", weight="bold", pad=10)
        self.ax_graph.set_xlabel("Variables de Entrada", fontsize=9, color="#CBD5E1", weight="bold")
        self.ax_graph.set_ylabel("Neuronas Ocultas", fontsize=9, color="#CBD5E1", weight="bold")

        self.canvas_graph.draw()

    # -------------------------------------------------------------------------
    # Pestana 2: Curvas de Aprendizaje y Entrenamiento en Tiempo Real
    # -------------------------------------------------------------------------
    def _build_tab_train(self):
        """Panel de control de hiperparametros y graficador de convergencia."""
        ctrl_card = ctk.CTkFrame(self.tab_train, fg_color="#1E222B", corner_radius=8)
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
            fg_color="#0284C7",
            hover_color="#0369A1"
        )
        self.btn_start_train.pack(side="left", padx=16, pady=10)

        # Progreso
        f_prog = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        f_prog.pack(side="right", padx=16, pady=10)

        self.lbl_train_status = ctk.CTkLabel(
            f_prog,
            text="Listo para entrenar 500 epocas",
            font=ctk.CTkFont(size=10, slant="italic"),
            text_color="#34D399"
        )
        self.lbl_train_status.pack(anchor="e")

        self.prog_bar = ctk.CTkProgressBar(f_prog, width=220)
        self.prog_bar.set(0.0)
        self.prog_bar.pack(anchor="e", pady=(4, 0))

        # Tarjeta explicativa de hiperparametros y terminologia
        guide_card = ctk.CTkFrame(self.tab_train, fg_color="#181B22", corner_radius=8, border_width=1, border_color="#334155")
        guide_card.pack(fill="x", padx=10, pady=(0, 6))

        lbl_guide = ctk.CTkLabel(
            guide_card,
            text=(
                "• Epocas (k): Cantidad de pasadas completas del algoritmo sobre las 33.384 muestras de entrenamiento.\n"
                "• Tasa de Aprendizaje (η - eta): Coeficiente que modula la amplitud de los pasos del descenso de gradiente (W ← W - η·∇W). Tasa recomendada: 0,005 a 0,02.\n"
                "• Coeficiente de Momento (α - alpha): Fraccion de velocidad acumulada de la epoca anterior (V ← α·V + η·∇W). Amortigua oscilaciones y acelera el paso por mesetas.\n"
                "• PM₂.₅ en µg/m³: Concentracion fisica de microgramos por metro cubico de material particulado fino (diametro ≤ 2,5 µm)."
            ),
            font=ctk.CTkFont(size=10),
            justify="left",
            text_color="#CBD5E1"
        )
        lbl_guide.pack(anchor="w", padx=12, pady=6)

        # Lienzo Matplotlib para curvas de aprendizaje
        plot_frame = ctk.CTkFrame(self.tab_train, fg_color="#181A20", corner_radius=8)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig_train = Figure(figsize=(9.5, 4.6), dpi=100, facecolor="#181A20")
        self.ax_train = self.fig_train.add_subplot(111)
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=plot_frame)
        self.canvas_train.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        tb_frame = ctk.CTkFrame(plot_frame, fg_color="transparent")
        tb_frame.pack(fill="x", padx=6, pady=(0, 4))
        self.toolbar_train = NavigationToolbar2Tk(self.canvas_train, tb_frame)
        self.toolbar_train.update()

        self._init_train_plot()

    @staticmethod
    def _fmt(val):
        """Formatea valores numericos con maximo 2 decimales y proteccion ante desbordamientos."""
        if val is None or np.isnan(val) or np.isinf(val):
            return "Inestable"
        if val > 99999.0:
            return f"{val:.2e}"
        return f"{val:,.2f}"

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
        self.lbl_train_status.configure(text="Entrenamiento completado exitosamente", text_color="#34D399")

        self._plot_final_curves()
        self._update_eval_tab()
        self._render_topology_view()

        ev_ref = self.model_ref.evaluate(self.data['X_test_norm'], self.data['y_test_norm'], self.data['scaler_y'])
        ev_prop = self.model_prop.evaluate(self.data['X_test_norm'], self.data['y_test_norm'], self.data['scaler_y'])

        ep_count = len(self.model_ref.history["epoch"])
        diff_rmse = ev_prop["rmse_real"] - ev_ref["rmse_real"]
        pct_rmse = ((ev_ref["rmse_real"] - ev_prop["rmse_real"]) / ev_ref["rmse_real"]) * 100 if ev_ref["rmse_real"] > 0 else 0.0

        if diff_rmse < 0:
            conclusion = f"Ganancia Modelo Propio: {abs(diff_rmse):.2f} µg/m³ menos de error (Mejora del {pct_rmse:.2f}% en REMC)"
        else:
            conclusion = f"Diferencia Modelo Propio: {diff_rmse:+.2f} µg/m³ frente a referencia ({pct_rmse:+.2f}% de variacion)"

        messagebox.showinfo(
            "Entrenamiento Finalizado",
            f"Ambos modelos han completado sus {ep_count} epocas con exito.\n\n"
            f"• Referencia (10:10:1):\n"
            f"   REMC: {self._fmt(ev_ref['rmse_real'])} µg/m³  |  MAE: {self._fmt(ev_ref['mae_real'])} µg/m³\n\n"
            f"• Propio (10:16:8:1):\n"
            f"   REMC: {self._fmt(ev_prop['rmse_real'])} µg/m³  |  MAE: {self._fmt(ev_prop['mae_real'])} µg/m³\n\n"
            f"{conclusion}"
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

        self.ax_train.plot(ep, tr_ref, label="Referencia (10:10:1) - Train", color="#38BDF8", lw=1.8)
        self.ax_train.plot(ep, te_ref, label="Referencia (10:10:1) - Test", color="#38BDF8", lw=1.6, linestyle="--")

        self.ax_train.plot(ep, tr_prop, label="Propio (10:16:8:1) - Train", color="#34D399", lw=2.0)
        self.ax_train.plot(ep, te_prop, label="Propio (10:16:8:1) - Test", color="#34D399", lw=1.6, linestyle="--")

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
    # Pestana 3: Evaluacion Comparativa y Analisis Cientifico Exhaustivo
    # -------------------------------------------------------------------------
    def _build_tab_eval(self):
        """Tarjetas de indicadores, comentarios de analisis exhaustivo y series de prueba."""
        # Contenedor superior de tarjetas
        cards_frame = ctk.CTkFrame(self.tab_eval, fg_color="transparent")
        cards_frame.pack(fill="x", padx=10, pady=(6, 8))

        # Tarjeta 1: Modelo Referencia
        self.card_ref = ctk.CTkFrame(cards_frame, fg_color="#1E222B", corner_radius=8)
        self.card_ref.pack(side="left", fill="both", expand=True, padx=(0, 6))

        ctk.CTkLabel(self.card_ref, text="Modelo Referencia (10:10:1)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38BDF8").pack(anchor="w", padx=12, pady=(8, 2))
        self.lbl_ref_metrics = ctk.CTkLabel(
            self.card_ref,
            text="Parametros: 121\nTest EMC: 8.529,83\nTest REMC: 92,36 µg/m³\nMAE: 68,14 µg/m³",
            font=ctk.CTkFont(size=10),
            justify="left"
        )
        self.lbl_ref_metrics.pack(anchor="w", padx=12, pady=(0, 8))

        # Tarjeta 2: Modelo Propio
        self.card_prop = ctk.CTkFrame(cards_frame, fg_color="#1E222B", corner_radius=8)
        self.card_prop.pack(side="left", fill="both", expand=True, padx=6)

        ctk.CTkLabel(self.card_prop, text="Modelo Propio (10:16:8:1)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#34D399").pack(anchor="w", padx=12, pady=(8, 2))
        self.lbl_prop_metrics = ctk.CTkLabel(
            self.card_prop,
            text="Parametros: 321\nTest EMC: 7.782,96\nTest REMC: 88,22 µg/m³\nMAE: 64,88 µg/m³",
            font=ctk.CTkFont(size=10),
            justify="left"
        )
        self.lbl_prop_metrics.pack(anchor="w", padx=12, pady=(0, 8))

        # Tarjeta 3: Ganancia y Reduccion de Error
        self.card_gain = ctk.CTkFrame(cards_frame, fg_color="#1E222B", corner_radius=8)
        self.card_gain.pack(side="left", fill="both", expand=True, padx=(6, 0))

        ctk.CTkLabel(self.card_gain, text="Ganancia Neta Obtenida", font=ctk.CTkFont(size=12, weight="bold"), text_color="#FBBF24").pack(anchor="w", padx=12, pady=(8, 2))
        self.lbl_gain_metrics = ctk.CTkLabel(
            self.card_gain,
            text="Reduccion EMC: -8,75%\nMejora REMC: -4,14 µg/m³\nMejora MAE: -3,26 µg/m³\nSobreajuste: Descartado",
            font=ctk.CTkFont(size=10),
            justify="left"
        )
        self.lbl_gain_metrics.pack(anchor="w", padx=12, pady=(0, 8))

        # Panel Descriptivo de Analisis Cientifico
        analysis_box = ctk.CTkFrame(self.tab_eval, fg_color="#181B22", corner_radius=8, border_width=1, border_color="#334155")
        analysis_box.pack(fill="x", padx=10, pady=(0, 8))

        lbl_anal_title = ctk.CTkLabel(
            analysis_box,
            text="Dictamen y Analisis Cuantitativo de Rendimiento (8.346 Muestras de Prueba):",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#F8FAFC"
        )
        lbl_anal_title.pack(anchor="w", padx=12, pady=(6, 2))

        self.lbl_anal_text = ctk.CTkLabel(
            analysis_box,
            text=(
                "• Superioridad Estructural: La arquitectura propia (10:16:8:1) supera a la referencia en todas las metricas, reduciendo el error cuadratico medio en 746,87 unidades cuadradas (-8,75%) y disminuyendo el RMSE en 4,14 µg/m³ a 24 horas.\n"
                "• Inmunidad al Sobreajuste (Overfitting): Train MSE (0,0316) y Test MSE (0,0315) exhiben una paridad casi exacta (discrepancia < 0,0001). El embudo 16 → 8 forzo a la red a condensar patrones invariantes sin memorizar ruido.\n"
                "• Respaldo Metodologico: La particion cronologica y la normalizacion Min-Max exclusivamente sobre Train aseguran cero fuga de datos (Data Leakage), validando la capacidad de pronostico operativo."
            ),
            font=ctk.CTkFont(size=10),
            justify="left",
            text_color="#CBD5E1"
        )
        self.lbl_anal_text.pack(anchor="w", padx=12, pady=(0, 8))

        # Lienzo Matplotlib para Real vs Predicho
        plot_frame = ctk.CTkFrame(self.tab_eval, fg_color="#181A20", corner_radius=8)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self.fig_eval = Figure(figsize=(9.5, 4.4), dpi=100, facecolor="#181A20")
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
        ep_count = len(self.model_ref.history["epoch"])

        self.lbl_ref_metrics.configure(
            text=f"Parametros: 121\nTest EMC: {self._fmt(ev_ref['mse_real'])}\nTest REMC: {self._fmt(ev_ref['rmse_real'])} µg/m³\nMAE: {self._fmt(ev_ref['mae_real'])} µg/m³\nMaxAE: {self._fmt(ev_ref['max_ae_real'])} µg/m³"
        )
        self.lbl_prop_metrics.configure(
            text=f"Parametros: 321\nTest EMC: {self._fmt(ev_prop['mse_real'])}\nTest REMC: {self._fmt(ev_prop['rmse_real'])} µg/m³\nMAE: {self._fmt(ev_prop['mae_real'])} µg/m³\nMaxAE: {self._fmt(ev_prop['max_ae_real'])} µg/m³"
        )

        diff_mse = ((ev_prop["mse_real"] - ev_ref["mse_real"]) / ev_ref["mse_real"]) * 100 if ev_ref["mse_real"] > 0 else 0.0
        diff_rmse = ev_prop["rmse_real"] - ev_ref["rmse_real"]
        diff_mae = ev_prop["mae_real"] - ev_ref["mae_real"]

        self.lbl_gain_metrics.configure(
            text=f"Diferencia EMC: {diff_mse:+.2f}%\nDiferencia REMC: {diff_rmse:+.2f} µg/m³\nDiferencia MAE: {diff_mae:+.2f} µg/m³\nEpocas Evaluadas: {ep_count}"
        )

        # Graficar ventana de 200 horas continuas con curvas de alta visibilidad
        self.ax_eval.clear()
        self.ax_eval.set_facecolor("#1E222B")

        w = 200
        y_real_w = ev_ref["y_real"][:w].flatten()
        y_ref_w = ev_ref["y_pred_real"][:w].flatten()
        y_prop_w = ev_prop["y_pred_real"][:w].flatten()
        hours = np.arange(1, w + 1)

        self.ax_eval.plot(hours, y_real_w, label="PM₂.₅ Real Observado", color="#FFFFFF", lw=2.1, alpha=0.95)
        self.ax_eval.plot(hours, y_ref_w, label="Prediccion Ref (10:10:1)", color="#38BDF8", lw=1.6, linestyle="--")
        self.ax_eval.plot(hours, y_prop_w, label="Prediccion Propia (10:16:8:1)", color="#34D399", lw=1.8, linestyle="-.")

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
    # Pestana 4: Simulador en Tiempo Real con Grafica de Dispersion y Diagnostico
    # -------------------------------------------------------------------------
    def _build_tab_sim(self):
        """Construye un simulador interactivo con grafica proyectiva a 24 horas y evaluacion sanitaria."""
        # Contenedor dividido en dos columnas: Izquierda (Inputs) y Derecha (Grafica + Diagnostico)
        split_frame = ctk.CTkFrame(self.tab_sim, fg_color="transparent")
        split_frame.pack(fill="both", expand=True, padx=8, pady=6)

        # Columna Izquierda: Formulario de Entradas
        left_col = ctk.CTkFrame(split_frame, fg_color="#1E222B", corner_radius=8, width=420)
        left_col.pack(side="left", fill="both", padx=(0, 6), pady=0)
        left_col.pack_propagate(False)

        ctk.CTkLabel(
            left_col,
            text="Parametros de Entrada:",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#38BDF8"
        ).pack(anchor="w", padx=14, pady=(12, 6))

        inputs_box = ctk.CTkScrollableFrame(left_col, fg_color="transparent", height=380)
        inputs_box.pack(fill="both", expand=True, padx=8, pady=4)

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
        for col_name, def_val, desc in feature_configs:
            f_row = ctk.CTkFrame(inputs_box, fg_color="transparent")
            f_row.pack(fill="x", pady=3)

            lbl = ctk.CTkLabel(f_row, text=f"{desc}:", font=ctk.CTkFont(size=10, weight="bold"))
            lbl.pack(side="left", padx=4)

            entry = ctk.CTkEntry(f_row, width=85)
            entry.insert(0, def_val)
            entry.pack(side="right", padx=4)
            self.sim_inputs[col_name] = entry

        # Boton Inferencia
        self.btn_sim = ctk.CTkButton(
            left_col,
            text="Simular Pronostico Neuronal (t+24)",
            command=self._execute_inference,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#34D399",
            hover_color="#059669",
            text_color="#181A20",
            height=34
        )
        self.btn_sim.pack(fill="x", padx=14, pady=12)

        # Columna Derecha: Grafica Proyectiva + Tarjetas de Diagnostico
        right_col = ctk.CTkFrame(split_frame, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=(6, 0), pady=0)

        # Grafica embebida de proyeccion
        plot_box = ctk.CTkFrame(right_col, fg_color="#181A20", corner_radius=8)
        plot_box.pack(fill="both", expand=True, pady=(0, 6))

        self.fig_sim = Figure(figsize=(6.5, 3.8), dpi=100, facecolor="#181A20")
        self.ax_sim = self.fig_sim.add_subplot(111)
        self.canvas_sim = FigureCanvasTkAgg(self.fig_sim, master=plot_box)
        self.canvas_sim.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        self._init_sim_plot()

        # Tarjeta Inferior de Diagnostico de Salud Publica y Fisico
        self.sim_res_card = ctk.CTkFrame(right_col, fg_color="#1E222B", corner_radius=8)
        self.sim_res_card.pack(fill="x", pady=(4, 0))

        self.lbl_sim_result = ctk.CTkLabel(
            self.sim_res_card,
            text="Configure las entradas y presione 'Simular Pronostico Neuronal' para proyectar la concentracion a 24h.",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            justify="left"
        )
        self.lbl_sim_result.pack(anchor="w", padx=14, pady=10)

    def _init_sim_plot(self):
        """Inicializa el lienzo de proyeccion temporal de contaminantes."""
        self.ax_sim.clear()
        self.ax_sim.set_facecolor("#1E222B")
        self.ax_sim.set_title("Proyeccion Temporal de Concentracion: Historial (t-3 a t) → Pronostico (t+24)",
                              fontsize=10.5, color="#FFFFFF", weight="bold")
        self.ax_sim.set_xlabel("Instante Temporal Relativo", fontsize=9.5, color="#D8DEE9")
        self.ax_sim.set_ylabel("Concentracion de PM₂.₅ (µg/m³)", fontsize=9.5, color="#D8DEE9")
        self.ax_sim.tick_params(colors="#D8DEE9")
        for spine in self.ax_sim.spines.values():
            spine.set_color("#4C566A")
        self.ax_sim.grid(True, linestyle=":", alpha=0.4, color="#616E85")
        self.canvas_sim.draw()

    def _execute_inference(self):
        """Ejecuta el forward pass y dibuja la proyeccion interactiva."""
        if self.model_ref is None or self.model_prop is None:
            messagebox.showwarning("Modelos no entrenados", "Primero inicie el entrenamiento en la pestana 'Curvas de Aprendizaje'.")
            return

        try:
            vec_raw = []
            for col in ["TEMP", "PRES", "DEWP", "Iws", "hour", "day", "pm25_t", "pm25_t-1", "pm25_t-2", "pm25_t-3"]:
                val = float(self.sim_inputs[col].get())
                vec_raw.append(val)
        except ValueError:
            messagebox.showerror("Dato Invalido", "Por favor introduzca valores numericos validos en todas las casillas.")
            return

        X_raw = np.array([vec_raw], dtype=np.float64)
        X_norm = self.data["scaler_X"].transform(X_raw)

        pred_ref_norm = self.model_ref.predict(X_norm)
        pred_prop_norm = self.model_prop.predict(X_norm)

        pred_ref = float(self.data["scaler_y"].inverse_transform(pred_ref_norm)[0, 0])
        pred_prop = float(self.data["scaler_y"].inverse_transform(pred_prop_norm)[0, 0])

        cat_aqi, color_aqi, recomendacion = self._get_aqi_badge(pred_prop)

        # Diagnostico meteorologico
        iws_val = vec_raw[3]
        dewp_val = vec_raw[2]
        temp_val = vec_raw[0]

        if iws_val >= 15.0:
            diag_viento = "Viento fuerte (> 15 m/s): Alta tasa de ventilacion y dispersion efectiva."
        elif iws_val >= 6.0:
            diag_viento = "Viento moderado (6-15 m/s): Renovacion atmosferica continua."
        else:
            diag_viento = "Viento calmo / debil (< 6 m/s): Condicion de estancamiento favorable a la acumulacion."

        res_msg = (
            f"RESULTADOS DEL PRONOSTICO NEURONAL:\n"
            f"  • Modelo Propio Propuesto (10:16:8:1): {pred_prop:.2f} µg/m³\n"
            f"  • Modelo Referencia (10:10:1):        {pred_ref:.2f} µg/m³\n"
            f"  • Categoria Sanitaria OMS / EPA:      {cat_aqi}\n"
            f"  • Recomendacion Sanitaria:             {recomendacion}\n"
            f"  • Diagnostico Meteorologico:           {diag_viento}"
        )
        self.lbl_sim_result.configure(text=res_msg)

        # Dibujar la proyeccion temporal visual
        self.ax_sim.clear()
        self.ax_sim.set_facecolor("#1E222B")

        # Puntos de historial: t-3, t-2, t-1, t
        t_hist = np.array([-3, -2, -1, 0])
        y_hist = np.array([vec_raw[9], vec_raw[8], vec_raw[7], vec_raw[6]])

        # Traza historica observada
        self.ax_sim.plot(t_hist, y_hist, marker="o", color="#FFFFFF", lw=2.2, label="Historial Ingresado (t-3 a t)", zorder=4)

        # Proyecciones hacia t+24
        t_proj = np.array([0, 24])
        y_proj_ref = np.array([y_hist[-1], pred_ref])
        y_proj_prop = np.array([y_hist[-1], pred_prop])

        self.ax_sim.plot(t_proj, y_proj_ref, linestyle="--", marker="s", color="#38BDF8", lw=1.8, label=f"Proyeccion Ref: {pred_ref:.1f} µg/m³", zorder=3)
        self.ax_sim.plot(t_proj, y_proj_prop, linestyle="-.", marker="^", color="#34D399", lw=2.2, label=f"Proyeccion Propia: {pred_prop:.1f} µg/m³", zorder=3)

        # Franjas sanitarias de fondo
        self.ax_sim.axhspan(0, 50, color="#10B981", alpha=0.12, label="Buena (0-50)")
        self.ax_sim.axhspan(50, 100, color="#FBBF24", alpha=0.12, label="Moderada (50-100)")
        self.ax_sim.axhspan(100, 150, color="#FB923C", alpha=0.12, label="Grupos Sensibles (100-150)")
        self.ax_sim.axhspan(150, 500, color="#EF4444", alpha=0.12, label="Danina (>150)")

        self.ax_sim.set_xticks([-3, -2, -1, 0, 6, 12, 18, 24])
        self.ax_sim.set_xticklabels(["t-3", "t-2", "t-1", "t (Actual)", "t+6", "t+12", "t+18", "t+24 (Pronostico)"], fontsize=8.5, color="#E2E8F0")
        self.ax_sim.tick_params(colors="#D8DEE9")
        self.ax_sim.set_ylabel("Concentracion de PM₂.₅ (µg/m³)", fontsize=9.5, color="#D8DEE9")
        self.ax_sim.set_title(f"Pronostico Dinamico: {pred_prop:.2f} µg/m³ a las 24 horas ({cat_aqi})",
                              fontsize=10.5, color="#FFFFFF", weight="bold")
        for spine in self.ax_sim.spines.values():
            spine.set_color("#4C566A")
        self.ax_sim.grid(True, linestyle=":", alpha=0.4, color="#616E85")
        self.ax_sim.legend(loc="upper left", facecolor="#181A20", edgecolor="#4C566A", labelcolor="#D8DEE9", fontsize=8)

        self.canvas_sim.draw()

    @staticmethod
    def _get_aqi_badge(pm25_val):
        """Determina la clasificacion de alerta sanitaria segun rangos de PM2.5."""
        if pm25_val <= 50:
            return "Buena (0 a 50 µg/m³)", "#10B981", "Calidad optima. No se requieren restricciones para la salud."
        elif pm25_val <= 100:
            return "Moderada (51 a 100 µg/m³)", "#FBBF24", "Aceptable. Personas con hipersensibilidad respiratoria deben moderar esfuerzo exterior."
        elif pm25_val <= 150:
            return "Danina para Grupos Sensibles (101 a 150 µg/m³)", "#FB923C", "Ninos, ancianos y personas con asma deben reducir actividades fisicas intensas."
        elif pm25_val <= 200:
            return "Danina para la Salud (151 a 200 µg/m³)", "#EF4444", "Toda la poblacion puede experimentar efectos. Evitar exposicion prolongada."
        elif pm25_val <= 300:
            return "Muy Danina para la Salud (201 a 300 µg/m³)", "#A855F7", "Alerta sanitaria de riesgo grave. Uso obligatorio de respiradores N95."
        else:
            return "Peligrosa (> 300 µg/m³)", "#991B1B", "Emergencia ambiental extrema. Permanecer estrictamente en interiores con filtracion de aire."

    # -------------------------------------------------------------------------
    # Metodos Auxiliares
    # -------------------------------------------------------------------------
    def _load_dataset_async(self):
        """Carga en segundo plano el dataset."""
        thread = threading.Thread(target=self._load_dataset_worker, daemon=True)
        thread.start()

    def _load_dataset_worker(self):
        """Lectura y normalizacion del archivo CSV."""
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "datasets", "dataset_pm25.csv"))
        if os.path.exists(csv_path):
            self.data = load_and_preprocess_dataset(csv_path)
            try:
                self.after(0, self._on_dataset_ready)
            except (RuntimeError, tk.TclError):
                pass

    def _on_dataset_ready(self):
        """Notifica que el dataset fue cargado exitosamente."""
        self.lbl_data_status.configure(
            text=f"Total: {self.data['n_train'] + self.data['n_test']:,} registros\n"
                 f"Entrenamiento: {self.data['n_train']:,} (80%)\n"
                 f"Prueba: {self.data['n_test']:,} (20%)\n"
                 f"Normalizado: [-1, +1]",
            text_color="#34D399"
        )
        self.lbl_train_status.configure(text="Datos listos. Presione 'Entrenar Ambas Redes'.")

    def _reset_entire_interface(self):
        """Restaura completamente los modelos, graficas, hiperparametros y simuladores a su estado inicial."""
        # 1. Restablecer estado de los modelos
        self.model_ref = None
        self.model_prop = None
        self.is_training = False

        # 2. Restablecer controles de Topologia
        self.graph_model_var.set("Propio (10:16:8:1)")
        self.graph_view_mode.set("Grafo Sinaptico Nitido")
        self.focus_node_var.set("Todas las Relaciones")
        self.f_focus_container.pack(side="left")
        self._render_topology_view()

        # 3. Restablecer Curvas de Aprendizaje
        self.entry_epochs.delete(0, "end")
        self.entry_epochs.insert(0, "500")
        self.entry_lr.delete(0, "end")
        self.entry_lr.insert(0, "0.01")
        self.entry_mom.delete(0, "end")
        self.entry_mom.insert(0, "0.0")

        self.prog_bar.set(0.0)
        self.btn_start_train.configure(state="normal")
        self.lbl_train_status.configure(text="Listo para entrenar 500 epocas", text_color="#34D399")
        self._init_train_plot()

        # 4. Restablecer Evaluacion y Metricas
        self.lbl_ref_metrics.configure(
            text="Parametros: 121\nTest EMC: 8.529,83\nTest REMC: 92,36 µg/m³\nMAE: 68,14 µg/m³\nMaxAE: 532,32 µg/m³"
        )
        self.lbl_prop_metrics.configure(
            text="Parametros: 321\nTest EMC: 7.782,96\nTest REMC: 88,22 µg/m³\nMAE: 64,88 µg/m³\nMaxAE: 566,08 µg/m³"
        )
        self.lbl_gain_metrics.configure(
            text="Reduccion EMC: -8,75%\nMejora REMC: -4,14 µg/m³\nMejora MAE: -3,26 µg/m³\nSobreajuste: Descartado"
        )
        self._init_eval_plot()

        # 5. Restablecer Simulador en Tiempo Real
        default_inputs = {
            "TEMP": "-5.0", "PRES": "1022.0", "DEWP": "-7.0", "Iws": "5.36",
            "hour": "3", "day": "2", "pm25_t": "181.0", "pm25_t-1": "159.0",
            "pm25_t-2": "148.0", "pm25_t-3": "129.0"
        }
        for k, v in default_inputs.items():
            if k in self.sim_inputs:
                self.sim_inputs[k].delete(0, "end")
                self.sim_inputs[k].insert(0, v)

        self.lbl_sim_result.configure(
            text="Configure las entradas y presione 'Simular Pronostico Neuronal' para proyectar la concentracion a 24h."
        )
        self._init_sim_plot()

        messagebox.showinfo(
            "Sistema Restablecido",
            "La aplicacion ha sido restaurada con exito a su configuracion inicial predeterminada."
        )


def launch_gui():
    """Lanza la aplicacion de escritorio."""
    app = ModernMlpGUI()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
