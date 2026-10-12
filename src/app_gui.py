"""
Sistema Inteligente MLP - Pronostico de Concentracion de PM2.5 a 24 Horas
Universidad de Cartagena - Facultad de Ingenieria
Autores: Dago David Palmera Navarro (0222321003) & Julian David Camargo Padilla (0222320016)

Modulo Principal de Interfaz Grafica (GUI) - Arquitectura Modular y Optimizada.
"""

import os
import queue
import threading
from tkinter import messagebox
from PIL import Image
import customtkinter as ctk
import numpy as np

from data_loader import load_and_preprocess_dataset
from mlp import MultilayerPerceptron
import gui_theme as theme
from gui_components import MetricCard, SectionBanner, embed_plot
import gui_views as views


class ModernMlpGUI(ctk.CTk):
    """Ventana principal de la aplicacion en CustomTkinter."""

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Sistema Inteligente MLP - Pronostico de PM₂.₅ (Universidad de Cartagena)")
        self.geometry("1340x840")
        self.minsize(1200, 750)
        self.configure(fg_color=theme.BG_MAIN)

        self.data = None
        self.model_ref = None
        self.model_prop = None
        self.eval_cache_ref = None
        self.eval_cache_prop = None
        self.is_training = False
        self.msg_queue = queue.Queue()

        self._build_sidebar()
        self._build_main_container()
        self._load_dataset_async()
        self._poll_queue()

    def _poll_queue(self):
        """Procesa mensajes de hilos secundarios en el hilo principal de Tkinter."""
        while not self.msg_queue.empty():
            try:
                msg = self.msg_queue.get_nowait()
                kind = msg[0]
                if kind == "dataset_ready":
                    self._on_dataset_ready()
                elif kind == "update_train":
                    self._update_train_ui(msg[1], msg[2], msg[3])
                elif kind == "finish_train":
                    self._finish_train_ui()
            except queue.Empty:
                break
        self.after(50, self._poll_queue)

    def _build_sidebar(self):
        """Construye panel lateral con logo institucional, autores, estado del dataset y boton de reinicio."""
        self.sidebar = ctk.CTkFrame(self, width=290, corner_radius=0, fg_color=theme.BG_MAIN)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)

        title_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_box.pack(fill="x", padx=16, pady=(16, 10))

        # Logo Institucional Universidad de Cartagena
        logo_path = os.path.join(os.path.dirname(__file__), "..", "docs", "img", "logo_udec.png")
        if os.path.exists(logo_path):
            try:
                pil_logo = Image.open(logo_path)
                self.logo_img = ctk.CTkImage(light_image=pil_logo, dark_image=pil_logo, size=(84, 74))
                lbl_logo = ctk.CTkLabel(title_box, image=self.logo_img, text="")
                lbl_logo.pack(anchor="w", pady=(0, 10))
            except Exception:
                pass

        ctk.CTkLabel(title_box, text="UNIVERSIDAD DE CARTAGENA\nFACULTAD DE INGENIERIA",
                     font=ctk.CTkFont(family=theme.FONT_FAMILY, size=12, weight="bold"),
                     text_color=theme.COLOR_PRIMARY).pack(anchor="w")

        ctk.CTkLabel(title_box, text="Redes Neuronales Multicapa\n(MLP - Corte II)",
                     font=ctk.CTkFont(family=theme.FONT_FAMILY, size=15, weight="bold"),
                     text_color=theme.TEXT_TITLE).pack(anchor="w", pady=(8, 0))

        self.card_authors = MetricCard(
            self.sidebar, title="Investigadores:",
            initial_text="• Dago D. Palmera Navarro\n  (0222321003)\n• Julian D. Camargo Padilla\n  (0222320016)",
            title_color=theme.COLOR_SUCCESS, text_color="#D8DEE9",
            title_size=12, body_size=11
        )
        self.card_authors.pack(fill="x", padx=16, pady=10)

        self.card_dataset = MetricCard(
            self.sidebar, title="Dataset de Beijing:",
            initial_text="Cargando 41.730 registros...\nParticion: 80% Tr / 20% Te\nEscala: [-1, +1] simetrica",
            title_color=theme.COLOR_WARNING, text_color="#A3BE8C",
            title_size=12, body_size=11
        )
        self.card_dataset.pack(fill="x", padx=16, pady=6)

        self.btn_reset_all = ctk.CTkButton(
            self.sidebar, text="↺ Restablecer / Limpiar Todo", command=self._reset_entire_interface,
            font=ctk.CTkFont(family=theme.FONT_FAMILY, size=12, weight="bold"),
            fg_color="#334155", hover_color="#475569", border_width=1, border_color="#475569",
            text_color="#F8FAFC", height=42
        )
        self.btn_reset_all.pack(side="bottom", fill="x", padx=16, pady=16)

    def _build_main_container(self):
        """Inicializa las 4 pestañas de la aplicacion."""
        self.tabview = ctk.CTkTabview(self, fg_color="transparent")
        self.tabview.pack(side="right", fill="both", expand=True, padx=12, pady=10)

        self.tab_graph = self.tabview.add("Topologia y Relaciones")
        self.tab_train = self.tabview.add("Curvas de Aprendizaje")
        self.tab_eval = self.tabview.add("Evaluacion y Metricas")
        self.tab_sim = self.tabview.add("Simulador en Tiempo Real")

        self._build_tab_graph()
        self._build_tab_train()
        self._build_tab_eval()
        self._build_tab_sim()

    def _build_tab_graph(self):
        """Pestaña 1: Grafo sinaptico, diagrama MATLAB y heatmap de pesos."""
        ctrl = ctk.CTkFrame(self.tab_graph, fg_color=theme.BG_CARD, corner_radius=8)
        ctrl.pack(fill="x", padx=10, pady=(6, 8))

        f_left = ctk.CTkFrame(ctrl, fg_color="transparent")
        f_left.pack(side="left", padx=12, pady=8)

        ctk.CTkLabel(f_left, text="Modelo:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=(0, 6))
        self.graph_model_var = ctk.StringVar(value="Propio (10:16:8:1)")
        self.cb_model = ctk.CTkComboBox(f_left, values=["Propio (10:16:8:1)", "Referencia (10:10:1)"],
                                        variable=self.graph_model_var, width=170, state="readonly",
                                        command=lambda _: self._render_topology_view())
        self.cb_model.pack(side="left", padx=4)

        ctk.CTkLabel(f_left, text="Vista:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=(12, 6))
        self.graph_view_mode = ctk.StringVar(value="Grafo Sinaptico Nitido")
        self.cb_view_mode = ctk.CTkComboBox(f_left, values=["Grafo Sinaptico Nitido", "Diagrama de Bloques MATLAB", "Mapa de Calor de Pesos (Heatmap)"],
                                            variable=self.graph_view_mode, width=220, state="readonly",
                                            command=self._on_view_mode_changed)
        self.cb_view_mode.pack(side="left", padx=4)

        self.f_focus_container = ctk.CTkFrame(f_left, fg_color="transparent")
        self.f_focus_container.pack(side="left", padx=(12, 0))

        ctk.CTkLabel(self.f_focus_container, text="Enfocar Entrada:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=(0, 6))
        self.focus_node_var = ctk.StringVar(value="Todas las Relaciones")
        features_list = [
            "Todas las Relaciones", "x₁ (TEMP)", "x₂ (PRES)", "x₃ (DEWP)", "x₄ (Iws)",
            "x₅ (hour)", "x₆ (day)", "x₇ (PM₂.₅ ᵗ)", "x₈ (PM₂.₅ ᵗ⁻¹)", "x₉ (PM₂.₅ ᵗ⁻²)", "x₁₀ (PM₂.₅ ᵗ⁻³)"
        ]
        self.cb_focus = ctk.CTkComboBox(self.f_focus_container, values=features_list, variable=self.focus_node_var,
                                        width=160, state="readonly", command=lambda _: self._render_topology_view())
        self.cb_focus.pack(side="left", padx=4)

        self.lbl_graph_params = ctk.CTkLabel(ctrl, text="Parametros Libres: 321  |  Topologia: 10 → 16 → 8 → 1",
                                            font=ctk.CTkFont(size=11, weight="bold"), text_color=theme.COLOR_SUCCESS)
        self.lbl_graph_params.pack(side="right", padx=16, pady=8)

        p_frame = ctk.CTkFrame(self.tab_graph, fg_color=theme.BG_PLOT, corner_radius=8)
        p_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.fig_graph, self.ax_graph = theme.create_dark_figure(figsize=(9.5, 5.2))
        self.canvas_graph, _ = embed_plot(p_frame, self.fig_graph)
        self._render_topology_view()

    def _on_view_mode_changed(self, mode):
        """Oculta o muestra el filtro de enfocar entrada de forma limpia."""
        if mode == "Grafo Sinaptico Nitido":
            self.f_focus_container.pack(side="left", padx=(12, 0))
        else:
            self.f_focus_container.pack_forget()
        self._render_topology_view()

    def _render_topology_view(self):
        """Dibuja la vista de topologia correspondiente."""
        mode = self.graph_view_mode.get()
        is_prop = "Propio" in self.graph_model_var.get()
        model_obj = self.model_prop if is_prop else self.model_ref

        txt_p = "Parametros Libres: 321  |  Topologia: 10 → 16 → 8 → 1" if is_prop else "Parametros Libres: 121  |  Topologia: 10 → 10 → 1"
        col_p = theme.COLOR_SUCCESS if is_prop else theme.COLOR_PRIMARY
        self.lbl_graph_params.configure(text=txt_p, text_color=col_p)

        self.fig_graph.clf()
        self.fig_graph.set_facecolor(theme.BG_PLOT)
        self.ax_graph = self.fig_graph.add_subplot(111)

        if mode == "Grafo Sinaptico Nitido":
            views.draw_synaptic_graph(self.ax_graph, self.canvas_graph, model_obj, is_prop, self.focus_node_var.get())
        elif mode == "Diagrama de Bloques MATLAB":
            views.draw_matlab_diagram(self.ax_graph, self.canvas_graph, is_prop)
        elif mode == "Mapa de Calor de Pesos (Heatmap)":
            self.ax_graph = views.draw_weight_heatmap(self.fig_graph, self.ax_graph, self.canvas_graph, model_obj)

    def _build_tab_train(self):
        """Pestaña 2: Control de hiperparametros y graficador de curvas."""
        ctrl = ctk.CTkFrame(self.tab_train, fg_color=theme.BG_CARD, corner_radius=8)
        ctrl.pack(fill="x", padx=10, pady=(6, 8))

        f_in = ctk.CTkFrame(ctrl, fg_color="transparent")
        f_in.pack(side="left", padx=12, pady=10)

        ctk.CTkLabel(f_in, text="Epocas:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, padx=6)
        self.entry_epochs = ctk.CTkEntry(f_in, width=70)
        self.entry_epochs.insert(0, "500")
        self.entry_epochs.grid(row=0, column=1, padx=4)

        ctk.CTkLabel(f_in, text="Tasa Aprendizaje (η):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=2, padx=8)
        self.entry_lr = ctk.CTkEntry(f_in, width=70)
        self.entry_lr.insert(0, "0.01")
        self.entry_lr.grid(row=0, column=3, padx=4)

        ctk.CTkLabel(f_in, text="Momento (α):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=4, padx=8)
        self.entry_mom = ctk.CTkEntry(f_in, width=70)
        self.entry_mom.insert(0, "0.0")
        self.entry_mom.grid(row=0, column=5, padx=4)

        self.btn_start_train = ctk.CTkButton(
            ctrl, text="Entrenar Ambas Redes (Full-Batch)", command=self._start_training_worker,
            font=ctk.CTkFont(size=11, weight="bold"), fg_color="#0284C7", hover_color="#0369A1"
        )
        self.btn_start_train.pack(side="left", padx=16, pady=10)

        f_prog = ctk.CTkFrame(ctrl, fg_color="transparent")
        f_prog.pack(side="right", padx=16, pady=10)

        self.lbl_train_status = ctk.CTkLabel(f_prog, text="Listo para entrenar 500 epocas",
                                             font=ctk.CTkFont(size=10, slant="italic"), text_color=theme.COLOR_SUCCESS)
        self.lbl_train_status.pack(anchor="e")
        self.prog_bar = ctk.CTkProgressBar(f_prog, width=220)
        self.prog_bar.set(0.0)
        self.prog_bar.pack(anchor="e", pady=(4, 0))

        guide_msg = (
            "• Epocas (k): Cantidad de pasadas completas del algoritmo sobre las 33.384 muestras de entrenamiento.\n"
            "• Tasa de Aprendizaje (η): Escalar que modula la amplitud de paso del gradiente (W ← W - η·∇W). Sugerido: 0,005 - 0,02.\n"
            "• Coeficiente de Momento (α): Amortigua oscilaciones e inyecta velocidad acumulada (V ← α·V + η·∇W)."
        )
        SectionBanner(self.tab_train, title="Parametros de Optimizacion y Dinamica de Convergencia:", body=guide_msg).pack(fill="x", padx=10, pady=(0, 6))

        p_frame = ctk.CTkFrame(self.tab_train, fg_color=theme.BG_PLOT, corner_radius=8)
        p_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.fig_train, self.ax_train = theme.create_dark_figure(figsize=(9.5, 4.6))
        self.canvas_train, _ = embed_plot(p_frame, self.fig_train)
        self._init_train_plot()

    def _init_train_plot(self):
        self.ax_train.clear()
        self.ax_train.set_facecolor(theme.BG_CARD)
        self.ax_train.set_title("Curvas de Aprendizaje Comparativas (EMC vs. Epocas)", fontsize=11, color=theme.TEXT_TITLE, weight="bold")
        self.ax_train.set_xlabel("Epoca de Entrenamiento (k)", fontsize=10, color=theme.TEXT_BODY)
        self.ax_train.set_ylabel("Error Cuadratico Medio (MSE normalizado)", fontsize=10, color=theme.TEXT_BODY)
        self.canvas_train.draw_idle()

    def _start_training_worker(self):
        if self.data is None:
            messagebox.showwarning("Cargando Dataset", "El conjunto de datos todavia se esta procesando.")
            return
        if self.is_training:
            return
        try:
            epochs = int(self.entry_epochs.get())
            lr = float(self.entry_lr.get())
            mom = float(self.entry_mom.get())
            if epochs <= 0 or lr <= 0 or mom < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Parametros Invalidos", "Verifique que epocas > 0, tasa > 0 y momento >= 0.")
            return

        self.is_training = True
        self.btn_start_train.configure(state="disabled")
        self.lbl_train_status.configure(text="Entrenando redes...", text_color=theme.COLOR_WARNING)
        self.prog_bar.set(0.0)
        threading.Thread(target=self._run_train_loop, args=(epochs, lr, mom), daemon=True).start()

    def _run_train_loop(self, epochs, lr, mom):
        """Bucle de entrenamiento optimizado con zancada periodica de evaluacion de test."""
        self.model_ref = MultilayerPerceptron(topology=[10, 10, 1], lr=lr, momentum=mom, seed=42)
        self.model_prop = MultilayerPerceptron(topology=[10, 16, 8, 1], lr=lr, momentum=mom, seed=42)

        X_tr, y_tr = self.data["X_train_norm"], self.data["y_train_norm"]
        X_te, y_te = self.data["X_test_norm"], self.data["y_test_norm"]

        last_te_ref = float(np.mean((y_te - self.model_ref.predict(X_te)) ** 2))
        last_te_prop = float(np.mean((y_te - self.model_prop.predict(X_te)) ** 2))

        for ep in range(1, epochs + 1):
            tr_mse_ref = self.model_ref.train_epoch(X_tr, y_tr)
            tr_mse_prop = self.model_prop.train_epoch(X_tr, y_tr)

            if ep % 10 == 0 or ep == epochs:
                last_te_ref = float(np.mean((y_te - self.model_ref.predict(X_te)) ** 2))
                last_te_prop = float(np.mean((y_te - self.model_prop.predict(X_te)) ** 2))

            self.model_ref.history["epoch"].append(ep)
            self.model_ref.history["train_mse"].append(tr_mse_ref)
            self.model_ref.history["test_mse"].append(last_te_ref)

            self.model_prop.history["epoch"].append(ep)
            self.model_prop.history["train_mse"].append(tr_mse_prop)
            self.model_prop.history["test_mse"].append(last_te_prop)

            if ep % 25 == 0 or ep == epochs:
                self.msg_queue.put(("update_train", ep, epochs, ep / epochs))

        self.msg_queue.put(("finish_train",))

    def _update_train_ui(self, ep, epochs, prog_val):
        self.prog_bar.set(prog_val)
        self.lbl_train_status.configure(text=f"Epoca {ep}/{epochs} ({int(prog_val*100)}%)")

    def _finish_train_ui(self):
        self.is_training = False
        self.btn_start_train.configure(state="normal")
        self.lbl_train_status.configure(text="Entrenamiento completado exitosamente", text_color=theme.COLOR_SUCCESS)

        self.eval_cache_ref = self.model_ref.evaluate(self.data["X_test_norm"], self.data["y_test_norm"], self.data["scaler_y"])
        self.eval_cache_prop = self.model_prop.evaluate(self.data["X_test_norm"], self.data["y_test_norm"], self.data["scaler_y"])

        views.draw_learning_curves(self.ax_train, self.canvas_train, self.model_ref, self.model_prop)
        self._update_eval_tab()
        self._render_topology_view()

        ev_r, ev_p = self.eval_cache_ref, self.eval_cache_prop
        diff_rmse = ev_p["rmse_real"] - ev_r["rmse_real"]
        pct = ((ev_r["rmse_real"] - ev_p["rmse_real"]) / ev_r["rmse_real"]) * 100 if ev_r["rmse_real"] > 0 else 0.0
        concl = (f"Ganancia Modelo Propio: {abs(diff_rmse):.2f} µg/m³ menos de error ({pct:.2f}% de mejora)"
                 if diff_rmse < 0 else f"Diferencia: {diff_rmse:+.2f} µg/m³ ({pct:+.2f}%)")

        messagebox.showinfo("Entrenamiento Finalizado",
                            f"Modelos calibrados con exito.\n\n"
                            f"• Referencia: REMC = {theme.format_number(ev_r['rmse_real'])} µg/m³  |  MAE = {theme.format_number(ev_r['mae_real'])} µg/m³\n"
                            f"• Propio:     REMC = {theme.format_number(ev_p['rmse_real'])} µg/m³  |  MAE = {theme.format_number(ev_p['mae_real'])} µg/m³\n\n"
                            f"{concl}")

    def _build_tab_eval(self):
        """Pestaña 3: Evaluacion cuantitativa, selector de graficas e histograma MATLAB."""
        cards = ctk.CTkFrame(self.tab_eval, fg_color="transparent")
        cards.pack(fill="x", padx=10, pady=(6, 8))

        self.card_ref = MetricCard(cards, title="Modelo Referencia (10:10:1)",
                                   initial_text="Parametros: 121\nTest EMC: 8.529,83\nTest REMC: 92,36 µg/m³\nMAE: 68,14 µg/m³\nMaxAE: 532,32 µg/m³\nMinAE: 0,02 µg/m³",
                                   title_color=theme.COLOR_PRIMARY)
        self.card_ref.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.card_prop = MetricCard(cards, title="Modelo Propio (10:16:8:1)",
                                    initial_text="Parametros: 321\nTest EMC: 7.782,96\nTest REMC: 88,22 µg/m³\nMAE: 64,88 µg/m³\nMaxAE: 566,08 µg/m³\nMinAE: 0,01 µg/m³",
                                    title_color=theme.COLOR_SUCCESS)
        self.card_prop.pack(side="left", fill="both", expand=True, padx=6)

        self.card_gain = MetricCard(cards, title="Ganancia Neta Obtenida",
                                    initial_text="Reduccion EMC: -8,75%\nMejora REMC: -4,14 µg/m³\nMejora MAE: -3,26 µg/m³\nSobreajuste: Descartado",
                                    title_color=theme.COLOR_WARNING)
        self.card_gain.pack(side="left", fill="both", expand=True, padx=(6, 0))

        eval_ctrl = ctk.CTkFrame(self.tab_eval, fg_color=theme.BG_CARD, corner_radius=8)
        eval_ctrl.pack(fill="x", padx=10, pady=(0, 6))
        ctk.CTkLabel(eval_ctrl, text="Tipo de Grafica:", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=(12, 6), pady=6)
        self.eval_mode_var = ctk.StringVar(value="Serie Temporal (200 Horas)")
        self.cb_eval_mode = ctk.CTkComboBox(eval_ctrl, values=["Serie Temporal (200 Horas)", "Dispersion Real vs Predicho", "Histograma de Errores (MATLAB)"],
                                            variable=self.eval_mode_var, width=240, state="readonly", command=lambda _: self._redraw_eval_view())
        self.cb_eval_mode.pack(side="left", padx=4, pady=6)

        anal_text = (
            "• Superioridad: La red propia (10:16:8:1) reduce el MSE en 746,87 unidades (-8,75%) y mejora el RMSE en 4,14 µg/m³ a 24h.\n"
            "• Inmunidad al Sobreajuste: Train MSE y Test MSE se alinean con discrepancia < 0,0001 gracias al embudo 16 → 8.\n"
            "• Evaluacion Residual (MATLAB): El histograma ratifica distribucion unimodal centrada en cero sin sesgo persistente."
        )
        SectionBanner(self.tab_eval, title="Dictamen y Analisis Cuantitativo de Rendimiento (8.346 Muestras de Prueba):", body=anal_text).pack(fill="x", padx=10, pady=(0, 6))

        p_frame = ctk.CTkFrame(self.tab_eval, fg_color=theme.BG_PLOT, corner_radius=8)
        p_frame.pack(fill="both", expand=True, padx=10, pady=4)
        self.fig_eval, self.ax_eval = theme.create_dark_figure(figsize=(9.5, 4.4))
        self.canvas_eval, _ = embed_plot(p_frame, self.fig_eval)
        self._init_eval_plot()

    def _init_eval_plot(self):
        self.ax_eval.clear()
        self.ax_eval.set_facecolor(theme.BG_CARD)
        self.ax_eval.set_title("Concentracion Real de PM₂.₅ frente a Prediccion a 24 Horas", fontsize=11, color=theme.TEXT_TITLE, weight="bold")
        self.ax_eval.set_xlabel("Horas de Prueba Consecutivas (Muestra Temporal)", fontsize=10, color=theme.TEXT_BODY)
        self.ax_eval.set_ylabel("Concentracion de PM₂.₅ (µg/m³)", fontsize=10, color=theme.TEXT_BODY)
        self.canvas_eval.draw_idle()

    def _update_eval_tab(self):
        if self.eval_cache_ref is None or self.eval_cache_prop is None:
            return
        er, ep = self.eval_cache_ref, self.eval_cache_prop
        self.card_ref.update_text(
            f"Parametros: 121\nTest EMC: {theme.format_number(er['mse_real'])}\nTest REMC: {theme.format_number(er['rmse_real'])} µg/m³\nMAE: {theme.format_number(er['mae_real'])} µg/m³\nMaxAE: {theme.format_number(er['max_ae_real'])} µg/m³\nMinAE: {theme.format_number(er.get('min_ae_real', 0))} µg/m³"
        )
        self.card_prop.update_text(
            f"Parametros: 321\nTest EMC: {theme.format_number(ep['mse_real'])}\nTest REMC: {theme.format_number(ep['rmse_real'])} µg/m³\nMAE: {theme.format_number(ep['mae_real'])} µg/m³\nMaxAE: {theme.format_number(ep['max_ae_real'])} µg/m³\nMinAE: {theme.format_number(ep.get('min_ae_real', 0))} µg/m³"
        )
        dmse = ((ep["mse_real"] - er["mse_real"]) / er["mse_real"]) * 100 if er["mse_real"] > 0 else 0.0
        self.card_gain.update_text(
            f"Diferencia EMC: {dmse:+.2f}%\nDiferencia REMC: {ep['rmse_real'] - er['rmse_real']:+.2f} µg/m³\nDiferencia MAE: {ep['mae_real'] - er['mae_real']:+.2f} µg/m³\nEpocas Evaluadas: {len(self.model_ref.history['epoch'])}"
        )
        self._redraw_eval_view()

    def _redraw_eval_view(self):
        if self.eval_cache_ref is None or self.eval_cache_prop is None:
            return
        m = "series" if "Serie Temporal" in self.eval_mode_var.get() else ("dispersion" if "Dispersion" in self.eval_mode_var.get() else "hist")
        views.draw_evaluation_view(self.ax_eval, self.canvas_eval, self.eval_cache_ref, self.eval_cache_prop, mode=m)

    def _build_tab_sim(self):
        """Pestaña 4: Simulador proyectivo a 24 horas y diagnostico de calidad del aire."""
        split = ctk.CTkFrame(self.tab_sim, fg_color="transparent")
        split.pack(fill="both", expand=True, padx=8, pady=6)

        left = ctk.CTkFrame(split, fg_color=theme.BG_CARD, corner_radius=8, width=420)
        left.pack(side="left", fill="both", padx=(0, 6), pady=0)
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="Parametros de Entrada:", font=ctk.CTkFont(family=theme.FONT_FAMILY, size=12, weight="bold"),
                     text_color=theme.COLOR_PRIMARY).pack(anchor="w", padx=14, pady=(12, 6))

        in_box = ctk.CTkScrollableFrame(left, fg_color="transparent", height=380)
        in_box.pack(fill="both", expand=True, padx=8, pady=4)

        configs = [
            ("TEMP", "-5.0", "Temperatura (°C)"), ("PRES", "1022.0", "Presion (hPa)"),
            ("DEWP", "-7.0", "Punto Rocio (°C)"), ("Iws", "5.36", "Viento (m/s)"),
            ("hour", "3", "Hora (0-23)"), ("day", "2", "Dia (1-31)"),
            ("pm25_t", "181.0", "PM₂.₅ Actual (µg/m³)"), ("pm25_t-1", "159.0", "PM₂.₅ t-1 (µg/m³)"),
            ("pm25_t-2", "148.0", "PM₂.₅ t-2 (µg/m³)"), ("pm25_t-3", "129.0", "PM₂.₅ t-3 (µg/m³)")
        ]
        self.sim_inputs = {}
        for col, val, desc in configs:
            row = ctk.CTkFrame(in_box, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=f"{desc}:", font=ctk.CTkFont(size=10, weight="bold")).pack(side="left", padx=4)
            entry = ctk.CTkEntry(row, width=85)
            entry.insert(0, val)
            entry.pack(side="right", padx=4)
            self.sim_inputs[col] = entry

        ctk.CTkButton(left, text="Simular Pronostico Neuronal (t+24)", command=self._execute_inference,
                      font=ctk.CTkFont(size=11, weight="bold"), fg_color=theme.COLOR_SUCCESS, hover_color="#059669",
                      text_color="#181A20", height=34).pack(fill="x", padx=14, pady=12)

        right = ctk.CTkFrame(split, fg_color="transparent")
        right.pack(side="right", fill="both", expand=True, padx=(6, 0), pady=0)

        p_box = ctk.CTkFrame(right, fg_color=theme.BG_PLOT, corner_radius=8)
        p_box.pack(fill="both", expand=True, pady=(0, 6))
        self.fig_sim, self.ax_sim = theme.create_dark_figure(figsize=(6.5, 3.8))
        self.canvas_sim, _ = embed_plot(p_box, self.fig_sim, with_toolbar=False)
        self._init_sim_plot()

        self.sim_res_card = MetricCard(right, title="Diagnostico Sanitario y Fisico:",
                                       initial_text="Configure las entradas y presione 'Simular Pronostico Neuronal'.",
                                       title_color=theme.COLOR_WARNING)
        self.sim_res_card.pack(fill="x", pady=(4, 0))

    def _init_sim_plot(self):
        self.ax_sim.clear()
        self.ax_sim.set_facecolor(theme.BG_CARD)
        self.ax_sim.set_title("Proyeccion Dinamica de PM₂.₅ a 24 Horas", fontsize=11, color=theme.TEXT_TITLE, weight="bold")
        self.ax_sim.set_xlabel("Horizonte Temporal (Horas)", fontsize=9.5, color=theme.TEXT_BODY)
        self.ax_sim.set_ylabel("Concentracion de PM₂.₅ (µg/m³)", fontsize=9.5, color=theme.TEXT_BODY)
        self.canvas_sim.draw_idle()

    def _execute_inference(self):
        if self.model_ref is None or self.model_prop is None:
            messagebox.showwarning("Modelos no entrenados", "Inicie primero el entrenamiento en 'Curvas de Aprendizaje'.")
            return
        try:
            vec_raw = [float(self.sim_inputs[c].get()) for c in [
                "TEMP", "PRES", "DEWP", "Iws", "hour", "day", "pm25_t", "pm25_t-1", "pm25_t-2", "pm25_t-3"
            ]]
        except ValueError:
            messagebox.showerror("Dato Invalido", "Introduzca numeros validos en todas las casillas.")
            return

        X_norm = self.data["scaler_X"].transform(np.array([vec_raw], dtype=np.float64))
        p_ref = max(0.0, min(1000.0, float(self.data["scaler_y"].inverse_transform(self.model_ref.predict(X_norm))[0, 0])))
        p_prop = max(0.0, min(1000.0, float(self.data["scaler_y"].inverse_transform(self.model_prop.predict(X_norm))[0, 0])))

        cat_aqi, rec = self._get_aqi_badge(p_prop)
        w_diag = ("Viento fuerte (> 15 m/s): Alta ventilacion y dispersion." if vec_raw[3] >= 15.0 else
                  "Viento moderado (6-15 m/s): Renovacion atmosferica continua." if vec_raw[3] >= 6.0 else
                  "Viento calmo (< 6 m/s): Estancamiento favorable a la acumulacion.")

        self.sim_res_card.update_text(
            f"  • Modelo Propio (10:16:8:1): {p_prop:.2f} µg/m³\n"
            f"  • Modelo Referencia (10:10:1): {p_ref:.2f} µg/m³\n"
            f"  • Categoria Sanitaria OMS:     {cat_aqi}\n"
            f"  • Recomendacion Sanitaria:      {rec}\n"
            f"  • Diagnostico Meteorologico:    {w_diag}"
        )
        views.draw_simulation_projection(self.ax_sim, self.canvas_sim, vec_raw, p_ref, p_prop, cat_aqi)

    def _get_aqi_badge(self, val):
        if val <= 50.0:
            return "Buena (0-50 µg/m³)", "Calidad satisfactoria. Riesgo nulo o minimo."
        elif val <= 100.0:
            return "Moderada (51-100 µg/m³)", "Aceptable; personas sensibles deben limitar esfuerzo prolongado."
        elif val <= 150.0:
            return "Danina Grupos Sensibles (101-150 µg/m³)", "Ninos, ancianos y afecciones respiratorias deben evitar exposicion."
        return "Danina / Peligrosa (> 150 µg/m³)", "Alerta sanitaria general. Evitar actividad en exteriores."

    def _load_dataset_async(self):
        threading.Thread(target=self._load_dataset_worker, daemon=True).start()

    def _load_dataset_worker(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        csv_path = os.path.join(project_root, "data", "datasets", "dataset_pm25.csv")
        self.data = load_and_preprocess_dataset(csv_path)
        self.msg_queue.put(("dataset_ready",))

    def _on_dataset_ready(self):
        n_tr, n_te = len(self.data["X_train_norm"]), len(self.data["X_test_norm"])
        self.card_dataset.update_text(f"Registros: {n_tr + n_te:,}\nTrain: {n_tr:,} (80%)\nTest: {n_te:,} (20%)\nEscala: [-1, +1] sin leakage")

    def _reset_entire_interface(self):
        """Restaura completamente los modelos, graficas, hiperparametros y simuladores."""
        self.model_ref = self.model_prop = self.eval_cache_ref = self.eval_cache_prop = None
        self.is_training = False

        self.graph_model_var.set("Propio (10:16:8:1)")
        self.graph_view_mode.set("Grafo Sinaptico Nitido")
        self.focus_node_var.set("Todas las Relaciones")
        self.f_focus_container.pack(side="left", padx=(12, 0))
        self._render_topology_view()

        for entry, val in [(self.entry_epochs, "500"), (self.entry_lr, "0.01"), (self.entry_mom, "0.0")]:
            entry.delete(0, "end")
            entry.insert(0, val)

        self.prog_bar.set(0.0)
        self.btn_start_train.configure(state="normal")
        self.lbl_train_status.configure(text="Listo para entrenar 500 epocas", text_color=theme.COLOR_SUCCESS)
        self._init_train_plot()

        self.card_ref.update_text("Parametros: 121\nTest EMC: 8.529,83\nTest REMC: 92,36 µg/m³\nMAE: 68,14 µg/m³\nMaxAE: 532,32 µg/m³\nMinAE: 0,02 µg/m³")
        self.card_prop.update_text("Parametros: 321\nTest EMC: 7.782,96\nTest REMC: 88,22 µg/m³\nMAE: 64,88 µg/m³\nMaxAE: 566,08 µg/m³\nMinAE: 0,01 µg/m³")
        self.card_gain.update_text("Reduccion EMC: -8,75%\nMejora REMC: -4,14 µg/m³\nMejora MAE: -3,26 µg/m³\nSobreajuste: Descartado")
        self._init_eval_plot()

        defaults = {"TEMP": "-5.0", "PRES": "1022.0", "DEWP": "-7.0", "Iws": "5.36", "hour": "3", "day": "2",
                    "pm25_t": "181.0", "pm25_t-1": "159.0", "pm25_t-2": "148.0", "pm25_t-3": "129.0"}
        for k, v in defaults.items():
            if k in self.sim_inputs:
                self.sim_inputs[k].delete(0, "end")
                self.sim_inputs[k].insert(0, v)

        self.sim_res_card.update_text("Configure las entradas y presione 'Simular Pronostico Neuronal'.")
        self._init_sim_plot()
        messagebox.showinfo("Sistema Restablecido", "Aplicacion restaurada con exito a su estado inicial.")


def launch_gui():
    """Lanza la aplicacion de escritorio."""
    app = ModernMlpGUI()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
