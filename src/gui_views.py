"""
Visualizaciones cientificas modulares para la interfaz grafica.
Incluye grafo sinaptico vectorial (LineCollection), diagrama MATLAB, heatmap
y el histograma de residuos de validacion inspirado en scripts de MATLAB del docente.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import gui_theme as theme


def draw_synaptic_graph(ax, canvas, model_obj, is_prop=True, focus_filter="Todas las Relaciones"):
    """Dibuja el grafo sinaptico optimizado vectorialmente mediante LineCollection."""
    ax.clear()
    ax.set_facecolor(theme.BG_PLOT)

    layers = [10, 16, 8, 1] if is_prop else [10, 10, 1]
    layer_titles = (
        ["Entrada (x)\n(10 vars)", "Oculta 1 (h¹)\n(16 tansig)", "Oculta 2 (h²)\n(8 tansig)", "Salida (ŷ)\n(purelin)"]
        if is_prop else
        ["Entrada (x)\n(10 vars)", "Oculta 1 (h¹)\n(10 tansig)", "Salida (ŷ)\n(purelin)"]
    )
    title = (
        "Grafo de Conexiones Sinapticas - Modelo Propio (10:16:8:1)"
        if is_prop else
        "Grafo de Conexiones Sinapticas - Modelo Referencia (10:10:1)"
    )

    n_layers = len(layers)
    x_positions = np.linspace(0.12, 0.88, n_layers)
    node_coords = [
        [(x_positions[l], y) for y in (np.linspace(0.10, 0.90, c) if c > 1 else [0.5])]
        for l, c in enumerate(layers)
    ]

    input_names = [
        "x₁ (TEMP)", "x₂ (PRES)", "x₃ (DEWP)", "x₄ (Iws)",
        "x₅ (hour)", "x₆ (day)", "x₇ (PM₂.₅ ᵗ)", "x₈ (PM₂.₅ ᵗ⁻¹)",
        "x₉ (PM₂.₅ ᵗ⁻²)", "x₁₀ (PM₂.₅ ᵗ⁻³)"
    ]
    focus_idx = input_names.index(focus_filter) if focus_filter in input_names else None
    has_w = model_obj is not None

    segments, colors, linewidths = [], [], []
    for l in range(n_layers - 1):
        src, dst = node_coords[l], node_coords[l + 1]
        w_mat = model_obj.weights[l] if has_w else None

        for i, (x1, y1) in enumerate(src):
            is_foc = (focus_idx is None) or (l == 0 and i == focus_idx)
            for j, (x2, y2) in enumerate(dst):
                segments.append([(x1, y1), (x2, y2)])
                if w_mat is not None:
                    w_val = w_mat[i, j]
                    c_hex = theme.COLOR_PRIMARY if w_val >= 0 else theme.COLOR_DANGER
                    w_abs = abs(w_val)
                    alpha = min(0.95, max(0.25, w_abs / 1.5)) if is_foc else 0.04
                    lw = min(2.4, max(0.7, w_abs * 1.1)) if is_foc else 0.3
                else:
                    c_hex = "#60A5FA" if is_foc else "#334155"
                    alpha = 0.55 if is_foc else 0.06
                    lw = 0.9 if is_foc else 0.3

                colors.append(plt.matplotlib.colors.to_rgba(c_hex, alpha=alpha))
                linewidths.append(lw)

    ax.add_collection(LineCollection(segments, colors=colors, linewidths=linewidths, zorder=1))

    for l, nodes in enumerate(node_coords):
        xs, ys = [p[0] for p in nodes], [p[1] for p in nodes]
        c_fill = theme.COLOR_PRIMARY if l == 0 else (theme.COLOR_WARNING if l == n_layers - 1 else (theme.COLOR_SUCCESS if is_prop and l == 1 else theme.COLOR_PURPLE))
        ax.scatter(xs, ys, s=320, color=c_fill, edgecolors="#FFFFFF", linewidths=1.8, zorder=4)

        for n, (x, y) in enumerate(nodes):
            if l == 0 and n < len(input_names):
                t_col = theme.COLOR_WARNING if (focus_idx is not None and n == focus_idx) else "#E2E8F0"
                ax.text(x - 0.035, y, input_names[n], fontsize=8, color=t_col, ha="right", va="center", weight="bold")
            elif l == n_layers - 1:
                ax.text(x + 0.035, y, "ŷ = PM₂.₅(t+24)", fontsize=9, color=theme.COLOR_WARNING, ha="left", va="center", weight="bold")

        ax.text(x_positions[l], 0.97, layer_titles[l], fontsize=8.5, color="#F1F5F9", ha="center", va="center", weight="bold")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.03, 1.03)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=theme.TEXT_TITLE, weight="bold", pad=12)
    canvas.draw_idle()


def draw_matlab_diagram(ax, canvas, is_prop=True):
    """Dibuja la arquitectura en diagrama de bloques modular (view(net))."""
    ax.clear()
    ax.set_facecolor(theme.BG_PLOT)

    blocks = [
        {"title": "Entrada x(k)", "dim": "10 × 1", "desc": "10 Variables", "color": "#0284C7"},
        {"title": "Oculta 1 (W¹+b¹)", "dim": f"W¹: {16 if is_prop else 10}×10", "desc": f"{16 if is_prop else 10} tansig", "color": "#059669"},
    ]
    if is_prop:
        blocks.append({"title": "Oculta 2 (W²+b²)", "dim": "W²: 8×16", "desc": "8 tansig", "color": "#7C3AED"})
    blocks.extend([
        {"title": "Salida (W+b)", "dim": f"W: 1×{8 if is_prop else 10}", "desc": "1 purelin", "color": "#D97706"},
        {"title": "Prediccion ŷ", "dim": "1 × 1", "desc": "PM₂.₅ a 24h", "color": "#DC2626"}
    ])

    xs = np.linspace(0.10, 0.90, len(blocks))
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.075, 0.5), xytext=(xs[i] + 0.075, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#F1F5F9", mutation_scale=18))

    for i, b in enumerate(blocks):
        xc = xs[i]
        rect = plt.Rectangle((xc - 0.07, 0.28), 0.14, 0.44, facecolor=b["color"],
                             edgecolor="#FFFFFF", lw=2.0, alpha=0.92, zorder=3)
        ax.add_patch(rect)
        ax.text(xc, 0.62, b["title"], fontsize=9, color="#FFFFFF", weight="bold", ha="center", va="center")
        ax.text(xc, 0.48, b["dim"], fontsize=8, color="#FDE047", weight="bold", ha="center", va="center")
        ax.text(xc, 0.36, b["desc"], fontsize=7.5, color="#F1F5F9", ha="center", va="center")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.1, 0.9)
    ax.axis("off")
    title = f"Diagrama de Bloques Estilo MATLAB (view(net)) - {'Embudo 10:16:8:1' if is_prop else 'Referencia 10:10:1'}"
    ax.set_title(title, fontsize=11, color=theme.TEXT_TITLE, weight="bold", pad=12)
    canvas.draw_idle()


def draw_weight_heatmap(fig, ax, canvas, model_obj):
    """Muestra la matriz de pesos W¹ como mapa de calor cuantitativo."""
    ax.set_facecolor(theme.BG_PLOT)

    if model_obj is None:
        ax.text(0.5, 0.5, "Entrene los modelos para visualizar los pesos en el Mapa de Calor.\nPresione 'Entrenar Ambas Redes' en la pestana de Curvas.",
                fontsize=11, color=theme.COLOR_WARNING, ha="center", va="center", weight="bold")
        ax.axis("off")
        canvas.draw_idle()
        return ax

    w1 = model_obj.weights[0].T
    input_names = ["TEMP", "PRES", "DEWP", "Iws", "hour", "day", "pm25_t", "t-1", "t-2", "t-3"]
    im = ax.imshow(w1, cmap="coolwarm", aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.04)
    cbar.ax.tick_params(colors=theme.TEXT_BODY)
    cbar.set_label("Magnitud del Peso W (Azul: Excitatorio, Rojo: Inhibitorio)", color=theme.TEXT_BODY, fontsize=8.5, weight="bold")

    ax.set_xticks(range(len(input_names)))
    ax.set_xticklabels(input_names, fontsize=8, color="#E2E8F0", rotation=25, weight="bold")
    ax.set_yticks(range(w1.shape[0]))
    ax.set_yticklabels([f"h{i+1}" for i in range(w1.shape[0])], fontsize=8, color="#E2E8F0")

    ax.set_title(f"Mapa de Calor de la Matriz W¹ ({w1.shape[0]}×10)", fontsize=11, color=theme.TEXT_TITLE, weight="bold", pad=10)
    ax.set_xlabel("Variables de Entrada", fontsize=9, color=theme.TEXT_BODY, weight="bold")
    ax.set_ylabel("Neuronas Ocultas", fontsize=9, color=theme.TEXT_BODY, weight="bold")
    canvas.draw_idle()
    return ax


def draw_learning_curves(ax, canvas, model_ref, model_prop):
    """Dibuja las curvas comparativas de convergencia (Train vs Test)."""
    ax.clear()
    ax.set_facecolor(theme.BG_CARD)
    ep = model_ref.history["epoch"]
    ax.plot(ep, model_ref.history["train_mse"], label="Referencia - Train", color=theme.COLOR_PRIMARY, lw=1.8)
    ax.plot(ep, model_ref.history["test_mse"], label="Referencia - Test", color=theme.COLOR_PRIMARY, lw=1.6, linestyle="--")
    ax.plot(ep, model_prop.history["train_mse"], label="Propio - Train", color=theme.COLOR_SUCCESS, lw=2.0)
    ax.plot(ep, model_prop.history["test_mse"], label="Propio - Test", color=theme.COLOR_SUCCESS, lw=1.6, linestyle="--")

    ax.set_title("Curvas de Aprendizaje Comparativas (EMC vs. Epocas)", fontsize=11, color=theme.TEXT_TITLE, weight="bold")
    ax.set_xlabel("Epoca de Entrenamiento (k)", fontsize=10, color=theme.TEXT_BODY)
    ax.set_ylabel("Error Cuadratico Medio (MSE normalizado)", fontsize=10, color=theme.TEXT_BODY)
    ax.tick_params(colors=theme.TEXT_BODY)
    for s in ax.spines.values():
        s.set_color(theme.BORDER_FOCUS)
    ax.grid(True, linestyle=":", alpha=0.4, color="#616E85")
    ax.legend(loc="upper right", facecolor=theme.BG_PLOT, edgecolor=theme.BORDER_FOCUS, labelcolor=theme.TEXT_BODY, fontsize=9)
    canvas.draw_idle()


def draw_evaluation_view(ax, canvas, ev_ref, ev_prop, mode="series"):
    """Dibuja graficos de evaluacion (Serie Temporal, Dispersion, o Histograma de Residuos)."""
    ax.clear()
    ax.set_facecolor(theme.BG_CARD)

    if mode == "series":
        w = 200
        hours = np.arange(1, w + 1)
        ax.plot(hours, ev_ref["y_real"][:w].flatten(), label="PM₂.₅ Real Observado", color="#FFFFFF", lw=2.1, alpha=0.95)
        ax.plot(hours, ev_ref["y_pred_real"][:w].flatten(), label="Prediccion Ref (10:10:1)", color=theme.COLOR_PRIMARY, lw=1.6, linestyle="--")
        ax.plot(hours, ev_prop["y_pred_real"][:w].flatten(), label="Prediccion Propia (10:16:8:1)", color=theme.COLOR_SUCCESS, lw=1.8, linestyle="-.")
        ax.set_title(f"Concentracion Real frente a Predicha (Ventana de {w} Horas en Test)", fontsize=11, color=theme.TEXT_TITLE, weight="bold")
        ax.set_xlabel("Horas de Prueba Consecutivas", fontsize=10, color=theme.TEXT_BODY)
        ax.set_ylabel("Concentracion de PM₂.₅ (µg/m³)", fontsize=10, color=theme.TEXT_BODY)

    elif mode == "dispersion":
        yr, yp = ev_prop["y_real"].flatten(), ev_prop["y_pred_real"].flatten()
        ax.scatter(yr, yp, alpha=0.35, s=14, color=theme.COLOR_SUCCESS, label="Predicciones (8.346 muestras)")
        max_v = max(float(np.max(yr)), float(np.max(yp)))
        ax.plot([0, max_v], [0, max_v], color="#EF4444", lw=1.8, linestyle="--", label="Ajuste Ideal (y = x)")
        ax.set_title("Diagrama de Dispersion: Real vs. Predicho (Modelo Propio)", fontsize=11, color=theme.TEXT_TITLE, weight="bold")
        ax.set_xlabel("PM₂.₅ Real Observado (µg/m³)", fontsize=10, color=theme.TEXT_BODY)
        ax.set_ylabel("PM₂.₅ Predicho a 24h (µg/m³)", fontsize=10, color=theme.TEXT_BODY)

    elif mode == "hist":
        err_ref = np.abs(ev_ref["y_real"].flatten() - ev_ref["y_pred_real"].flatten())
        err_prop = np.abs(ev_prop["y_real"].flatten() - ev_prop["y_pred_real"].flatten())
        bins = np.linspace(0, 300, 35)
        ax.hist(err_ref, bins=bins, alpha=0.55, color=theme.COLOR_PRIMARY, label=f"Ref (MAE: {ev_ref['mae_real']:.1f} µg/m³)")
        ax.hist(err_prop, bins=bins, alpha=0.65, color=theme.COLOR_SUCCESS, label=f"Propio (MAE: {ev_prop['mae_real']:.1f} µg/m³)")
        ax.axvline(ev_prop["mae_real"], color=theme.COLOR_WARNING, lw=1.8, linestyle="--", label="Media Error Propio")
        ax.set_title("Histograma de Residuos de Validacion (Inspirado en scripts MATLAB del Docente)", fontsize=11, color=theme.TEXT_TITLE, weight="bold")
        ax.set_xlabel("Error Absoluto |y_real - y_pred| (µg/m³)", fontsize=10, color=theme.TEXT_BODY)
        ax.set_ylabel("Frecuencia (Cantidad de Muestras)", fontsize=10, color=theme.TEXT_BODY)

    ax.tick_params(colors=theme.TEXT_BODY)
    for s in ax.spines.values():
        s.set_color(theme.BORDER_FOCUS)
    ax.grid(True, linestyle=":", alpha=0.4, color="#616E85")
    ax.legend(loc="upper right", facecolor=theme.BG_PLOT, edgecolor=theme.BORDER_FOCUS, labelcolor=theme.TEXT_BODY, fontsize=8.5)
    canvas.draw_idle()


def draw_simulation_projection(ax, canvas, vec_raw, pred_ref, pred_prop, cat_aqi):
    """Dibuja la proyeccion interactiva a 24 horas con bandas sanitarias de fondo."""
    ax.clear()
    ax.set_facecolor(theme.BG_CARD)

    t_hist = np.array([-3, -2, -1, 0])
    y_hist = np.array([vec_raw[9], vec_raw[8], vec_raw[7], vec_raw[6]])
    ax.plot(t_hist, y_hist, marker="o", color="#FFFFFF", lw=2.2, label="Historial (t-3 a t)", zorder=4)

    t_proj = np.array([0, 24])
    ax.plot(t_proj, [y_hist[-1], pred_ref], linestyle="--", marker="s", color=theme.COLOR_PRIMARY, lw=1.8, label=f"Ref: {pred_ref:.1f} µg/m³", zorder=3)
    ax.plot(t_proj, [y_hist[-1], pred_prop], linestyle="-.", marker="^", color=theme.COLOR_SUCCESS, lw=2.2, label=f"Propio: {pred_prop:.1f} µg/m³", zorder=3)

    ax.axhspan(0, 50, color="#10B981", alpha=0.12, label="Buena (0-50)")
    ax.axhspan(50, 100, color="#FBBF24", alpha=0.12, label="Moderada (50-100)")
    ax.axhspan(100, 150, color="#FB923C", alpha=0.12, label="Grupos Sensibles (100-150)")
    ax.axhspan(150, 500, color="#EF4444", alpha=0.12, label="Danina (>150)")

    ax.set_xticks([-3, -2, -1, 0, 6, 12, 18, 24])
    ax.set_xticklabels(["t-3", "t-2", "t-1", "t", "t+6", "t+12", "t+18", "t+24"], fontsize=8.5, color="#E2E8F0")
    ax.tick_params(colors=theme.TEXT_BODY)
    ax.set_ylabel("Concentracion PM₂.₅ (µg/m³)", fontsize=9.5, color=theme.TEXT_BODY)
    ax.set_title(f"Pronostico a 24h: {pred_prop:.2f} µg/m³ ({cat_aqi})", fontsize=10.5, color=theme.TEXT_TITLE, weight="bold")
    for s in ax.spines.values():
        s.set_color(theme.BORDER_FOCUS)
    ax.grid(True, linestyle=":", alpha=0.4, color="#616E85")
    ax.legend(loc="upper left", facecolor=theme.BG_PLOT, edgecolor=theme.BORDER_FOCUS, labelcolor=theme.TEXT_BODY, fontsize=8)
    canvas.draw_idle()
