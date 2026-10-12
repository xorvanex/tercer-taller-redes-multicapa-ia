"""
Configuracion tematica, paleta cromatica y utilidades de estilo para la GUI.
"""

from matplotlib.figure import Figure

BG_MAIN = "#13161C"
BG_CARD = "#1E222B"
BG_CARD_ALT = "#181B22"
BG_PLOT = "#181A20"

COLOR_PRIMARY = "#38BDF8"    # Azul accion / Modelo Referencia
COLOR_SUCCESS = "#34D399"    # Verde exito / Modelo Propio
COLOR_WARNING = "#FBBF24"    # Ambar / Ganancia / Atencion
COLOR_DANGER = "#F43F5E"     # Rojo alerta
COLOR_PURPLE = "#A78BFA"     # Purpura topologia

BORDER_COLOR = "#2D333F"
BORDER_FOCUS = "#475569"
TEXT_TITLE = "#FFFFFF"
TEXT_BODY = "#CBD5E1"
TEXT_MUTED = "#8B949E"
FONT_FAMILY = "Segoe UI"


def create_dark_figure(figsize=(9.5, 4.5), dpi=100):
    """Crea figura Matplotlib con fondo oscuro y ejes estilizados."""
    fig = Figure(figsize=figsize, dpi=dpi, facecolor=BG_PLOT)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_CARD)
    for s in ax.spines.values():
        s.set_color(BORDER_FOCUS)
    ax.tick_params(colors=TEXT_BODY, labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.4, color="#616E85")
    return fig, ax


def format_number(val, decimals=2):
    """Formatea numeros flotantes con formato decimal y miles latinoamericano."""
    if val is None:
        return "N/A"
    try:
        f_val = float(val)
        fmt = f"{{:,.{decimals}f}}"
        return fmt.format(f_val).replace(",", "@").replace(".", ",").replace("@", ".")
    except (ValueError, TypeError):
        return str(val)
