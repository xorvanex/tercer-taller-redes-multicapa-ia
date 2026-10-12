"""
Componentes de interfaz grafica reutilizables para la aplicacion en CustomTkinter.
"""

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import gui_theme as theme


class MetricCard(ctk.CTkFrame):
    """Tarjeta estandarizada para presentar indicadores, metricas o estados."""

    def __init__(self, parent, title, initial_text="", title_color=theme.COLOR_PRIMARY,
                 text_color=theme.TEXT_BODY, title_size=11, body_size=10, **kwargs):
        super().__init__(parent, fg_color=theme.BG_CARD, corner_radius=8,
                         border_width=1, border_color=theme.BORDER_COLOR, **kwargs)

        self.lbl_title = ctk.CTkLabel(
            self, text=title, font=ctk.CTkFont(family=theme.FONT_FAMILY, size=title_size, weight="bold"),
            text_color=title_color
        )
        self.lbl_title.pack(anchor="w", padx=12, pady=(8, 2))

        self.lbl_content = ctk.CTkLabel(
            self, text=initial_text, font=ctk.CTkFont(family=theme.FONT_FAMILY, size=body_size),
            justify="left", text_color=text_color
        )
        self.lbl_content.pack(anchor="w", padx=12, pady=(0, 8))

    def update_text(self, text, text_color=None):
        """Actualiza el texto descriptivo o numerico de la tarjeta."""
        self.lbl_content.configure(text=text)
        if text_color:
            self.lbl_content.configure(text_color=text_color)


class SectionBanner(ctk.CTkFrame):
    """Banner horizontal para exponer guias operativas o analisis cientifico."""

    def __init__(self, parent, title, body, title_color=theme.TEXT_TITLE, **kwargs):
        super().__init__(parent, fg_color=theme.BG_CARD_ALT, corner_radius=8,
                         border_width=1, border_color=theme.BORDER_COLOR, **kwargs)

        ctk.CTkLabel(
            self, text=title, font=ctk.CTkFont(family=theme.FONT_FAMILY, size=11, weight="bold"),
            text_color=title_color
        ).pack(anchor="w", padx=12, pady=(6, 2))

        self.lbl_body = ctk.CTkLabel(
            self, text=body, font=ctk.CTkFont(family=theme.FONT_FAMILY, size=10),
            justify="left", text_color=theme.TEXT_BODY
        )
        self.lbl_body.pack(anchor="w", padx=12, pady=(0, 6))

    def update_body(self, new_body):
        self.lbl_body.configure(text=new_body)


def embed_plot(parent, fig, with_toolbar=True):
    """Incrusta una figura Matplotlib en el contenedor Tkinter con barra opcional."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
    toolbar = None
    if with_toolbar:
        tb_frame = ctk.CTkFrame(parent, fg_color="transparent")
        tb_frame.pack(fill="x", padx=6, pady=(0, 4))
        toolbar = NavigationToolbar2Tk(canvas, tb_frame)
        toolbar.update()
    return canvas, toolbar
