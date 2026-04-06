"""
Visualización — Ley de Enfriamiento de Newton
Genera gráficos Plotly interactivos para la aplicación.
"""

import numpy as np
import plotly.graph_objects as go

from solver import NewtonCoolingSolver


class CoolingVisualizer:
    """Genera los gráficos del problema de enfriamiento."""

    COLORS = {
        'primary': '#4361ee',
        'danger': '#e63946',
        'warning': '#f77f00',
        'success': '#2a9d8f',
        'purple': '#7209b7',
        'muted': '#adb5bd',
    }
    BG_COLOR = '#fafafa'

    def __init__(self, solver: NewtonCoolingSolver):
        self.solver = solver
        self.params = solver.params
        self.results = solver.solve()

    # ------------------------------------------------------------------
    # Shared layout helper
    # ------------------------------------------------------------------
    @staticmethod
    def _common_layout(fig: go.Figure, y_title: str = "Temperatura (°C)",
                       title: str = "") -> None:
        """Apply the shared layout settings to any figure."""
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=60, r=20, t=50, b=50),
            height=450,
            xaxis_title="Tiempo (minutos)",
            yaxis_title=y_title,
            title=dict(text=title, x=0.5, font=dict(size=15)),
            font=dict(family="Inter, sans-serif"),
        )

    # ------------------------------------------------------------------
    # 1. Curva de enfriamiento
    # ------------------------------------------------------------------
    def plot_cooling_curve(self, t_max: float) -> go.Figure:
        """Gráfico principal: curva de enfriamiento con puntos anotados."""
        p = self.params
        r = self.results
        k = r.k

        t_arr, T_arr = self.solver.generate_curve(t_max)

        fig = go.Figure()

        # Main curve with light-blue fill down to zero
        fig.add_trace(go.Scatter(
            x=t_arr, y=T_arr,
            mode='lines',
            name=r'T(t) = Tₐ + (T₀ − Tₐ)·e⁻ᵏᵗ',
            line=dict(color=self.COLORS['primary'], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(67, 97, 238, 0.06)',
            hovertemplate="t = %{x:.1f} min<br>T = %{y:.1f} °C<extra></extra>",
        ))

        # Horizontal dashed line for Ta (ambient)
        fig.add_hline(
            y=p.Ta,
            line_dash="dash",
            line_color=self.COLORS['muted'],
            line_width=1,
            opacity=0.8,
            annotation_text=f"Tₐ = {p.Ta:.0f} °C",
            annotation_position="top right",
            annotation_font_color=self.COLORS['muted'],
        )

        # Annotated points
        points = [
            (0, p.T0,
             f'T₀ = {p.T0:.0f} °C',
             self.COLORS['danger'], -10, 20),
            (p.t1, p.Tm,
             f'T({p.t1:.1f}) = {p.Tm:.0f} °C',
             self.COLORS['warning'], 10, -25),
            (p.t2, r.T2,
             f'T({p.t2:.0f}) = {r.T2:.1f} °C',
             self.COLORS['success'], 10, 20),
            (r.t_goal, p.Tgoal,
             f'T* = {p.Tgoal:.0f} °C @ {r.t_goal:.1f} min',
             self.COLORS['purple'], 10, -25),
        ]

        for px, py, label, color, dx, dy in points:
            # Marker
            fig.add_trace(go.Scatter(
                x=[px], y=[py],
                mode='markers',
                marker=dict(size=11, color=color,
                            line=dict(width=2, color='white')),
                showlegend=False,
                hovertemplate=f"{label}<extra></extra>",
            ))

            # Vertical dashed guide line from Ta to point
            fig.add_shape(
                type="line", x0=px, x1=px,
                y0=p.Ta - 2, y1=py,
                line=dict(color=color, width=1, dash="dash"),
                opacity=0.35,
            )
            # Horizontal dashed guide line from y-axis to point
            fig.add_shape(
                type="line", x0=0, x1=px,
                y0=py, y1=py,
                line=dict(color=color, width=1, dash="dash"),
                opacity=0.35,
            )

            # Annotation box
            fig.add_annotation(
                x=px, y=py,
                text=label,
                showarrow=True,
                ax=dx, ay=dy,
                font=dict(size=10, color=color, family="Inter, sans-serif"),
                bgcolor="white",
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
                opacity=0.92,
            )

        # Axis ranges
        fig.update_xaxes(range=[0, t_max])
        fig.update_yaxes(range=[max(0, p.Ta - 5), p.T0 + 8])

        self._common_layout(
            fig,
            title=f'Curva de enfriamiento — GPU (k = {k:.4f} min⁻¹)',
        )

        return fig

    # ------------------------------------------------------------------
    # 2. Comparación de valores de k
    # ------------------------------------------------------------------
    def plot_k_comparison(self, t_max: float) -> go.Figure:
        """Gráfico comparativo: efecto de diferentes valores de k."""
        p = self.params
        k = self.results.k

        t_comp = np.linspace(0, t_max, 500)

        k_configs = [
            (k * 0.5, 'Ventilación pobre (k/2)',
             self.COLORS['danger'], 'dash'),
            (k, f'Caso actual (k={k:.4f})',
             self.COLORS['primary'], 'solid'),
            (k * 2.0, 'Refrigeración líquida (2k)',
             self.COLORS['success'], 'dashdot'),
        ]

        fig = go.Figure()

        for kv, lbl, clr, sty in k_configs:
            T_comp = p.Ta + p.initial_difference * np.exp(-kv * t_comp)
            fig.add_trace(go.Scatter(
                x=t_comp, y=T_comp,
                mode='lines',
                name=lbl,
                line=dict(color=clr, width=2, dash=sty),
                hovertemplate="t = %{x:.1f} min<br>T = %{y:.1f} °C<extra></extra>",
            ))

        # Horizontal line for ambient temperature
        fig.add_hline(
            y=p.Ta,
            line_dash="dot",
            line_color=self.COLORS['muted'],
            line_width=1,
            opacity=0.6,
        )

        fig.update_xaxes(range=[0, t_max])
        fig.update_yaxes(range=[max(0, p.Ta - 5), p.T0 + 5])

        # Clickable legend to toggle curves
        fig.update_layout(legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ))

        self._common_layout(
            fig,
            title='Comparación: impacto del tipo de refrigeración',
        )

        return fig

    # ------------------------------------------------------------------
    # 3. Gráfico semilogarítmico (linealización)
    # ------------------------------------------------------------------
    def plot_semilog(self, t_max: float) -> go.Figure:
        """Gráfico semilogarítmico: linealización del modelo."""
        p = self.params
        k = self.results.k

        t_log = np.linspace(0, t_max * 0.95, 500)
        T_log = p.Ta + p.initial_difference * np.exp(-k * t_log)
        ln_diff = np.log(T_log - p.Ta)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=t_log, y=ln_diff,
            mode='lines',
            name=f'ln(T − Tₐ) = −{k:.4f}·t + {np.log(p.initial_difference):.2f}',
            line=dict(color=self.COLORS['primary'], width=2.5),
            hovertemplate="t = %{x:.1f} min<br>ln(T−Tₐ) = %{y:.2f}<extra></extra>",
        ))

        # Slope annotation at 40 % of the time range
        t_mid = t_max * 0.4
        y_mid = np.log(p.initial_difference) - k * t_mid

        fig.add_annotation(
            x=t_mid, y=y_mid,
            text=f'pendiente = −k = −{k:.4f}',
            showarrow=False,
            font=dict(size=11, color=self.COLORS['danger'],
                      family="Inter, sans-serif"),
            bgcolor="white",
            bordercolor=self.COLORS['danger'],
            borderwidth=1,
            borderpad=5,
        )

        self._common_layout(
            fig,
            y_title='ln(T − Tₐ)',
            title='Linealización — verificación del modelo newtoniano',
        )

        return fig
