"""
Visualización — Ley de Enfriamiento de Newton
Genera gráficos matplotlib para la aplicación.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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

    def _setup_axes(self, ax: plt.Axes) -> None:
        """Configuración visual compartida para todos los gráficos."""
        ax.set_facecolor(self.BG_COLOR)
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plot_cooling_curve(self, t_max: float) -> Figure:
        """Gráfico principal: curva de enfriamiento con puntos anotados."""
        p = self.params
        r = self.results
        k = r.k

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(self.BG_COLOR)
        self._setup_axes(ax)

        t_arr, T_arr = self.solver.generate_curve(t_max)

        ax.plot(t_arr, T_arr, color=self.COLORS['primary'], linewidth=2.5,
                label=r'$T(t) = T_a + (T_0 - T_a) \cdot e^{-kt}$', zorder=3)
        ax.fill_between(t_arr, p.Ta, T_arr, color=self.COLORS['primary'],
                        alpha=0.06, zorder=1)

        ax.axhline(y=p.Ta, color=self.COLORS['muted'], linestyle='--',
                   linewidth=1, alpha=0.8, label=f'$T_a$ = {p.Ta:.0f} °C')

        points = [
            (0, p.T0, f'$T_0$ = {p.T0:.0f} °C',
             self.COLORS['danger'], -10, 12),
            (p.t1, p.Tm, f'T({p.t1:.1f}) = {p.Tm:.0f} °C',
             self.COLORS['warning'], 8, -18),
            (p.t2, r.T2, f'T({p.t2:.0f}) = {r.T2:.1f} °C',
             self.COLORS['success'], 8, 12),
            (r.t_goal, p.Tgoal,
             f'T* = {p.Tgoal:.0f} °C @ {r.t_goal:.1f} min',
             self.COLORS['purple'], 8, -18),
        ]

        for px, py, label, color, dx, dy in points:
            ax.plot(px, py, 'o', color=color, markersize=9, zorder=5,
                    markeredgecolor='white', markeredgewidth=2)
            ax.annotate(label, (px, py), textcoords="offset points",
                        xytext=(dx, dy), fontsize=10, fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=color, alpha=0.9),
                        zorder=6)
            ax.plot([px, px], [p.Ta - 2, py], '--', color=color,
                    alpha=0.35, linewidth=1)
            ax.plot([0, px], [py, py], '--', color=color,
                    alpha=0.35, linewidth=1)

        ax.set_xlabel('Tiempo (minutos)', fontsize=13, fontweight='500', labelpad=10)
        ax.set_ylabel('Temperatura (°C)', fontsize=13, fontweight='500', labelpad=10)
        ax.set_title(f'Curva de enfriamiento — GPU (k = {k:.4f} min⁻¹)',
                     fontsize=15, fontweight='600', pad=15)
        ax.set_xlim(0, t_max)
        ax.set_ylim(max(0, p.Ta - 5), p.T0 + 8)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='#dee2e6')
        ax.grid(True, alpha=0.2, linestyle='-')

        plt.tight_layout()
        return fig

    def plot_k_comparison(self, t_max: float) -> Figure:
        """Gráfico comparativo: efecto de diferentes valores de k."""
        p = self.params
        k = self.results.k

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(self.BG_COLOR)
        self._setup_axes(ax)

        k_configs = [
            (k * 0.5, 'Ventilación pobre (k/2)', self.COLORS['danger'], '--'),
            (k, f'Caso actual (k={k:.4f})', self.COLORS['primary'], '-'),
            (k * 2.0, 'Refrigeración líquida (2k)', self.COLORS['success'], '-.'),
        ]

        t_comp = np.linspace(0, t_max, 500)
        for kv, lbl, clr, sty in k_configs:
            T_comp = p.Ta + p.initial_difference * np.exp(-kv * t_comp)
            ax.plot(t_comp, T_comp, color=clr, linewidth=2, linestyle=sty, label=lbl)

        ax.axhline(y=p.Ta, color=self.COLORS['muted'], linestyle=':',
                   linewidth=1, alpha=0.6)
        ax.set_xlabel('Tiempo (minutos)', fontsize=12)
        ax.set_ylabel('Temperatura (°C)', fontsize=12)
        ax.set_title('Comparación: impacto del tipo de refrigeración',
                     fontsize=14, fontweight='600')
        ax.legend(fontsize=10, loc='upper right')
        ax.set_xlim(0, t_max)
        ax.set_ylim(max(0, p.Ta - 5), p.T0 + 5)

        plt.tight_layout()
        return fig

    def plot_semilog(self, t_max: float) -> Figure:
        """Gráfico semilogarítmico: linealización del modelo."""
        p = self.params
        k = self.results.k

        fig, ax = plt.subplots(figsize=(12, 4.5))
        fig.patch.set_facecolor(self.BG_COLOR)
        self._setup_axes(ax)

        t_log = np.linspace(0, t_max * 0.95, 500)
        T_log = p.Ta + p.initial_difference * np.exp(-k * t_log)
        ln_diff = np.log(T_log - p.Ta)

        ax.plot(t_log, ln_diff, color=self.COLORS['primary'], linewidth=2.5,
                label=f'ln(T − Tₐ) = −{k:.4f}·t + {np.log(p.initial_difference):.2f}')

        t_mid = t_max * 0.4
        y_mid = np.log(p.initial_difference) - k * t_mid
        ax.annotate(f'pendiente = −k = −{k:.4f}', xy=(t_mid, y_mid),
                    fontsize=11, color=self.COLORS['danger'], fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor=self.COLORS['danger'],
                              boxstyle='round,pad=0.4'))

        ax.set_xlabel('Tiempo (minutos)', fontsize=12)
        ax.set_ylabel('ln(T − Tₐ)', fontsize=12)
        ax.set_title('Linealización — verificación del modelo newtoniano',
                     fontsize=14, fontweight='600')
        ax.legend(fontsize=10)

        plt.tight_layout()
        return fig
