"""
Tests para el módulo charts.py
Verifica que CoolingVisualizer genera figuras matplotlib válidas.
"""

import sys
import os
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import CoolingParameters
from solver import NewtonCoolingSolver
from charts import CoolingVisualizer


@pytest.fixture
def visualizer():
    params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
    solver = NewtonCoolingSolver(params)
    return CoolingVisualizer(solver)


class TestCoolingVisualizer:

    def test_plot_cooling_curve_returns_figure(self, visualizer):
        fig = visualizer.plot_cooling_curve(t_max=30.0)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_k_comparison_returns_figure(self, visualizer):
        fig = visualizer.plot_k_comparison(t_max=30.0)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_semilog_returns_figure(self, visualizer):
        fig = visualizer.plot_semilog(t_max=30.0)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_cooling_curve_has_axes(self, visualizer):
        fig = visualizer.plot_cooling_curve(t_max=30.0)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Tiempo (minutos)'
        assert ax.get_ylabel() == 'Temperatura (°C)'
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_k_comparison_has_three_lines(self, visualizer):
        fig = visualizer.plot_k_comparison(t_max=30.0)
        ax = fig.axes[0]
        lines = [l for l in ax.get_lines() if len(l.get_xdata()) > 1]
        assert len(lines) >= 3
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_semilog_has_linear_trend(self, visualizer):
        fig = visualizer.plot_semilog(t_max=30.0)
        ax = fig.axes[0]
        assert ax.get_ylabel() == 'ln(T − Tₐ)'
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_colors_class_attribute(self):
        assert 'primary' in CoolingVisualizer.COLORS
        assert 'danger' in CoolingVisualizer.COLORS
        assert CoolingVisualizer.BG_COLOR == '#fafafa'

    def test_visualizer_stores_results(self, visualizer):
        assert visualizer.results.k > 0
        assert visualizer.params.T0 == 95.0
