"""
Tests para el módulo charts.py
Verifica que CoolingVisualizer genera figuras Plotly válidas.
"""

import sys
import os
import pytest
import plotly.graph_objects as go

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
        assert isinstance(fig, go.Figure)

    def test_plot_k_comparison_returns_figure(self, visualizer):
        fig = visualizer.plot_k_comparison(t_max=30.0)
        assert isinstance(fig, go.Figure)

    def test_plot_semilog_returns_figure(self, visualizer):
        fig = visualizer.plot_semilog(t_max=30.0)
        assert isinstance(fig, go.Figure)

    def test_cooling_curve_has_traces_and_labels(self, visualizer):
        fig = visualizer.plot_cooling_curve(t_max=30.0)
        assert len(fig.data) >= 2  # main curve + marker points
        assert fig.layout.xaxis.title.text == 'Tiempo (minutos)'
        assert fig.layout.yaxis.title.text == 'Temperatura (°C)'

    def test_k_comparison_has_three_curves(self, visualizer):
        fig = visualizer.plot_k_comparison(t_max=30.0)
        assert len(fig.data) >= 3

    def test_semilog_has_correct_ylabel(self, visualizer):
        fig = visualizer.plot_semilog(t_max=30.0)
        assert fig.layout.yaxis.title.text == 'ln(T − Tₐ)'

    def test_colors_class_attribute(self):
        assert 'primary' in CoolingVisualizer.COLORS
        assert 'danger' in CoolingVisualizer.COLORS
        assert CoolingVisualizer.BG_COLOR == '#fafafa'

    def test_visualizer_stores_results(self, visualizer):
        assert visualizer.results.k > 0
        assert visualizer.params.T0 == 95.0
