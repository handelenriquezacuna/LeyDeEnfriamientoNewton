"""
Tests unitarios para el módulo models.py
Validación de parámetros y propiedades derivadas.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import CoolingParameters, CoolingResults


class TestCoolingParametersValidation:
    """Pruebas para CoolingParameters.validate()."""

    def _params(self, **overrides) -> CoolingParameters:
        defaults = dict(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        defaults.update(overrides)
        return CoolingParameters(**defaults)

    def test_parametros_validos(self):
        valido, msg = self._params().validate()
        assert valido is True
        assert msg == ""

    def test_T0_menor_que_Ta(self):
        valido, msg = self._params(T0=20.0, Ta=28.0).validate()
        assert valido is False
        assert "T₀" in msg

    def test_T0_igual_a_Ta(self):
        valido, msg = self._params(T0=28.0, Ta=28.0).validate()
        assert valido is False
        assert "T₀" in msg

    def test_Tm_menor_o_igual_a_Ta(self):
        valido, msg = self._params(Tm=28.0).validate()
        assert valido is False
        assert "T(t₁)" in msg

    def test_Tm_menor_que_Ta(self):
        valido, msg = self._params(Tm=20.0).validate()
        assert valido is False
        assert "T(t₁)" in msg

    def test_Tm_mayor_o_igual_a_T0(self):
        valido, msg = self._params(Tm=95.0).validate()
        assert valido is False
        assert "T(t₁)" in msg

    def test_Tm_mayor_que_T0(self):
        valido, msg = self._params(Tm=100.0).validate()
        assert valido is False
        assert "T(t₁)" in msg

    def test_t1_cero(self):
        valido, msg = self._params(t1=0.0).validate()
        assert valido is False
        assert "t₁" in msg

    def test_t1_negativo(self):
        valido, msg = self._params(t1=-5.0).validate()
        assert valido is False
        assert "t₁" in msg

    def test_Tgoal_menor_o_igual_a_Ta(self):
        valido, msg = self._params(Tgoal=28.0).validate()
        assert valido is False
        assert "T*" in msg

    def test_Tgoal_menor_que_Ta(self):
        valido, msg = self._params(Tgoal=20.0).validate()
        assert valido is False
        assert "T*" in msg

    def test_Tgoal_mayor_o_igual_a_T0(self):
        valido, msg = self._params(Tgoal=95.0).validate()
        assert valido is False
        assert "T*" in msg

    def test_Tgoal_mayor_que_T0(self):
        valido, msg = self._params(Tgoal=100.0).validate()
        assert valido is False
        assert "T*" in msg


class TestCoolingParametersProperties:
    """Pruebas para las propiedades derivadas de CoolingParameters."""

    def test_initial_difference(self):
        p = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        assert p.initial_difference == pytest.approx(67.0)

    def test_measured_difference(self):
        p = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        assert p.measured_difference == pytest.approx(44.0)

    def test_goal_difference(self):
        p = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        assert p.goal_difference == pytest.approx(7.0)

    def test_measurement_ratio(self):
        p = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        assert p.measurement_ratio == pytest.approx(44.0 / 67.0, rel=1e-10)


class TestCoolingResults:
    """Pruebas para la construcción de CoolingResults."""

    def test_construction(self):
        r = CoolingResults(k=0.1, T2=50.0, t_goal=20.0, half_life=6.93, tau=10.0)
        assert r.k == 0.1
        assert r.T2 == 50.0
        assert r.t_goal == 20.0
        assert r.half_life == 6.93
        assert r.tau == 10.0
