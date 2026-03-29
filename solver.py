"""Solver OOP para la Ley de Enfriamiento de Newton.

Resuelve la EDO dT/dt = -k(T - Ta) mediante separacion de variables,
reemplazando las funciones procedurales de calculations.py.
"""

import numpy as np

from models import CoolingParameters, CoolingResults


class NewtonCoolingSolver:
    """Resuelve la EDO dT/dt = -k(T - Ta) mediante separación de variables.

    Solución analítica: T(t) = Ta + (T0 - Ta) * e^(-kt)
    """

    def __init__(self, params: CoolingParameters):
        self.params = params
        self._k: float | None = None

    @property
    def k(self) -> float:
        """Constante de enfriamiento k (min⁻¹), calculada bajo demanda y cacheada."""
        if self._k is None:
            p = self.params
            self._k = -np.log((p.Tm - p.Ta) / (p.T0 - p.Ta)) / p.t1
        return self._k

    def temperature_at(self, t: float) -> float:
        """Evalúa T(t) = Ta + (T0 - Ta) * e^(-kt)."""
        p = self.params
        return p.Ta + p.initial_difference * np.exp(-self.k * t)

    def time_for_temperature(self, T_target: float) -> float:
        """Despeja t de la ecuación para una temperatura objetivo."""
        p = self.params
        return -np.log((T_target - p.Ta) / p.initial_difference) / self.k

    @property
    def half_life(self) -> float:
        """Vida media térmica: tiempo para que (T-Ta) se reduzca a la mitad."""
        return np.log(2) / self.k

    @property
    def time_constant(self) -> float:
        """Constante de tiempo τ = 1/k."""
        return 1.0 / self.k

    def solve(self) -> CoolingResults:
        """Resuelve el problema completo y retorna todos los resultados."""
        p = self.params
        return CoolingResults(
            k=self.k,
            T2=self.temperature_at(p.t2),
            t_goal=self.time_for_temperature(p.Tgoal),
            half_life=self.half_life,
            tau=self.time_constant,
        )

    def generate_curve(self, t_max: float, n_points: int = 500):
        """Genera arrays (t, T) para graficar la curva de enfriamiento."""
        t_arr = np.linspace(0, t_max, n_points)
        T_arr = self.params.Ta + self.params.initial_difference * np.exp(-self.k * t_arr)
        return t_arr, T_arr

    def generate_table(self, t_max: float, n_rows: int = 20) -> list[dict]:
        """Genera datos tabulares para el dataframe."""
        p = self.params
        step = max(1, int(t_max / n_rows))
        time_points = list(range(0, int(t_max) + 1, step))

        for tp in [p.t1, p.t2]:
            if tp not in time_points:
                time_points.append(tp)
        time_points.sort()

        tabla = []
        for t in time_points:
            T_val = self.temperature_at(t)
            pct = ((p.T0 - T_val) / p.initial_difference) * 100
            tabla.append({
                "t (min)": round(t, 1),
                "T(t) °C": round(T_val, 2),
                "T − Tₐ (°C)": round(T_val - p.Ta, 2),
                "% enfriado": round(pct, 1),
            })
        return tabla

    @staticmethod
    def classify_k(k: float) -> tuple[str, str]:
        """Clasifica el valor de k y retorna (clasificación, descripción_sistema)."""
        if k > 0.1:
            return "alta", "refrigeración activa o líquida"
        elif k > 0.03:
            return "moderada", "disipadores estándar de aluminio con ventilación"
        else:
            return "baja", "ventilación pasiva o ambiente confinado"
