"""
Tests unitarios para el modulo solver.py (API OOP)
Ley de Enfriamiento de Newton - Disipacion Termica

Cubre los mismos escenarios y edge cases que test_calculations.py,
pero usando la interfaz orientada a objetos: NewtonCoolingSolver + CoolingParameters.

Secciones:
  1. Calculo de la constante k
  2. Evaluacion de temperatura en un instante dado
  3. Inversion: tiempo para alcanzar una temperatura objetivo
  4. Vida media y constante de tiempo
  5. Resolucion completa (solve) para cada escenario predefinido
  6. Generacion de tabla de datos
  7. Generacion de curva para graficado
  8. Clasificacion del valor de k
  9. Casos extremos (edge cases)
"""

import sys
import os
import math

import numpy as np
import pytest

# Asegurar que el directorio padre este en sys.path para importar los modulos
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from models import CoolingParameters
from solver import NewtonCoolingSolver
from scenarios import ESCENARIOS


# ============================================================================
# 1. Pruebas de calculo de k
# ============================================================================

class TestCalculateK:
    """Pruebas para el calculo de la constante de enfriamiento k via solver.k."""

    def test_gpu_gaming_k_exact_formula(self):
        """
        Escenario GPU gaming: T0=95, Ta=28, Tm=72, t1=4.
        k = -ln((72-28)/(95-28)) / 4 = -ln(44/67) / 4.
        """
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        esperado = -math.log(44.0 / 67.0) / 4.0
        assert solver.k == pytest.approx(esperado, rel=1e-10)

    def test_gpu_gaming_k_numeric_value(self):
        """
        Verificacion numerica directa del valor de k para GPU gaming.
        k debe ser aproximadamente 0.10513.
        """
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        assert solver.k == pytest.approx(0.10513, rel=1e-3)

    def test_k_always_positive(self):
        """k siempre debe ser positivo cuando Tm < T0 (enfriamiento real)."""
        params = CoolingParameters(T0=100.0, Ta=20.0, t1=5.0, Tm=60.0, t2=15.0, Tgoal=25.0)
        solver = NewtonCoolingSolver(params)
        assert solver.k > 0

    def test_k_positive_for_all_scenarios(self):
        """k debe ser positivo para todos los escenarios predefinidos."""
        for nombre, p in ESCENARIOS.items():
            solver = NewtonCoolingSolver(p)
            assert solver.k > 0, f"Escenario '{nombre}': k debe ser positivo"

    def test_k_larger_when_cooling_faster(self):
        """Si la temperatura baja mas rapido, k debe ser mayor."""
        params_slow = CoolingParameters(T0=100.0, Ta=20.0, t1=5.0, Tm=90.0, t2=15.0, Tgoal=25.0)
        params_fast = CoolingParameters(T0=100.0, Ta=20.0, t1=5.0, Tm=50.0, t2=15.0, Tgoal=25.0)
        solver_slow = NewtonCoolingSolver(params_slow)
        solver_fast = NewtonCoolingSolver(params_fast)
        assert solver_fast.k > solver_slow.k

    def test_k_is_cached(self):
        """k se calcula una sola vez y se cachea (acceso repetido da el mismo valor)."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        k1 = solver.k
        k2 = solver.k
        assert k1 == k2

    def test_k_gpu_servidor(self):
        """Verificar k para escenario GPU servidor."""
        p = ESCENARIOS["GPU servidor (data center)"]
        solver = NewtonCoolingSolver(p)
        esperado = -math.log((p.Tm - p.Ta) / (p.T0 - p.Ta)) / p.t1
        assert solver.k == pytest.approx(esperado, rel=1e-10)

    def test_k_cpu_laptop(self):
        """Verificar k para escenario CPU laptop."""
        p = ESCENARIOS["CPU laptop (uso normal)"]
        solver = NewtonCoolingSolver(p)
        esperado = -math.log((p.Tm - p.Ta) / (p.T0 - p.Ta)) / p.t1
        assert solver.k == pytest.approx(esperado, rel=1e-10)


# ============================================================================
# 2. Pruebas de temperature_at
# ============================================================================

class TestTemperatureAt:
    """Pruebas para la evaluacion T(t) = Ta + (T0 - Ta) * e^(-kt)."""

    def setup_method(self):
        """Configuracion comun: escenario GPU gaming."""
        self.params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        self.solver = NewtonCoolingSolver(self.params)

    def test_T_at_time_zero_is_T0(self):
        """En t=0 la temperatura debe ser exactamente T0."""
        T = self.solver.temperature_at(0.0)
        assert T == pytest.approx(self.params.T0, abs=1e-12)

    def test_T_at_infinity_approaches_Ta(self):
        """Cuando t -> infinito, T(t) debe tender a Ta."""
        T = self.solver.temperature_at(1e6)
        assert T == pytest.approx(self.params.Ta, abs=1e-6)

    def test_T_at_t1_equals_Tm(self):
        """En t=t1, la temperatura debe coincidir con Tm (dato de calibracion)."""
        T = self.solver.temperature_at(self.params.t1)
        assert T == pytest.approx(self.params.Tm, rel=1e-10)

    def test_T_decreases_with_time(self):
        """La temperatura debe ser funcion decreciente del tiempo."""
        T_5 = self.solver.temperature_at(5.0)
        T_10 = self.solver.temperature_at(10.0)
        T_20 = self.solver.temperature_at(20.0)
        assert T_5 > T_10 > T_20

    def test_T_always_greater_or_equal_Ta(self):
        """La temperatura nunca debe bajar por debajo de Ta."""
        for t in [0, 1, 5, 10, 50, 100, 1000]:
            T = self.solver.temperature_at(t)
            assert T >= self.params.Ta

    def test_T_at_t2_between_Ta_and_T0(self):
        """La temperatura en t2 debe estar entre Ta y T0."""
        T2 = self.solver.temperature_at(self.params.t2)
        assert self.params.Ta < T2 < self.params.T0

    def test_T_at_small_time_close_to_T0(self):
        """En un tiempo muy pequeno, T(t) debe estar cerca de T0."""
        T = self.solver.temperature_at(0.001)
        assert T == pytest.approx(self.params.T0, rel=1e-3)

    def test_T_at_various_times_monotonic(self):
        """La temperatura en una secuencia de tiempos debe ser monotonamente decreciente."""
        times = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        temps = [self.solver.temperature_at(t) for t in times]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]


# ============================================================================
# 3. Pruebas de time_for_temperature (inversion)
# ============================================================================

class TestTimeForTemperature:
    """Pruebas para la funcion inversa: dado T objetivo, encontrar t."""

    def setup_method(self):
        """Configuracion comun."""
        self.params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        self.solver = NewtonCoolingSolver(self.params)

    def test_round_trip_temperature_to_time(self):
        """time_for_temperature debe ser la inversa de temperature_at."""
        T_target = 50.0
        t_calc = self.solver.time_for_temperature(T_target)
        T_verif = self.solver.temperature_at(t_calc)
        assert T_verif == pytest.approx(T_target, rel=1e-10)

    def test_round_trip_multiple_temperatures(self):
        """Verificar la inversion para varias temperaturas objetivo."""
        for T_obj in [90.0, 80.0, 60.0, 40.0, 30.0]:
            t = self.solver.time_for_temperature(T_obj)
            T_check = self.solver.temperature_at(t)
            assert T_check == pytest.approx(T_obj, rel=1e-9), (
                f"Fallo para T_obj={T_obj}: T_check={T_check}"
            )

    def test_time_zero_for_T0(self):
        """Si la temperatura objetivo es T0, el tiempo debe ser 0."""
        t = self.solver.time_for_temperature(self.params.T0)
        assert t == pytest.approx(0.0, abs=1e-12)

    def test_positive_time_for_T_less_than_T0(self):
        """Para cualquier T < T0, el tiempo debe ser positivo."""
        t = self.solver.time_for_temperature(50.0)
        assert t > 0

    def test_time_grows_as_T_target_decreases(self):
        """Temperaturas mas bajas requieren mas tiempo."""
        t_high = self.solver.time_for_temperature(70.0)
        t_low = self.solver.time_for_temperature(40.0)
        assert t_low > t_high

    def test_time_for_Tm_equals_t1(self):
        """El tiempo para alcanzar Tm debe ser t1 (consistencia con calibracion)."""
        t = self.solver.time_for_temperature(self.params.Tm)
        assert t == pytest.approx(self.params.t1, rel=1e-10)

    def test_time_for_Tgoal_positive(self):
        """El tiempo para alcanzar Tgoal debe ser positivo."""
        t = self.solver.time_for_temperature(self.params.Tgoal)
        assert t > 0

    def test_time_for_near_Ta_is_large(self):
        """El tiempo para acercarse a Ta debe ser muy grande."""
        T_near_Ta = self.params.Ta + 0.01
        t = self.solver.time_for_temperature(T_near_Ta)
        assert t > 50  # debe ser un tiempo largo

    def test_round_trip_from_time_to_temperature(self):
        """Dando un tiempo, calcular T, y luego invertir para recuperar el tiempo."""
        t_original = 7.5
        T_at_t = self.solver.temperature_at(t_original)
        t_recovered = self.solver.time_for_temperature(T_at_t)
        assert t_recovered == pytest.approx(t_original, rel=1e-10)


# ============================================================================
# 4. Pruebas de half_life y time_constant
# ============================================================================

class TestHalfLifeAndTimeConstant:
    """Pruebas para half_life (t_1/2) y time_constant (tau)."""

    def test_half_life_formula(self):
        """half_life = ln(2) / k."""
        params = CoolingParameters(T0=100.0, Ta=20.0, t1=5.0, Tm=60.0, t2=15.0, Tgoal=25.0)
        solver = NewtonCoolingSolver(params)
        assert solver.half_life == pytest.approx(math.log(2) / solver.k, rel=1e-12)

    def test_half_life_formula_with_known_k(self):
        """Verificar half_life con un k conocido (k=0.1 -> hl=6.9315...)."""
        params = CoolingParameters(T0=100.0, Ta=20.0, t1=5.0, Tm=60.0, t2=15.0, Tgoal=25.0)
        solver = NewtonCoolingSolver(params)
        expected = math.log(2) / solver.k
        assert solver.half_life == pytest.approx(expected, rel=1e-12)

    def test_half_life_physical_meaning(self):
        """
        Tras una vida media, la diferencia (T - Ta) debe reducirse a la mitad.
        """
        params = CoolingParameters(T0=100.0, Ta=20.0, t1=3.0, Tm=70.0, t2=10.0, Tgoal=25.0)
        solver = NewtonCoolingSolver(params)
        hl = solver.half_life
        T_half = solver.temperature_at(hl)
        diferencia_inicial = params.T0 - params.Ta
        diferencia_en_hl = T_half - params.Ta
        assert diferencia_en_hl == pytest.approx(diferencia_inicial / 2.0, rel=1e-10)

    def test_time_constant_formula(self):
        """time_constant = 1 / k."""
        params = CoolingParameters(T0=80.0, Ta=25.0, t1=5.0, Tm=55.0, t2=15.0, Tgoal=30.0)
        solver = NewtonCoolingSolver(params)
        assert solver.time_constant == pytest.approx(1.0 / solver.k, rel=1e-12)

    def test_time_constant_physical_meaning(self):
        """
        Tras un tau, la diferencia (T - Ta) se reduce al factor 1/e ~ 0.3679.
        """
        params = CoolingParameters(T0=80.0, Ta=25.0, t1=5.0, Tm=55.0, t2=15.0, Tgoal=30.0)
        solver = NewtonCoolingSolver(params)
        tau = solver.time_constant
        T_tau = solver.temperature_at(tau)
        diferencia_inicial = params.T0 - params.Ta
        diferencia_en_tau = T_tau - params.Ta
        assert diferencia_en_tau == pytest.approx(
            diferencia_inicial / math.e, rel=1e-10
        )

    def test_relationship_half_life_tau(self):
        """La vida media y tau se relacionan por t_1/2 = ln(2) * tau."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        assert solver.half_life == pytest.approx(
            math.log(2) * solver.time_constant, rel=1e-12
        )

    def test_half_life_positive(self):
        """half_life debe ser siempre positivo."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        assert solver.half_life > 0

    def test_time_constant_positive(self):
        """time_constant debe ser siempre positivo."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        assert solver.time_constant > 0

    def test_half_life_shorter_with_larger_k(self):
        """Un k mayor implica una vida media mas corta."""
        params_slow = CoolingParameters(T0=100.0, Ta=20.0, t1=5.0, Tm=90.0, t2=15.0, Tgoal=25.0)
        params_fast = CoolingParameters(T0=100.0, Ta=20.0, t1=5.0, Tm=50.0, t2=15.0, Tgoal=25.0)
        solver_slow = NewtonCoolingSolver(params_slow)
        solver_fast = NewtonCoolingSolver(params_fast)
        assert solver_fast.half_life < solver_slow.half_life

    def test_time_constant_shorter_with_larger_k(self):
        """Un k mayor implica una constante de tiempo mas corta."""
        params_slow = CoolingParameters(T0=100.0, Ta=20.0, t1=5.0, Tm=90.0, t2=15.0, Tgoal=25.0)
        params_fast = CoolingParameters(T0=100.0, Ta=20.0, t1=5.0, Tm=50.0, t2=15.0, Tgoal=25.0)
        solver_slow = NewtonCoolingSolver(params_slow)
        solver_fast = NewtonCoolingSolver(params_fast)
        assert solver_fast.time_constant < solver_slow.time_constant


# ============================================================================
# 5. Pruebas de solve (parametrizado sobre ESCENARIOS)
# ============================================================================

class TestSolve:
    """Pruebas de la funcion solve con los escenarios de ESCENARIOS."""

    @pytest.mark.parametrize("nombre,params", list(ESCENARIOS.items()))
    def test_all_scenarios_produce_valid_results(self, nombre, params):
        """Verifica que solve retorne resultados coherentes para cada escenario."""
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()

        # k debe ser positivo
        assert resultado.k > 0, f"Escenario '{nombre}': k debe ser positivo"

        # T2 debe estar entre Ta y T0
        assert params.Ta < resultado.T2 < params.T0, (
            f"Escenario '{nombre}': T2={resultado.T2} fuera de rango"
        )

        # t_goal debe ser positivo
        assert resultado.t_goal > 0, (
            f"Escenario '{nombre}': t_goal debe ser positivo"
        )

        # half_life y tau deben ser positivos
        assert resultado.half_life > 0
        assert resultado.tau > 0

    @pytest.mark.parametrize("nombre,params", list(ESCENARIOS.items()))
    def test_k_consistency_with_direct_calculation(self, nombre, params):
        """Verificar que k del solve coincida con solver.k directamente."""
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        assert resultado.k == pytest.approx(solver.k, rel=1e-12)

    @pytest.mark.parametrize("nombre,params", list(ESCENARIOS.items()))
    def test_T2_consistency_with_temperature_at(self, nombre, params):
        """Verificar que T2 coincida con temperature_at evaluada en t2."""
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        T2_directo = solver.temperature_at(params.t2)
        assert resultado.T2 == pytest.approx(T2_directo, rel=1e-12)

    @pytest.mark.parametrize("nombre,params", list(ESCENARIOS.items()))
    def test_t_goal_consistency(self, nombre, params):
        """Verificar que evaluar T en t_goal de como resultado Tgoal."""
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        T_check = solver.temperature_at(resultado.t_goal)
        assert T_check == pytest.approx(params.Tgoal, rel=1e-9)

    @pytest.mark.parametrize("nombre,params", list(ESCENARIOS.items()))
    def test_half_life_consistency(self, nombre, params):
        """Verificar que half_life del resultado coincida con solver.half_life."""
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        assert resultado.half_life == pytest.approx(solver.half_life, rel=1e-12)

    @pytest.mark.parametrize("nombre,params", list(ESCENARIOS.items()))
    def test_tau_consistency(self, nombre, params):
        """Verificar que tau del resultado coincida con solver.time_constant."""
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        assert resultado.tau == pytest.approx(solver.time_constant, rel=1e-12)

    def test_gpu_gaming_numerical_verification(self):
        """Verificacion numerica puntual para el escenario GPU gaming."""
        params = ESCENARIOS["GPU gaming (alta carga)"]
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()

        # k esperado ~ 0.10513
        assert resultado.k == pytest.approx(0.10513, rel=1e-3)

        # T2 en t=12 min
        T2_esperado = 28.0 + 67.0 * math.exp(-resultado.k * 12.0)
        assert resultado.T2 == pytest.approx(T2_esperado, rel=1e-10)

        # half_life = ln(2)/k
        assert resultado.half_life == pytest.approx(
            math.log(2) / resultado.k, rel=1e-10
        )

        # tau = 1/k
        assert resultado.tau == pytest.approx(1.0 / resultado.k, rel=1e-10)

    def test_gpu_servidor_numerical(self):
        """Verificacion numerica para el escenario GPU servidor."""
        params = ESCENARIOS["GPU servidor (data center)"]
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        # k = -ln((55-20)/(85-20))/6 = -ln(35/65)/6
        k_esperado = -math.log(35.0 / 65.0) / 6.0
        assert resultado.k == pytest.approx(k_esperado, rel=1e-10)

    def test_bateria_ev_numerical(self):
        """Verificacion numerica para el escenario Bateria EV."""
        params = ESCENARIOS["Batería EV (post carga rápida)"]
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        k_esperado = -math.log((42.0 - 25.0) / (52.0 - 25.0)) / 10.0
        assert resultado.k == pytest.approx(k_esperado, rel=1e-10)


# ============================================================================
# 6. Pruebas de generate_table
# ============================================================================

class TestGenerateTable:
    """Pruebas para la generacion de datos tabulares via solver.generate_table."""

    def setup_method(self):
        """Configuracion comun: escenario GPU gaming."""
        self.params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        self.solver = NewtonCoolingSolver(self.params)
        self.t_max = 30.0

    def test_table_includes_t1_and_t2(self):
        """La tabla debe contener filas para t1 y t2."""
        tabla = self.solver.generate_table(self.t_max)
        tiempos = [fila["t (min)"] for fila in tabla]
        assert self.params.t1 in tiempos, f"t1={self.params.t1} no encontrado en {tiempos}"
        assert self.params.t2 in tiempos, f"t2={self.params.t2} no encontrado en {tiempos}"

    def test_table_correct_column_names(self):
        """Cada fila de la tabla debe tener las 4 columnas esperadas."""
        tabla = self.solver.generate_table(self.t_max)
        columnas_esperadas = {"t (min)", "T(t) °C", "T − Tₐ (°C)", "% enfriado"}
        for fila in tabla:
            assert set(fila.keys()) == columnas_esperadas

    def test_table_sorted_by_time(self):
        """Los tiempos en la tabla deben estar en orden creciente."""
        tabla = self.solver.generate_table(self.t_max)
        tiempos = [fila["t (min)"] for fila in tabla]
        assert tiempos == sorted(tiempos)

    def test_table_first_row_at_t_zero(self):
        """La primera fila debe corresponder a t=0."""
        tabla = self.solver.generate_table(self.t_max)
        assert tabla[0]["t (min)"] == 0.0

    def test_table_percentage_in_range(self):
        """El porcentaje de enfriamiento debe estar entre 0 y 100."""
        tabla = self.solver.generate_table(self.t_max)
        for fila in tabla:
            assert 0.0 <= fila["% enfriado"] <= 100.0, (
                f"% enfriado fuera de rango en t={fila['t (min)']}"
            )

    def test_table_T_minus_Ta_consistent(self):
        """La columna T - Ta debe ser consistente con T(t) - Ta."""
        tabla = self.solver.generate_table(self.t_max)
        for fila in tabla:
            diff_esperada = fila["T(t) °C"] - self.params.Ta
            assert fila["T − Tₐ (°C)"] == pytest.approx(diff_esperada, abs=0.01)

    def test_table_n_rows_parameter(self):
        """Verificar que n_rows controla la cantidad aproximada de filas."""
        tabla_10 = self.solver.generate_table(self.t_max, n_rows=10)
        tabla_30 = self.solver.generate_table(self.t_max, n_rows=30)
        # Con mas filas, la tabla deberia tener mas entradas
        assert len(tabla_30) >= len(tabla_10)

    def test_table_has_at_least_one_row(self):
        """La tabla siempre debe tener al menos una fila."""
        tabla = self.solver.generate_table(self.t_max)
        assert len(tabla) > 0

    def test_table_temperature_at_t0_is_T0(self):
        """La temperatura en t=0 de la tabla debe ser T0."""
        tabla = self.solver.generate_table(self.t_max)
        assert tabla[0]["T(t) °C"] == pytest.approx(self.params.T0, abs=0.01)

    def test_table_percentage_at_t0_is_zero(self):
        """El porcentaje de enfriamiento en t=0 debe ser 0."""
        tabla = self.solver.generate_table(self.t_max)
        assert tabla[0]["% enfriado"] == pytest.approx(0.0, abs=0.1)

    def test_table_all_rows_have_four_columns(self):
        """Todas las filas deben tener exactamente 4 columnas."""
        tabla = self.solver.generate_table(self.t_max)
        for fila in tabla:
            assert len(fila) == 4


# ============================================================================
# 7. Pruebas de generate_curve
# ============================================================================

class TestGenerateCurve:
    """Pruebas para la generacion de datos de curva via solver.generate_curve."""

    def setup_method(self):
        """Configuracion comun."""
        self.params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        self.solver = NewtonCoolingSolver(self.params)
        self.t_max = 30.0
        self.n_points = 500

    def test_correct_shapes(self):
        """Los arrays t y T deben tener la misma longitud (n_points default=500)."""
        t_arr, T_arr = self.solver.generate_curve(self.t_max)
        assert t_arr.shape == (500,)
        assert T_arr.shape == (500,)

    def test_custom_n_points(self):
        """Verificar que se respete un n_points diferente al default."""
        n = 100
        t_arr, T_arr = self.solver.generate_curve(self.t_max, n_points=n)
        assert len(t_arr) == n
        assert len(T_arr) == n

    def test_first_point_is_T0(self):
        """El primer punto de T debe ser T0 (t=0)."""
        t_arr, T_arr = self.solver.generate_curve(self.t_max)
        assert t_arr[0] == pytest.approx(0.0)
        assert T_arr[0] == pytest.approx(self.params.T0, rel=1e-10)

    def test_last_point_near_Ta_for_large_t_max(self):
        """El ultimo punto debe estar cerca de Ta para t_max suficientemente grande."""
        t_arr, T_arr = self.solver.generate_curve(t_max=200.0)
        # Con t_max=200 y k~0.105, e^(-0.105*200) ~ e^(-21) ~ 7e-10
        assert T_arr[-1] == pytest.approx(self.params.Ta, abs=0.01)

    def test_time_spans_0_to_t_max(self):
        """t_arr debe ir de 0 a t_max."""
        t_arr, T_arr = self.solver.generate_curve(self.t_max)
        assert t_arr[0] == pytest.approx(0.0)
        assert t_arr[-1] == pytest.approx(self.t_max)

    def test_temperature_monotonically_decreasing(self):
        """Toda la curva de temperatura debe ser monotonamente decreciente."""
        _, T_arr = self.solver.generate_curve(self.t_max)
        diferencias = np.diff(T_arr)
        assert np.all(diferencias <= 0), "La temperatura debe decrecer monotonamente"

    def test_temperature_always_geq_Ta(self):
        """Todos los valores de T deben ser >= Ta."""
        _, T_arr = self.solver.generate_curve(self.t_max)
        assert np.all(T_arr >= self.params.Ta)

    def test_curve_with_n_points_1000(self):
        """Verificar curva con 1000 puntos."""
        t_arr, T_arr = self.solver.generate_curve(self.t_max, n_points=1000)
        assert len(t_arr) == 1000
        assert len(T_arr) == 1000

    def test_curve_t_array_increasing(self):
        """t_arr debe ser estrictamente creciente."""
        t_arr, _ = self.solver.generate_curve(self.t_max)
        diffs = np.diff(t_arr)
        assert np.all(diffs > 0)

    def test_curve_intermediate_value(self):
        """Verificar que un valor intermedio de la curva sea correcto."""
        t_arr, T_arr = self.solver.generate_curve(self.t_max, n_points=500)
        # Tomar un punto intermedio y verificar contra temperature_at
        mid_idx = 250
        t_mid = t_arr[mid_idx]
        T_expected = self.solver.temperature_at(t_mid)
        assert T_arr[mid_idx] == pytest.approx(T_expected, rel=1e-10)


# ============================================================================
# 8. Pruebas de classify_k
# ============================================================================

class TestClassifyK:
    """Pruebas para la clasificacion del valor de k."""

    def test_k_high(self):
        """k > 0.1 se clasifica como 'alta'."""
        clasificacion, descripcion = NewtonCoolingSolver.classify_k(0.15)
        assert clasificacion == "alta"
        assert "líquida" in descripcion or "activa" in descripcion

    def test_k_high_at_0_11(self):
        """k = 0.11 (justo por encima de 0.1) debe ser 'alta'."""
        clasificacion, _ = NewtonCoolingSolver.classify_k(0.11)
        assert clasificacion == "alta"

    def test_k_moderate(self):
        """k entre 0.03 y 0.1 se clasifica como 'moderada'."""
        clasificacion, descripcion = NewtonCoolingSolver.classify_k(0.06)
        assert clasificacion == "moderada"
        assert "aluminio" in descripcion or "disipadores" in descripcion

    def test_k_moderate_exact_0_1(self):
        """k = 0.1 exacto debe ser 'moderada' (no 'alta')."""
        clasificacion, _ = NewtonCoolingSolver.classify_k(0.1)
        assert clasificacion == "moderada"

    def test_k_moderate_at_0_031(self):
        """k = 0.031 (justo por encima de 0.03) debe ser 'moderada'."""
        clasificacion, _ = NewtonCoolingSolver.classify_k(0.031)
        assert clasificacion == "moderada"

    def test_k_low(self):
        """k <= 0.03 se clasifica como 'baja'."""
        clasificacion, descripcion = NewtonCoolingSolver.classify_k(0.02)
        assert clasificacion == "baja"
        assert "pasiva" in descripcion or "confinado" in descripcion

    def test_k_low_exact_0_03(self):
        """k = 0.03 exacto debe ser 'baja'."""
        clasificacion, _ = NewtonCoolingSolver.classify_k(0.03)
        assert clasificacion == "baja"

    def test_k_very_low_0_001(self):
        """k muy pequeno sigue siendo 'baja'."""
        clasificacion, _ = NewtonCoolingSolver.classify_k(0.001)
        assert clasificacion == "baja"

    def test_k_very_high_5_0(self):
        """k muy alto sigue siendo 'alta'."""
        clasificacion, _ = NewtonCoolingSolver.classify_k(5.0)
        assert clasificacion == "alta"

    def test_classify_returns_tuple_of_strings(self):
        """classify_k debe retornar una tupla de dos strings."""
        result = NewtonCoolingSolver.classify_k(0.05)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_classify_k_0_05_moderate(self):
        """k = 0.05 esta en el rango moderado."""
        clasificacion, _ = NewtonCoolingSolver.classify_k(0.05)
        assert clasificacion == "moderada"

    def test_classify_k_0_2_alta(self):
        """k = 0.2 esta en el rango alta."""
        clasificacion, _ = NewtonCoolingSolver.classify_k(0.2)
        assert clasificacion == "alta"

    def test_classify_k_0_01_baja(self):
        """k = 0.01 esta en el rango baja."""
        clasificacion, _ = NewtonCoolingSolver.classify_k(0.01)
        assert clasificacion == "baja"


# ============================================================================
# 9. Casos extremos (edge cases)
# ============================================================================

class TestEdgeCases:
    """Pruebas de casos extremos y comportamiento limite."""

    # --- k muy pequeno (enfriamiento lento) ---

    def test_very_small_k_temperature_barely_changes(self):
        """Con k muy pequeno, la temperatura apenas cambia en poco tiempo."""
        params = CoolingParameters(T0=100.0, Ta=20.0, t1=10.0, Tm=99.9, t2=50.0, Tgoal=25.0)
        solver = NewtonCoolingSolver(params)
        T_1min = solver.temperature_at(1.0)
        # Con k muy pequeno, T(1) debe estar muy cerca de T0
        assert T_1min == pytest.approx(params.T0, rel=1e-3)

    def test_very_small_k_long_half_life(self):
        """Con k muy pequeno, la vida media es muy grande."""
        params = CoolingParameters(T0=100.0, Ta=20.0, t1=10.0, Tm=99.9, t2=50.0, Tgoal=25.0)
        solver = NewtonCoolingSolver(params)
        assert solver.half_life > 1000  # vida media > 1000 minutos

    def test_very_small_k_huge_t_goal(self):
        """Con k muy pequeno, el tiempo para alcanzar una T baja es enorme."""
        params = CoolingParameters(T0=100.0, Ta=20.0, t1=10.0, Tm=99.9, t2=50.0, Tgoal=25.0)
        solver = NewtonCoolingSolver(params)
        t = solver.time_for_temperature(25.0)
        assert t > 10000  # mas de 10000 minutos

    # --- k muy grande (enfriamiento rapido) ---

    def test_very_large_k_fast_cooling(self):
        """Con k muy grande, la temperatura cae rapidamente a Ta."""
        params = CoolingParameters(T0=100.0, Ta=20.0, t1=1.0, Tm=25.0, t2=5.0, Tgoal=21.0)
        solver = NewtonCoolingSolver(params)
        T_5min = solver.temperature_at(5.0)
        # Debe estar muy cerca de Ta
        assert T_5min == pytest.approx(params.Ta, abs=0.1)

    def test_very_large_k_short_half_life(self):
        """Con k muy grande, la vida media es muy pequena."""
        params = CoolingParameters(T0=100.0, Ta=20.0, t1=1.0, Tm=25.0, t2=5.0, Tgoal=21.0)
        solver = NewtonCoolingSolver(params)
        assert solver.half_life < 1.0  # vida media < 1 minuto

    # --- Tiempos extremos ---

    def test_very_large_time_T_essentially_Ta(self):
        """En un tiempo enorme, la temperatura debe ser esencialmente Ta."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        T = solver.temperature_at(1e10)
        assert T == pytest.approx(params.Ta, abs=1e-10)

    def test_T_at_zero_is_T0_for_any_k(self):
        """Para cualquier escenario, T(0) = T0."""
        for nombre, p in ESCENARIOS.items():
            solver = NewtonCoolingSolver(p)
            T = solver.temperature_at(0.0)
            assert T == pytest.approx(p.T0, abs=1e-12), (
                f"Escenario '{nombre}': T(0)={T} != T0={p.T0}"
            )

    # --- Diferencia minima T0 - Ta ---

    def test_small_T0_minus_Ta_difference(self):
        """Con T0 apenas por encima de Ta, los calculos deben seguir funcionando."""
        params = CoolingParameters(T0=20.1, Ta=20.0, t1=1.0, Tm=20.05, t2=5.0, Tgoal=20.02)
        solver = NewtonCoolingSolver(params)
        assert solver.k > 0
        T = solver.temperature_at(0.5)
        assert params.Ta < T < params.T0

    # --- Curva con pocos puntos ---

    def test_curve_with_2_points(self):
        """generate_curve debe funcionar con n_points=2 (minimo util)."""
        params = CoolingParameters(T0=80.0, Ta=20.0, t1=3.0, Tm=60.0, t2=10.0, Tgoal=25.0)
        solver = NewtonCoolingSolver(params)
        t_arr, T_arr = solver.generate_curve(10.0, n_points=2)
        assert len(t_arr) == 2
        assert len(T_arr) == 2
        assert t_arr[0] == pytest.approx(0.0)
        assert t_arr[1] == pytest.approx(10.0)

    # --- Solver con valores extremos ---

    def test_solver_slow_cooling_scenario(self):
        """
        Escenario con enfriamiento muy lento: Tm casi igual a T0.
        """
        params = CoolingParameters(
            T0=100.0, Ta=20.0, t1=10.0, Tm=99.0, t2=50.0, Tgoal=25.0
        )
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        assert resultado.k > 0
        assert resultado.k < 0.02  # k debe ser muy bajo
        assert resultado.T2 > params.Ta
        assert resultado.t_goal > 0

    def test_solver_fast_cooling_scenario(self):
        """
        Escenario con enfriamiento rapido: Tm mucho menor que T0.
        """
        params = CoolingParameters(
            T0=100.0, Ta=20.0, t1=2.0, Tm=30.0, t2=5.0, Tgoal=22.0
        )
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        assert resultado.k > 0.5  # k debe ser alto
        assert resultado.T2 > params.Ta
        assert resultado.t_goal > 0

    # --- Table con tiempos fraccionarios ---

    def test_table_with_fractional_t1_t2(self):
        """
        Verificar que t1 y t2 fraccionarios se incluyan en la tabla.
        """
        params = CoolingParameters(T0=80.0, Ta=20.0, t1=3.5, Tm=55.0, t2=7.5, Tgoal=25.0)
        solver = NewtonCoolingSolver(params)
        tabla = solver.generate_table(t_max=20.0)
        tiempos = [fila["t (min)"] for fila in tabla]
        assert 3.5 in tiempos
        assert 7.5 in tiempos

    # --- Resultado completo para scenario lento ---

    def test_slow_cooling_half_life_is_large(self):
        """Con enfriamiento lento, la vida media debe ser grande."""
        params = CoolingParameters(
            T0=100.0, Ta=20.0, t1=10.0, Tm=99.0, t2=50.0, Tgoal=25.0
        )
        solver = NewtonCoolingSolver(params)
        assert solver.half_life > 50

    def test_fast_cooling_half_life_is_small(self):
        """Con enfriamiento rapido, la vida media debe ser corta."""
        params = CoolingParameters(
            T0=100.0, Ta=20.0, t1=2.0, Tm=30.0, t2=5.0, Tgoal=22.0
        )
        solver = NewtonCoolingSolver(params)
        assert solver.half_life < 1.0

    # --- Temperatura intermedia correcta ---

    def test_intermediate_temperature_correct(self):
        """Verificar que una temperatura intermedia sea coherente."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        T_6 = solver.temperature_at(6.0)
        # T(6) = 28 + 67*exp(-k*6), debe estar entre Tm(t=4) y T(t=12)
        T_at_t2 = solver.temperature_at(12.0)
        assert self.params_Tm_value(params) > T_6 > T_at_t2

    def params_Tm_value(self, params):
        """Helper para obtener Tm."""
        return params.Tm

    # --- Curve and Table consistency ---

    def test_curve_and_temperature_at_consistency(self):
        """Los valores de la curva deben coincidir con temperature_at."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        t_arr, T_arr = solver.generate_curve(30.0, n_points=50)
        for i in range(len(t_arr)):
            T_expected = solver.temperature_at(t_arr[i])
            assert T_arr[i] == pytest.approx(T_expected, rel=1e-10)

    def test_solve_results_dataclass_fields(self):
        """Verificar que el resultado de solve tenga los campos correctos."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        resultado = solver.solve()
        assert hasattr(resultado, "k")
        assert hasattr(resultado, "T2")
        assert hasattr(resultado, "t_goal")
        assert hasattr(resultado, "half_life")
        assert hasattr(resultado, "tau")

    def test_multiple_solvers_independent(self):
        """Dos solvers con parametros distintos deben ser independientes."""
        params1 = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        params2 = CoolingParameters(T0=85.0, Ta=20.0, t1=6.0, Tm=55.0, t2=20.0, Tgoal=25.0)
        solver1 = NewtonCoolingSolver(params1)
        solver2 = NewtonCoolingSolver(params2)
        assert solver1.k != solver2.k
        assert solver1.temperature_at(5.0) != solver2.temperature_at(5.0)

    def test_generate_curve_returns_numpy_arrays(self):
        """generate_curve debe retornar arrays de numpy."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        t_arr, T_arr = solver.generate_curve(30.0)
        assert isinstance(t_arr, np.ndarray)
        assert isinstance(T_arr, np.ndarray)

    def test_generate_table_returns_list_of_dicts(self):
        """generate_table debe retornar una lista de diccionarios."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        tabla = solver.generate_table(30.0)
        assert isinstance(tabla, list)
        for fila in tabla:
            assert isinstance(fila, dict)

    def test_large_t_max_table_still_valid(self):
        """Una tabla con t_max grande debe seguir teniendo valores validos."""
        params = CoolingParameters(T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0)
        solver = NewtonCoolingSolver(params)
        tabla = solver.generate_table(t_max=500.0, n_rows=50)
        for fila in tabla:
            assert fila["T(t) °C"] >= params.Ta - 0.01
            assert 0.0 <= fila["% enfriado"] <= 100.1
