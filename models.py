"""
Modelos de datos para el problema de la Ley de Enfriamiento de Newton.

Define las estructuras de entrada (parametros) y salida (resultados)
como dataclasses con validacion y propiedades derivadas.
"""

from dataclasses import dataclass


@dataclass
class CoolingParameters:
    """Parametros de entrada del problema de enfriamiento de Newton."""

    T0: float       # Temperatura inicial (grados C)
    Ta: float       # Temperatura ambiente (grados C)
    t1: float       # Tiempo de la medicion conocida (min)
    Tm: float       # Temperatura medida en t1 (grados C)
    t2: float       # Tiempo a evaluar (min)
    Tgoal: float    # Temperatura objetivo (grados C)

    # ── Propiedades derivadas ──────────────────────────────────────

    @property
    def initial_difference(self) -> float:
        """Diferencia inicial: T0 - Ta."""
        return self.T0 - self.Ta

    @property
    def measured_difference(self) -> float:
        """Diferencia medida: Tm - Ta."""
        return self.Tm - self.Ta

    @property
    def goal_difference(self) -> float:
        """Diferencia objetivo: Tgoal - Ta."""
        return self.Tgoal - self.Ta

    @property
    def measurement_ratio(self) -> float:
        """Razon de medicion: (Tm - Ta) / (T0 - Ta)."""
        return self.measured_difference / self.initial_difference

    # ── Validacion ─────────────────────────────────────────────────

    def validate(self) -> tuple[bool, str]:
        """
        Verifica que los parametros sean fisicamente validos.

        Retorna (es_valido, mensaje_error). Si es valido, el mensaje
        es una cadena vacia.
        """
        if self.T0 <= self.Ta:
            return False, "T₀ debe ser mayor que Tₐ"
        if self.Tm <= self.Ta:
            return False, "T(t₁) debe ser mayor que Tₐ"
        if self.Tm >= self.T0:
            return False, "T(t₁) debe ser menor que T₀"
        if self.t1 <= 0:
            return False, "t₁ debe ser mayor que 0"
        if self.Tgoal <= self.Ta:
            return False, "T* debe ser mayor que Tₐ"
        if self.Tgoal >= self.T0:
            return False, "T* debe ser menor que T₀"
        return True, ""

    # AGENTE-4: métodos de formato y validación en español

    def format_value(self, value: float, decimals: int = 2) -> str:
        """Formatea un valor numérico con punto decimal (no coma)."""
        return f"{value:.{decimals}f}"

    def validation_messages(self) -> dict[str, str]:
        """Retorna mensajes de validación para cada campo en español."""
        return {
            "T0": f"T₀ debe ser mayor que Tₐ ({self.Ta}°C)",
            "Ta": f"Tₐ debe ser menor que T₀ ({self.T0}°C)",
            "t1": "t₁ debe ser mayor que 0",
            "Tm": f"T(t₁) debe estar entre Tₐ ({self.Ta}°C) y T₀ ({self.T0}°C)",
            "Tgoal": f"T* debe estar entre Tₐ ({self.Ta}°C) y T₀ ({self.T0}°C)",
        }


@dataclass
class CoolingResults:
    """Resultados calculados del problema de enfriamiento."""

    k: float            # Constante de enfriamiento (min^-1)
    T2: float           # Temperatura en t2 (grados C)
    t_goal: float       # Tiempo para alcanzar Tgoal (min)
    half_life: float    # Vida media termica (min)
    tau: float          # Constante de tiempo tau = 1/k (min)
