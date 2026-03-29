"""
Escenarios predefinidos — Ley de Enfriamiento de Newton
Configuraciones realistas para distintos casos de uso.
"""

from models import CoolingParameters

ESCENARIOS: dict[str, CoolingParameters] = {
    "GPU gaming (alta carga)": CoolingParameters(
        T0=95.0, Ta=28.0, t1=4.0, Tm=72.0, t2=12.0, Tgoal=35.0
    ),
    "GPU servidor (data center)": CoolingParameters(
        T0=85.0, Ta=20.0, t1=6.0, Tm=55.0, t2=20.0, Tgoal=25.0
    ),
    "Batería EV (post carga rápida)": CoolingParameters(
        T0=52.0, Ta=25.0, t1=10.0, Tm=42.0, t2=30.0, Tgoal=28.0
    ),
    "CPU laptop (uso normal)": CoolingParameters(
        T0=78.0, Ta=22.0, t1=3.0, Tm=58.0, t2=10.0, Tgoal=28.0
    ),
}
