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
    "☕ Café en la oficina": CoolingParameters(
        T0=90.0, Ta=22.0, t1=4.0, Tm=72.0, t2=10.0, Tgoal=30.0
    ),
    "🏥 Cuerpo humano (forense)": CoolingParameters(
        T0=37.0, Ta=18.0, t1=60.0, Tm=33.0, t2=120.0, Tgoal=20.0
    ),
}

# AGENTE-4: Datos para los problemas de práctica (usados por app.py)
PROBLEMAS_PRACTICA: list[dict] = [
    {
        "nombre": "☕ Café en la oficina",
        "desc": "Una taza de café se sirve a **90°C** en una oficina a **22°C**.\nDespués de **4 minutos**, la temperatura baja a **72°C**.",
        "preguntas": "(a) Encontrá la constante k.\n(b) ¿Cuál será la temperatura a los 10 minutos?\n(c) ¿Cuánto tarda en llegar a 30°C?",
        "params": CoolingParameters(T0=90.0, Ta=22.0, t1=4.0, Tm=72.0, t2=10.0, Tgoal=30.0),
    },
    {
        "nombre": "🏭 Motor industrial",
        "desc": "Un motor industrial se apaga a **180°C** en una planta a **28°C**.\nTras **10 minutos**, la carcasa marca **130°C**.",
        "preguntas": "(a) Encontrá la constante k.\n(b) ¿Cuál será la temperatura a los 30 minutos?\n(c) ¿Cuánto tarda en ser seguro tocarlo (40°C)?",
        "params": CoolingParameters(T0=180.0, Ta=28.0, t1=10.0, Tm=130.0, t2=30.0, Tgoal=40.0),
    },
    {
        "nombre": "🧪 Muestra de laboratorio",
        "desc": "Una muestra biológica a **37°C** se coloca en un refrigerador a **4°C**.\nDespués de **6 minutos**, la temperatura es **30°C**.",
        "preguntas": "(a) Encontrá la constante k.\n(b) ¿Cuál será la temperatura a los 15 minutos?\n(c) ¿Cuánto tarda en llegar a 8°C para almacenamiento seguro?",
        "params": CoolingParameters(T0=37.0, Ta=4.0, t1=6.0, Tm=30.0, t2=15.0, Tgoal=8.0),
    },
]
