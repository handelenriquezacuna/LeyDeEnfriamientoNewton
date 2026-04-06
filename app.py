"""
Ley de Enfriamiento de Newton — Simulador Interactivo
Aplicación de EDO: Disipación Térmica en GPU
Universidad Fidélitas | MA-106 Ecuaciones Diferenciales
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models import CoolingParameters
from solver import NewtonCoolingSolver
from scenarios import ESCENARIOS
from charts import CoolingVisualizer

# ─── Page config ───
st.set_page_config(
    page_title="Ley de Enfriamiento de Newton",
    page_icon="🌡️",
    layout="wide",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main .block-container { max-width: 1100px; padding-top: 2rem; }

    .stMetric > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        color: #1a1a2e !important;
    }
    .stMetric > div label, .stMetric > div [data-testid="stMetricValue"],
    .stMetric > div [data-testid="stMetricDelta"] {
        color: #1a1a2e !important;
    }

    .math-step {
        background: #f8f9fc;
        border-left: 4px solid #4361ee;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin: 10px 0;
        color: #1a1a2e;
    }

    .step-label {
        font-size: 12px;
        font-weight: 600;
        color: #4361ee;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    .step-desc {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 8px;
    }

    .problem-box {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
        font-size: 15px;
        line-height: 1.8;
        color: #1a1a2e !important;
    }

    .result-highlight {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 1px solid #6ee7b7;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        color: #065f46 !important;
    }

    .analysis-box {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 12px;
        padding: 20px 24px;
        line-height: 1.8;
        color: #1a1a1a !important;
    }

    div[data-testid="stSidebar"] { background: #f8f9fc; }

    .header-badge {
        display: inline-block;
        background: #4361ee;
        color: white;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        margin-left: 8px;
        vertical-align: middle;
    }

    .glosario-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #7dd3fc;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 13px;
        line-height: 1.9;
        color: #0c4a6e !important;
    }
    .glosario-box code {
        background: #bae6fd;
        color: #0c4a6e;
        padding: 1px 5px;
        border-radius: 4px;
        font-weight: 600;
    }

    .intuition-box {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border: 1px solid #d8b4fe;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0 16px 0;
        font-size: 14px;
        line-height: 1.7;
        color: #3b0764 !important;
    }
    .intuition-box strong { color: #7c3aed; }

    .quiz-box {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border: 1px solid #fdba74;
        border-radius: 12px;
        padding: 20px 24px;
        line-height: 1.7;
        color: #7c2d12 !important;
    }

    .quiz-success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 12px 16px;
        color: #14532d !important;
        font-size: 14px;
    }

    .quiz-fail {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fca5a5;
        border-radius: 10px;
        padding: 12px 16px;
        color: #7f1d1d !important;
        font-size: 14px;
    }

    .progress-thermal {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        color: #1e293b !important;
    }
    .progress-thermal .bar-bg {
        background: #e2e8f0;
        border-radius: 6px;
        height: 24px;
        overflow: hidden;
        margin: 8px 0;
    }
    .progress-thermal .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── Defaults y session_state (única fuente de verdad) ───
_DEFAULTS = {
    "input_T0": 92.0,
    "input_Ta": 24.0,
    "input_t1": 5.0,
    "input_Tm": 68.0,
    "input_t2": 15.0,
    "input_Tgoal": 30.0,
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ─── Callbacks ───
def _cargar_escenario():
    """Actualiza los number_input vía session_state cuando se elige un escenario."""
    nombre = st.session_state.preset_select
    if nombre != "— Personalizado —" and nombre in ESCENARIOS:
        esc = ESCENARIOS[nombre]
        st.session_state.input_T0 = esc.T0
        st.session_state.input_Ta = esc.Ta
        st.session_state.input_t1 = esc.t1
        st.session_state.input_Tm = esc.Tm
        st.session_state.input_t2 = esc.t2
        st.session_state.input_Tgoal = esc.Tgoal


def _restablecer():
    """Restaura los valores por defecto del problema original."""
    for key, val in _DEFAULTS.items():
        st.session_state[key] = val
    st.session_state.preset_select = "— Personalizado —"


# ─── Sidebar: Escenarios y Parámetros ───
with st.sidebar:
    st.markdown("## ⚙️ Parámetros del problema")
    st.markdown("Ajustá los valores para ver cómo cambia la solución completa.")

    st.markdown("---")

    modo_aprendizaje = st.toggle(
        "Modo aprendizaje",
        value=True,
        help="Activa explicaciones intuitivas, analogías y un quiz interactivo",
    )

    if modo_aprendizaje:
        with st.expander("📖 Glosario de variables", expanded=False):
            st.markdown("""
<div class="glosario-box">
<code>T₀</code> — Temperatura inicial del objeto (al apagarse)<br>
<code>Tₐ</code> — Temperatura del ambiente (hacia donde converge)<br>
<code>t₁</code> — Momento en que se toma una segunda medición<br>
<code>T(t₁)</code> — Temperatura medida en el instante t₁<br>
<code>k</code> — Constante de enfriamiento (qué tan rápido se enfría)<br>
<code>τ</code> — Constante de tiempo: tras 1τ se pierde el 63% de la diferencia<br>
<code>t½</code> — Vida media: tiempo para que la diferencia se reduzca a la mitad<br>
<code>T*</code> — Temperatura objetivo que queremos alcanzar
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Escenarios predefinidos")

    opciones_escenario = ["— Personalizado —"] + list(ESCENARIOS.keys())
    preset = st.selectbox(
        "Cargar escenario:", opciones_escenario,
        key="preset_select", on_change=_cargar_escenario,
    )

    st.button("Restablecer valores", on_click=_restablecer, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌡️ Temperaturas")

    T0 = st.number_input(
        "Temperatura inicial T₀ (°C)",
        min_value=30.0, max_value=200.0, step=1.0,
        help="Temperatura de la GPU al momento de apagarse",
        key="input_T0",
    )

    Ta = st.number_input(
        "Temperatura ambiente Tₐ (°C)",
        min_value=0.0, max_value=50.0, step=1.0,
        help="Temperatura del entorno donde se enfría",
        key="input_Ta",
    )

    st.markdown("### 📏 Dato conocido")

    t1 = st.number_input(
        "Tiempo de medición t₁ (min)",
        min_value=0.5, max_value=60.0, step=0.5,
        help="Momento en el que se mide la segunda temperatura",
        key="input_t1",
    )

    Tm = st.number_input(
        "Temperatura en t₁ (°C)",
        min_value=0.0, max_value=200.0, step=1.0,
        help="Temperatura medida en el tiempo t₁",
        key="input_Tm",
    )

    st.markdown("### 🎯 Preguntas")

    t2 = st.number_input(
        "Tiempo a evaluar t₂ (min)",
        min_value=1.0, max_value=120.0, step=1.0,
        help="¿Cuál será la temperatura en este momento?",
        key="input_t2",
    )

    Tgoal = st.number_input(
        "Temperatura objetivo T* (°C)",
        min_value=0.0, max_value=200.0, step=1.0,
        help="¿Cuánto tiempo tarda en llegar a esta temperatura?",
        key="input_Tgoal",
    )


# ─── Construir parámetros y validar ───
params = CoolingParameters(T0=T0, Ta=Ta, t1=t1, Tm=Tm, t2=t2, Tgoal=Tgoal)
es_valido, msg_error = params.validate()


# ═══════════════════════════════════════════
# CONTENIDO PRINCIPAL
# ═══════════════════════════════════════════

st.markdown("# 🌡️ Ley de Enfriamiento de Newton")
st.markdown("**Aplicación de EDO** · Disipación térmica en GPU · Universidad Fidélitas · MA-106")
st.markdown("---")

if not es_valido:
    st.error(f"⚠️ Parámetros inválidos: {msg_error}. Se necesita T₀ > T(t₁) > Tₐ y t₁ > 0.")
    st.info("Los rangos de cada campo están limitados por Streamlit. "
            "Si un valor no se acepta, revisá que esté dentro del rango indicado.")
    st.stop()

# ─── Resolver ───
solver = NewtonCoolingSolver(params)
res = solver.solve()
k = res.k
T2 = res.T2
t_goal = res.t_goal
half_life = res.half_life
tau = res.tau


# ─── 1. PLANTEAMIENTO DEL PROBLEMA ───
st.markdown("## 📝 Planteamiento del problema")

st.markdown(f"""
<div class="problem-box">
<strong>Problema:</strong> Una GPU de alto rendimiento opera a <strong>{T0:.0f} °C</strong>
y se apaga en un ambiente a <strong>{Ta:.0f} °C</strong>.
Se sabe que tras <strong>{t1:.1f} minutos</strong> la temperatura desciende a <strong>{Tm:.0f} °C</strong>.
Determinar:<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;<strong>(a)</strong> La constante de enfriamiento <em>k</em><br>
&nbsp;&nbsp;&nbsp;&nbsp;<strong>(b)</strong> La temperatura tras <strong>{t2:.0f} minutos</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;<strong>(c)</strong> El tiempo necesario para alcanzar <strong>{Tgoal:.0f} °C</strong>
</div>
""", unsafe_allow_html=True)


# ─── 2. RESULTADOS RÁPIDOS ───
st.markdown("## 📊 Resultados")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("(a) Constante k", f"{k:.4f} min⁻¹", help="Tasa de enfriamiento")
with col2:
    st.metric(f"(b) T en t = {t2:.0f} min", f"{T2:.2f} °C",
              delta=f"{T2 - T0:.1f} °C vs T₀", delta_color="off")
with col3:
    st.metric(f"(c) Tiempo → {Tgoal:.0f} °C", f"{t_goal:.2f} min",
              help="Tiempo para alcanzar la temperatura objetivo")

col4, col5 = st.columns(2)
with col4:
    st.metric("Vida media térmica", f"{half_life:.2f} min",
              help="Tiempo para que (T−Tₐ) se reduzca a la mitad")
with col5:
    st.metric("Constante de tiempo τ", f"{tau:.2f} min",
              help="Tras 5τ el sistema alcanza ~99.3% del equilibrio")


# ─── 3. DESARROLLO MATEMÁTICO PASO A PASO ───
st.markdown("---")
st.markdown("## 📐 Desarrollo matemático paso a paso")

diff0 = params.initial_difference
diffm = params.measured_difference
ratio = params.measurement_ratio


def math_step(num: str, title: str, desc: str, latex: str):
    """Renderiza un paso matemático con estilo."""
    st.markdown(f"""
    <div class="math-step">
        <div class="step-label">{num}</div>
        <div class="step-desc">{title}</div>
    </div>
    """, unsafe_allow_html=True)
    if desc:
        st.markdown(f"*{desc}*")
    st.latex(latex)


# Paso 1
math_step(
    "Paso 1 — Ecuación diferencial",
    "Escribir la EDO que modela el fenómeno según la Ley de Enfriamiento de Newton",
    "La tasa de cambio de temperatura es proporcional a la diferencia entre el objeto y el ambiente:",
    r"\frac{dT}{dt} = -k(T - T_a)"
)

if modo_aprendizaje:
    st.markdown("""<div class="intuition-box">
    <strong>Intuición:</strong> Imagina que sostenés una taza de café caliente.
    Al principio se enfría rápido porque la diferencia con el ambiente es grande.
    Conforme se acerca a la temperatura del cuarto, se enfría cada vez más lento.
    Eso es exactamente lo que dice esta ecuación: <em>la velocidad de cambio depende
    de qué tan lejos estés del equilibrio</em>.
    </div>""", unsafe_allow_html=True)

# Paso 2
math_step(
    "Paso 2 — Separación de variables",
    "Llevar T a un lado y t al otro para poder integrar",
    "Se reorganiza la ecuación separando las variables T y t:",
    r"\frac{dT}{T - T_a} = -k \, dt"
)

if modo_aprendizaje:
    st.markdown("""<div class="intuition-box">
    <strong>¿Por qué separar variables?</strong> Es como organizar una ecuación
    para que cada lado hable de una sola cosa: el lado izquierdo solo tiene temperatura,
    el derecho solo tiene tiempo. Así podemos integrar cada lado por separado.
    </div>""", unsafe_allow_html=True)

# Paso 3
math_step(
    "Paso 3 — Integración",
    "Integrar ambos miembros de la ecuación",
    "El lado izquierdo es una integral de la forma ∫du/u = ln|u|:",
    r"\int \frac{dT}{T - T_a} = \int -k \, dt \quad \Longrightarrow \quad \ln|T - T_a| = -kt + C"
)

# Paso 4
math_step(
    "Paso 4 — Solución general",
    "Despejar T(t) aplicando la exponencial",
    "Se aplica e^(·) a ambos lados y se renombra la constante:",
    r"T(t) = T_a + (T_0 - T_a) \cdot e^{-kt}"
)

if modo_aprendizaje:
    st.markdown("""<div class="intuition-box">
    <strong>Esta es la ecuación clave.</strong> Dice que la temperatura
    arranca en T₀ y <em>decae exponencialmente</em> hacia Tₐ. El término
    e<sup>-kt</sup> es el que controla la velocidad: cuanto mayor sea k,
    más rápido cae el exponencial y más rápido se enfría el objeto.
    </div>""", unsafe_allow_html=True)

# Paso 5
math_step(
    "Paso 5 — Condición inicial",
    f"Verificar con T(0) = {T0:.0f} °C",
    "Al sustituir t = 0, e⁰ = 1, se confirma la condición inicial:",
    rf"T(0) = {Ta:.0f} + ({T0:.0f} - {Ta:.0f}) \cdot e^{{0}} = {Ta:.0f} + {diff0:.0f} \cdot 1 = {T0:.0f} \; °C \;\; \checkmark"
)

# Paso 6 (a) — Encontrar k
st.markdown("""
<div class="math-step">
    <div class="step-label">Paso 6 (a) — Encontrar k</div>
    <div class="step-desc">Usar el dato conocido T(t₁) para determinar la constante de enfriamiento</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"*Se sustituye T({t1:.1f}) = {Tm:.0f} °C en la solución:*")

st.latex(rf"{Tm:.0f} = {Ta:.0f} + {diff0:.0f} \cdot e^{{-k \cdot {t1:.1f}}}")

st.latex(rf"{Tm:.0f} - {Ta:.0f} = {diff0:.0f} \cdot e^{{-{t1:.1f}k}}")

st.latex(rf"{diffm:.0f} = {diff0:.0f} \cdot e^{{-{t1:.1f}k}}")

st.latex(rf"e^{{-{t1:.1f}k}} = \frac{{{diffm:.0f}}}{{{diff0:.0f}}} = {ratio:.4f}")

st.markdown("*Se aplica logaritmo natural a ambos lados:*")

st.latex(rf"-{t1:.1f}k = \ln({ratio:.4f}) = {np.log(ratio):.4f}")

st.latex(rf"k = \frac{{-({np.log(ratio):.4f})}}{{{t1:.1f}}}")

st.markdown(f"""
<div class="result-highlight">
    k = {k:.4f} min⁻¹
</div>
""", unsafe_allow_html=True)

if modo_aprendizaje:
    clasif_tmp, desc_tmp = NewtonCoolingSolver.classify_k(k)
    st.markdown(f"""<div class="intuition-box">
    <strong>¿Qué significa k = {k:.4f}?</strong> Es la "velocidad de enfriamiento".
    Un k de {k:.4f} se clasifica como <strong>{clasif_tmp}</strong>, típico de {desc_tmp}.
    Si k fuera el doble ({k*2:.4f}), el objeto se enfriaría al doble de velocidad.
    Probá cambiando los valores en el sidebar para ver cómo cambia k.
    </div>""", unsafe_allow_html=True)

st.markdown("")

# Paso 7 (b) — T(t2)
exp_val = np.exp(-k * t2)

st.markdown(f"""
<div class="math-step">
    <div class="step-label">Paso 7 (b) — Temperatura en t = {t2:.0f} min</div>
    <div class="step-desc">Sustituir el valor de k y t₂ en la solución</div>
</div>
""", unsafe_allow_html=True)

st.latex(rf"T({t2:.0f}) = {Ta:.0f} + {diff0:.0f} \cdot e^{{-({k:.4f})({t2:.0f})}}")

st.latex(rf"T({t2:.0f}) = {Ta:.0f} + {diff0:.0f} \cdot e^{{{-k * t2:.4f}}}")

st.latex(rf"T({t2:.0f}) = {Ta:.0f} + {diff0:.0f} \cdot {exp_val:.4f}")

st.latex(rf"T({t2:.0f}) = {Ta:.0f} + {diff0 * exp_val:.2f}")

st.markdown(f"""
<div class="result-highlight">
    T({t2:.0f}) = {T2:.2f} °C
</div>
""", unsafe_allow_html=True)

st.markdown("")

# Paso 8 (c) — Tiempo para Tgoal
diffg = params.goal_difference
ratio_g = diffg / diff0

st.markdown(f"""
<div class="math-step">
    <div class="step-label">Paso 8 (c) — Tiempo para alcanzar {Tgoal:.0f} °C</div>
    <div class="step-desc">Despejar t de la ecuación</div>
</div>
""", unsafe_allow_html=True)

st.latex(rf"{Tgoal:.0f} = {Ta:.0f} + {diff0:.0f} \cdot e^{{-{k:.4f} \cdot t}}")

st.latex(rf"{diffg:.0f} = {diff0:.0f} \cdot e^{{-{k:.4f} \cdot t}}")

st.latex(rf"e^{{-{k:.4f} \cdot t}} = \frac{{{diffg:.0f}}}{{{diff0:.0f}}} = {ratio_g:.4f}")

st.latex(rf"-{k:.4f} \cdot t = \ln({ratio_g:.4f}) = {np.log(ratio_g):.4f}")

st.latex(rf"t = \frac{{{np.log(ratio_g):.4f}}}{{-{k:.4f}}}")

st.markdown(f"""
<div class="result-highlight">
    t = {t_goal:.2f} minutos
</div>
""", unsafe_allow_html=True)


# ─── 4. GRÁFICOS ───
viz = CoolingVisualizer(solver)
t_max = max(t_goal * 1.4, t2 * 1.5, 30)

st.markdown("---")
st.markdown("## 📈 Curva de enfriamiento")

fig1 = viz.plot_cooling_curve(t_max)
st.pyplot(fig1)
plt.close(fig1)


st.markdown("### Efecto de la constante k en la curva de enfriamiento")
st.markdown("*Comparación de diferentes escenarios de refrigeración con los mismos T₀ y Tₐ:*")

fig2 = viz.plot_k_comparison(t_max)
st.pyplot(fig2)
plt.close(fig2)


st.markdown("### Representación semilogarítmica (linealización)")
st.markdown("*Al graficar ln(T − Tₐ) vs t, la curva exponencial se convierte en una recta con pendiente −k:*")

fig3 = viz.plot_semilog(t_max)
st.pyplot(fig3)
plt.close(fig3)


# ─── 5. PROGRESO TÉRMICO VISUAL ───
if modo_aprendizaje:
    st.markdown("---")
    st.markdown("## 🌡️ Progreso térmico hacia el equilibrio")

    pct_t2 = min(((T0 - T2) / (T0 - Ta)) * 100, 100)
    pct_goal = min(((T0 - Tgoal) / (T0 - Ta)) * 100, 100)

    # Color gradient based on percentage
    def _bar_color(pct: float) -> str:
        if pct < 40:
            return "#e63946"
        elif pct < 70:
            return "#f77f00"
        else:
            return "#2a9d8f"

    st.markdown(f"""
<div class="progress-thermal">
<strong>En t = {t2:.0f} min</strong> — el objeto ha recorrido el <strong>{pct_t2:.1f}%</strong>
del camino de T₀ ({T0:.0f}°C) hacia Tₐ ({Ta:.0f}°C)
<div class="bar-bg">
    <div class="bar-fill" style="width: {pct_t2:.1f}%; background: {_bar_color(pct_t2)};">
        {pct_t2:.0f}%
    </div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    st.markdown(f"""
<div class="progress-thermal">
<strong>Para llegar a T* = {Tgoal:.0f}°C</strong> — se necesita recorrer el
<strong>{pct_goal:.1f}%</strong> del camino, lo que toma <strong>{t_goal:.2f} min</strong>
<div class="bar-bg">
    <div class="bar-fill" style="width: {pct_goal:.1f}%; background: {_bar_color(pct_goal)};">
        {pct_goal:.0f}%
    </div>
</div>
</div>
""", unsafe_allow_html=True)

    five_tau_pct = min((1 - np.exp(-5)) * 100, 100)
    st.markdown("")
    st.markdown(f"""
<div class="progress-thermal">
<strong>Tras 5τ = {5 * tau:.1f} min</strong> — el sistema alcanza el
<strong>{five_tau_pct:.1f}%</strong> del equilibrio (regla práctica de ingeniería)
<div class="bar-bg">
    <div class="bar-fill" style="width: {five_tau_pct:.1f}%; background: #4361ee;">
        {five_tau_pct:.1f}%
    </div>
</div>
</div>
""", unsafe_allow_html=True)


# ─── 6. QUIZ INTERACTIVO ───
if modo_aprendizaje:
    st.markdown("---")
    st.markdown("## 🧠 Ponete a prueba")

    st.markdown("""<div class="quiz-box">
    <strong>Desafío:</strong> Sin cambiar nada, intentá predecir qué pasará
    antes de revelar la respuesta.
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Quiz 1
    with st.expander("Pregunta 1: Si duplicamos k, ¿qué pasa con el tiempo para llegar a T*?"):
        opciones_q1 = [
            "Se duplica (tarda el doble)",
            "Se reduce a la mitad (tarda la mitad)",
            "No cambia",
            "Se reduce, pero no exactamente a la mitad",
        ]
        resp_q1 = st.radio("Tu respuesta:", opciones_q1, key="quiz_q1", index=None)
        if resp_q1 is not None:
            t_goal_2k = solver.time_for_temperature(Tgoal)
            t_goal_2k_real = -np.log((Tgoal - Ta) / (T0 - Ta)) / (k * 2)
            if resp_q1 == opciones_q1[1]:
                st.markdown(f"""<div class="quiz-success">
                <strong>Correcto.</strong> Como t = ln(...)/<strong>k</strong>, al duplicar k
                el tiempo se divide exactamente por 2.
                Con k actual: {t_goal:.2f} min → con 2k: {t_goal_2k_real:.2f} min.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="quiz-fail">
                <strong>No exactamente.</strong> La respuesta correcta es "se reduce a la mitad".
                Como t = ln(...)/<strong>k</strong>, duplicar k divide el tiempo por 2.
                Con k actual: {t_goal:.2f} min → con 2k: {t_goal_2k_real:.2f} min.
                </div>""", unsafe_allow_html=True)

    # Quiz 2
    with st.expander("Pregunta 2: ¿La temperatura del objeto puede bajar por debajo de Tₐ?"):
        opciones_q2 = [
            "Sí, si esperamos suficiente tiempo",
            "No, Tₐ es el límite inferior (asíntota)",
            "Sí, pero solo con refrigeración activa",
        ]
        resp_q2 = st.radio("Tu respuesta:", opciones_q2, key="quiz_q2", index=None)
        if resp_q2 is not None:
            if resp_q2 == opciones_q2[1]:
                st.markdown(f"""<div class="quiz-success">
                <strong>Correcto.</strong> La función e<sup>-kt</sup> nunca llega a cero,
                solo se acerca infinitamente. Por eso T(t) = Tₐ + (algo positivo) siempre
                será mayor que Tₐ = {Ta:.0f}°C. Es una <strong>asíntota horizontal</strong>.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="quiz-fail">
                <strong>No.</strong> Según el modelo, T(t) = Tₐ + (T₀−Tₐ)·e<sup>-kt</sup>.
                Como e<sup>-kt</sup> &gt; 0 siempre, la temperatura nunca baja de Tₐ = {Ta:.0f}°C.
                Es una <strong>asíntota horizontal</strong>.
                </div>""", unsafe_allow_html=True)

    # Quiz 3
    with st.expander(f"Pregunta 3: ¿Cuánto tarda en enfriarse el 50% de la diferencia inicial?"):
        opciones_q3 = [
            f"Exactamente {half_life:.2f} min (la vida media)",
            f"Exactamente {tau:.2f} min (la constante de tiempo τ)",
            f"Exactamente {t_goal:.2f} min",
        ]
        resp_q3 = st.radio("Tu respuesta:", opciones_q3, key="quiz_q3", index=None)
        if resp_q3 is not None:
            if resp_q3 == opciones_q3[0]:
                st.markdown(f"""<div class="quiz-success">
                <strong>Correcto.</strong> La vida media t½ = ln(2)/k = {half_life:.2f} min
                es por definición el tiempo para que (T−Tₐ) se reduzca a la mitad.
                Dato: τ = {tau:.2f} min es cuando se pierde el 63.2% (1 − 1/e).
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="quiz-fail">
                <strong>No.</strong> La respuesta es la vida media: t½ = ln(2)/k = {half_life:.2f} min.
                No confundir con τ = {tau:.2f} min, que es cuando se pierde el 63.2%.
                </div>""", unsafe_allow_html=True)


# ─── 7. ANALOGÍAS DEL MUNDO REAL ───
if modo_aprendizaje:
    st.markdown("---")
    st.markdown("## 🌍 ¿Dónde más aparece esta ecuación?")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        with st.expander("☕ Enfriamiento de café"):
            st.markdown("""
            Una taza de café a 85°C en un cuarto a 22°C sigue **exactamente** esta ley.
            Con k ≈ 0.03 min⁻¹ (taza cerámica sin tapa), la vida media es ~23 min.
            Por eso a los 20 min ya está tibio.
            """)
        with st.expander("🔋 Baterías de autos eléctricos"):
            st.markdown("""
            Tras una carga rápida, la batería alcanza ~52°C.
            Los ingenieros de Tesla usan esta misma EDO para diseñar los
            circuitos de refrigeración líquida y garantizar que la batería
            baje de 35°C antes del siguiente ciclo de carga.
            """)
    with col_a2:
        with st.expander("🏥 Hora de muerte (medicina forense)"):
            st.markdown("""
            Los forenses usan la Ley de Enfriamiento de Newton *al revés*:
            miden la temperatura del cuerpo y la del ambiente para estimar
            cuánto tiempo lleva muerto. Es literalmente esta ecuación
            despejando t.
            """)
        with st.expander("🏗️ Concreto y construcción"):
            st.markdown("""
            El concreto recién vertido genera calor por reacción exotérmica.
            Los ingenieros civiles modelan la disipación de ese calor con
            esta EDO para evitar fisuras por gradientes térmicos.
            """)


# ─── 8. TABLA DE VALORES ───
st.markdown("---")
st.markdown("## 📋 Tabla de valores")

tabla_datos = solver.generate_table(t_max)
df = pd.DataFrame(tabla_datos)
st.dataframe(df, use_container_width=True, hide_index=True)

st.download_button(
    label="Descargar tabla como CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="tabla_enfriamiento.csv",
    mime="text/csv",
)


# ─── 9. ANÁLISIS DE RESULTADOS ───
st.markdown("---")
st.markdown("## 🔍 Análisis de resultados")

pct_cooled = ((T0 - T2) / (T0 - Ta)) * 100
five_tau = 5 * tau
clasif_k, desc_sistema = NewtonCoolingSolver.classify_k(k)

st.markdown(f"""
<div class="analysis-box">

**Constante de enfriamiento:** El valor obtenido de **k = {k:.4f} min⁻¹** indica que
la GPU disipa calor a una tasa {clasif_k},
característica de sistemas con {desc_sistema}.

**Vida media térmica:** La diferencia de temperatura (T − Tₐ) se reduce a la mitad cada
**{half_life:.2f} minutos**. La constante de tiempo es τ = {tau:.2f} min, lo que significa que
tras **5τ = {five_tau:.1f} minutos**, el sistema habrá alcanzado el ~99.3% del equilibrio térmico.

**Temperatura en t = {t2:.0f} min:** El componente alcanza **{T2:.2f} °C**, lo que representa
una reducción del **{pct_cooled:.1f}%** de la diferencia térmica inicial.
{"Esto sugiere un enfriamiento eficiente." if pct_cooled > 70 else "El componente aún retiene calor significativo, lo que debe considerarse para ciclos de trabajo repetitivos."}

**Tiempo para alcanzar {Tgoal:.0f} °C:** Se requieren **{t_goal:.2f} minutos**, confirmando
la naturaleza asintótica del modelo — conforme la temperatura se acerca a Tₐ,
la tasa de enfriamiento disminuye progresivamente (Singh et al., 2024).

**Implicación para diseño:** Un valor mayor de *k* (obtenible con refrigeración líquida o
disipadores de cobre con ventilación forzada) reduciría significativamente el tiempo de enfriamiento,
lo cual es crítico para evitar la degradación térmica y el *thermal throttling* en GPUs de alto rendimiento.

</div>
""", unsafe_allow_html=True)


# ─── 10. ECUACIÓN RESUMEN FINAL ───
st.markdown("---")
st.markdown("## 📌 Solución particular del problema")

st.latex(rf"T(t) = {Ta:.0f} + {diff0:.0f} \cdot e^{{-{k:.4f} \, t}}")

st.markdown(f"""
Con dominio $t \\geq 0$ (minutos) y $T \\in ({Ta:.0f}, {T0:.0f}]$ (°C).
""")


# ─── 11. FOOTER ───
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #adb5bd; font-size: 12px; padding: 16px 0;">
    Universidad Fidélitas · MA-106 Ecuaciones Diferenciales · Ley de Enfriamiento de Newton
</div>
""", unsafe_allow_html=True)
