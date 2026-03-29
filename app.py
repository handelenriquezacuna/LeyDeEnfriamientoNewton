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

# Paso 2
math_step(
    "Paso 2 — Separación de variables",
    "Llevar T a un lado y t al otro para poder integrar",
    "Se reorganiza la ecuación separando las variables T y t:",
    r"\frac{dT}{T - T_a} = -k \, dt"
)

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


# ─── 5. TABLA DE VALORES ───
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


# ─── 6. ANÁLISIS DE RESULTADOS ───
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


# ─── 7. ECUACIÓN RESUMEN FINAL ───
st.markdown("---")
st.markdown("## 📌 Solución particular del problema")

st.latex(rf"T(t) = {Ta:.0f} + {diff0:.0f} \cdot e^{{-{k:.4f} \, t}}")

st.markdown(f"""
Con dominio $t \\geq 0$ (minutos) y $T \\in ({Ta:.0f}, {T0:.0f}]$ (°C).
""")


# ─── 8. FOOTER ───
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #adb5bd; font-size: 12px; padding: 16px 0;">
    Universidad Fidélitas · MA-106 Ecuaciones Diferenciales · Ley de Enfriamiento de Newton
</div>
""", unsafe_allow_html=True)
