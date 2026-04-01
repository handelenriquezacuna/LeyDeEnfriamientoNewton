# Ley de Enfriamiento de Newton — Simulador Interactivo

Aplicacion de EDO (separacion de variables) para modelar disipacion termica en GPUs.

**Live demo:** [handelenriquezacuna.github.io/LeyDeEnfriamientoNewton](https://handelenriquezacuna.github.io/LeyDeEnfriamientoNewton/)

Universidad Fidelitas · MA-106 Ecuaciones Diferenciales

---

## Inicio rapido

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`. Tambien corre en GitHub Pages via [stlite](https://github.com/whitphx/stlite) (WebAssembly, sin servidor).

---

## Arquitectura del sistema

```mermaid
graph TB
    subgraph Datos ["Capa de datos"]
        M[models.py<br><i>CoolingParameters</i><br><i>CoolingResults</i>]
        S[scenarios.py<br><i>ESCENARIOS dict</i>]
    end

    subgraph Logica ["Capa de logica"]
        SV[solver.py<br><i>NewtonCoolingSolver</i>]
    end

    subgraph Presentacion ["Capa de presentacion"]
        CH[charts.py<br><i>CoolingVisualizer</i>]
        APP[app.py<br><i>Streamlit UI</i>]
    end

    subgraph Deploy ["Deployment"]
        IDX[index.html<br><i>stlite loader</i>]
        GH[GitHub Pages<br><i>WebAssembly</i>]
        LOCAL[localhost:8501<br><i>Streamlit server</i>]
    end

    M --> SV
    S --> APP
    SV --> CH
    SV --> APP
    CH --> APP
    APP --> IDX
    IDX --> GH
    APP --> LOCAL

    style Datos fill:#e0e7ff,stroke:#4361ee,color:#1a1a2e
    style Logica fill:#d1fae5,stroke:#2a9d8f,color:#1a1a2e
    style Presentacion fill:#fffbeb,stroke:#f77f00,color:#1a1a2e
    style Deploy fill:#f8f9fa,stroke:#adb5bd,color:#1a1a2e
```

### Modulos

| Archivo | Clase / Contenido | Responsabilidad |
|---------|-------------------|-----------------|
| `models.py` | `CoolingParameters`, `CoolingResults` | Dataclasses con validacion y propiedades derivadas |
| `solver.py` | `NewtonCoolingSolver` | Calculo de k, T(t), t(T), vida media, tabla, curva |
| `scenarios.py` | `ESCENARIOS` | 4 escenarios predefinidos |
| `charts.py` | `CoolingVisualizer` | 3 graficos matplotlib |
| `app.py` | Streamlit UI | Sidebar, LaTeX, metricas, graficos |
| `index.html` | stlite mount | Entry point para GitHub Pages |

---

## Flujo de datos de la aplicacion

```mermaid
flowchart LR
    U["Usuario<br>(sidebar)"] -->|T0, Ta, t1, Tm, t2, Tgoal| P[CoolingParameters]
    P -->|validate| V{Valido?}
    V -->|No| ERR[st.error + st.stop]
    V -->|Si| SOL[NewtonCoolingSolver]
    SOL -->|solve| R[CoolingResults<br>k, T2, t_goal,<br>half_life, tau]
    SOL --> VIZ[CoolingVisualizer]
    VIZ --> F1["Curva de<br>enfriamiento"]
    VIZ --> F2["Comparacion<br>de k"]
    VIZ --> F3["Grafico<br>semilog"]
    R --> MET["5 Metricas"]
    R --> LATEX["8 Pasos<br>LaTeX"]
    SOL -->|generate_table| TBL["Tabla +<br>CSV download"]
    R --> ANA["Analisis de<br>resultados"]

    style P fill:#e0e7ff,stroke:#4361ee,color:#1a1a2e
    style SOL fill:#d1fae5,stroke:#2a9d8f,color:#1a1a2e
    style R fill:#d1fae5,stroke:#2a9d8f,color:#1a1a2e
    style VIZ fill:#fffbeb,stroke:#f77f00,color:#1a1a2e
    style ERR fill:#fee2e2,stroke:#e63946,color:#1a1a2e
```

---

## Flujo de renderizado de la UI

```mermaid
flowchart TB
    subgraph Sidebar
        PRESET["Selectbox<br>escenarios"] --> CB["on_change<br>callback"]
        CB --> SS["session_state<br>(unica fuente<br>de verdad)"]
        RESET["Boton<br>restablecer"] --> SS
        SS --> INPUTS["6 number_input<br>T0, Ta, t1, Tm, t2, Tgoal"]
    end

    subgraph Main ["Contenido principal (scroll)"]
        direction TB
        H["Titulo + subtitulo"]
        PROB["Planteamiento<br>del problema"]
        METRICS["5 metric cards<br>k | T2 | t_goal | half_life | tau"]
        STEPS["8 pasos LaTeX<br>EDO → separacion → integracion →<br>solucion → C.I. → k → T(t2) → t*"]
        CHART1["Curva de enfriamiento<br>(4 puntos anotados)"]
        CHART2["Comparacion k<br>(k/2, k, 2k)"]
        CHART3["Semilog<br>(linealización)"]
        TABLE["Tabla de valores<br>+ descarga CSV"]
        ANALYSIS["Analisis de<br>resultados"]
        EQ["Ecuacion final<br>particular"]
        FOOTER["Footer<br>Universidad Fidelitas"]

        H --> PROB --> METRICS --> STEPS --> CHART1 --> CHART2 --> CHART3 --> TABLE --> ANALYSIS --> EQ --> FOOTER
    end

    INPUTS --> Main

    style Sidebar fill:#f8f9fc,stroke:#dee2e6,color:#1a1a2e
    style Main fill:#ffffff,stroke:#dee2e6,color:#1a1a2e
```

---

## Pipeline CI/CD

```mermaid
flowchart LR
    PUSH["git push<br>main"] --> TEST["Job: test<br>ubuntu + Python 3.12"]
    TEST -->|pip install| DEPS[requirements.txt]
    DEPS --> PYTEST["pytest tests/ -v<br>156 tests"]
    PYTEST -->|pass| DEPLOY["Job: deploy"]
    PYTEST -->|fail| STOP["Pipeline<br>detenido"]
    DEPLOY --> PAGES["configure-pages<br>enablement: true"]
    PAGES --> UPLOAD["upload-pages-artifact<br>_site/"]
    UPLOAD --> LIVE["deploy-pages<br>github.io"]

    style PUSH fill:#e0e7ff,stroke:#4361ee,color:#1a1a2e
    style PYTEST fill:#d1fae5,stroke:#2a9d8f,color:#1a1a2e
    style STOP fill:#fee2e2,stroke:#e63946,color:#1a1a2e
    style LIVE fill:#d1fae5,stroke:#2a9d8f,color:#1a1a2e
```

---

## Diagrama de clases

```mermaid
classDiagram
    class CoolingParameters {
        +float T0
        +float Ta
        +float t1
        +float Tm
        +float t2
        +float Tgoal
        +initial_difference() float
        +measured_difference() float
        +goal_difference() float
        +measurement_ratio() float
        +validate() tuple~bool, str~
    }

    class CoolingResults {
        +float k
        +float T2
        +float t_goal
        +float half_life
        +float tau
    }

    class NewtonCoolingSolver {
        +CoolingParameters params
        +k() float
        +temperature_at(t) float
        +time_for_temperature(T) float
        +half_life() float
        +time_constant() float
        +solve() CoolingResults
        +generate_curve(t_max) tuple
        +generate_table(t_max) list
        +classify_k(k) tuple$
    }

    class CoolingVisualizer {
        +NewtonCoolingSolver solver
        +CoolingParameters params
        +CoolingResults results
        +plot_cooling_curve(t_max) Figure
        +plot_k_comparison(t_max) Figure
        +plot_semilog(t_max) Figure
    }

    CoolingParameters --> NewtonCoolingSolver : params
    NewtonCoolingSolver --> CoolingResults : solve()
    NewtonCoolingSolver --> CoolingVisualizer : solver
    CoolingVisualizer --> CoolingResults : results
```

---

## Que resuelve

Dada la EDO `dT/dt = -k(T - Ta)` con solucion analitica `T(t) = Ta + (T0 - Ta) * e^(-kt)`:

1. **(a)** Calcula la constante de enfriamiento `k` a partir de un dato conocido
2. **(b)** Evalua la temperatura en cualquier instante `t`
3. **(c)** Determina el tiempo para alcanzar una temperatura objetivo

Todo resuelto por **separacion de variables** (no metodos numericos).

---

## Funcionalidades

- Sidebar con parametros editables y 4 escenarios predefinidos
- Desarrollo matematico completo paso a paso (8 pasos con LaTeX)
- 3 graficos: curva de enfriamiento, comparacion de k, semilogaritmico
- Tabla de valores con descarga CSV
- Analisis de resultados automatico
- Boton restablecer valores

---

## Tests

```bash
python -m pytest tests/ -v
```

156 tests en 4 modulos:

| Modulo | Tests | Cobertura |
|--------|-------|-----------|
| `test_models.py` | 18 | Validacion de parametros, propiedades derivadas |
| `test_solver.py` | 117 | Calculo de k, T(t), t(T), escenarios, edge cases |
| `test_charts.py` | 8 | Generacion de figuras matplotlib |
| `test_app_visual.py` | 13 | Integracion Streamlit (carga, metricas, LaTeX) |

---

## Estructura de archivos

```
LeyDeEnfriamientoNewton/
├── .github/workflows/
│   └── deploy.yml              CI: test + deploy a Pages
├── .streamlit/
│   └── config.toml             Tema light forzado
├── tests/
│   ├── test_models.py          18 tests
│   ├── test_solver.py          117 tests
│   ├── test_charts.py          8 tests
│   └── test_app_visual.py      13 tests
├── models.py                   Dataclasses
├── solver.py                   Logica matematica
├── scenarios.py                Escenarios predefinidos
├── charts.py                   Visualizacion
├── app.py                      Streamlit UI
├── index.html                  stlite (GitHub Pages)
└── requirements.txt            Dependencias
```

---

## Stack

- Python 3.12+
- Streamlit
- NumPy, Matplotlib, Pandas
- stlite (deployment WebAssembly)
- GitHub Actions (CI/CD)
