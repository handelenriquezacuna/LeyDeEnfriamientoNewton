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

## Arquitectura

```
models.py        CoolingParameters, CoolingResults (dataclasses + validacion)
solver.py        NewtonCoolingSolver (k cacheado, T(t), t(T), tabla, curva)
scenarios.py     4 escenarios predefinidos (GPU gaming, servidor, EV, laptop)
charts.py        CoolingVisualizer (3 graficos matplotlib)
app.py           Streamlit UI (sidebar, LaTeX paso a paso, metricas)
index.html       stlite loader para GitHub Pages
```

### Flujo de datos

```
CoolingParameters → NewtonCoolingSolver → CoolingResults
                          ↓
                   CoolingVisualizer → Figures
                          ↓
                       app.py → Streamlit UI
```

## Que resuelve

Dada la EDO `dT/dt = -k(T - Ta)` con solucion analitica `T(t) = Ta + (T0 - Ta) * e^(-kt)`:

1. **(a)** Calcula la constante de enfriamiento `k` a partir de un dato conocido
2. **(b)** Evalua la temperatura en cualquier instante `t`
3. **(c)** Determina el tiempo para alcanzar una temperatura objetivo

Todo resuelto por **separacion de variables** (no metodos numericos).

## Funcionalidades

- Sidebar con parametros editables y 4 escenarios predefinidos
- Desarrollo matematico completo paso a paso (8 pasos con LaTeX)
- 3 graficos: curva de enfriamiento, comparacion de k, semilogaritmico
- Tabla de valores con descarga CSV
- Analisis de resultados automatico
- Boton restablecer valores

## Tests

```bash
python -m pytest tests/ -v
```

156 tests en 4 modulos:

| Modulo | Tests | Cobertura |
|--------|-------|-----------|
| test_models.py | 18 | Validacion de parametros, propiedades derivadas |
| test_solver.py | 117 | Calculo de k, T(t), t(T), escenarios, edge cases |
| test_charts.py | 8 | Generacion de figuras matplotlib |
| test_app_visual.py | 13 | Integracion Streamlit (carga, metricas, LaTeX) |

## CI/CD

Push a `main` ejecuta automaticamente:
1. `python -m pytest tests/ -v` (ubuntu, Python 3.12)
2. Deploy a GitHub Pages si los tests pasan

## Stack

- Python 3.12+
- Streamlit
- NumPy, Matplotlib, Pandas
- stlite (deployment WebAssembly)
