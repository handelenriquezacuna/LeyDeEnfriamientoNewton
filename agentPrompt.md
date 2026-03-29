# Agent Prompt — Newton's Law of Cooling Simulator

> Universal prompt for any AI coding agent (Claude Code, Cursor, Copilot, Windsurf, Aider, etc.)
> Copy-paste this entire file as context when working on this project.

---

## 1. PROJECT IDENTITY

| Field | Value |
|-------|-------|
| Name | Ley de Enfriamiento de Newton — Simulador Interactivo |
| Purpose | Interactive Streamlit app that solves and visualizes Newton's Law of Cooling as an ODE problem |
| Domain | University differential equations course (MA-106, Universidad Fidelitas) |
| Language | Python 3.12+ |
| UI Framework | Streamlit |
| Deployment | GitHub Pages via stlite (WebAssembly) + local `streamlit run app.py` |
| Live URL | https://handelenriquezacuna.github.io/LeyDeEnfriamientoNewton/ |
| Repo | https://github.com/handelenriquezacuna/LeyDeEnfriamientoNewton |

---

## 2. ARCHITECTURE

```
models.py        → CoolingParameters (dataclass + validate())
                   CoolingResults (dataclass)

solver.py        → NewtonCoolingSolver
                     .k (cached property)
                     .temperature_at(t) → float
                     .time_for_temperature(T) → float
                     .half_life, .time_constant (properties)
                     .solve() → CoolingResults
                     .generate_curve(t_max) → (t_arr, T_arr)
                     .generate_table(t_max) → list[dict]
                     .classify_k(k) → (str, str)  [static]

scenarios.py     → ESCENARIOS: dict[str, CoolingParameters]
                     "GPU gaming (alta carga)"
                     "GPU servidor (data center)"
                     "Batería EV (post carga rápida)"
                     "CPU laptop (uso normal)"

charts.py        → CoolingVisualizer(solver)
                     .plot_cooling_curve(t_max) → Figure
                     .plot_k_comparison(t_max) → Figure
                     .plot_semilog(t_max) → Figure

app.py           → Streamlit UI (thin layer)
                     Sidebar: inputs, presets, reset button
                     Main: problem statement, metrics, 8 LaTeX steps,
                           3 charts, data table, analysis, final equation

index.html       → stlite mountable loader (GitHub Pages entry point)
```

### Data flow

```
User inputs (sidebar)
    → CoolingParameters
    → params.validate()
    → NewtonCoolingSolver(params)
    → solver.solve() → CoolingResults
    → CoolingVisualizer(solver) → 3 Figures
    → Streamlit renders everything
```

---

## 3. MATH CORE

The app solves the ODE `dT/dt = -k(T - Ta)` via separation of variables.

**Analytical solution:** `T(t) = Ta + (T0 - Ta) * e^(-kt)`

```python
k = -ln((Tm - Ta) / (T0 - Ta)) / t1      # from known measurement
T2 = Ta + (T0 - Ta) * exp(-k * t2)         # temperature at time t2
t_goal = -ln((Tgoal - Ta) / (T0 - Ta)) / k # time to reach target
half_life = ln(2) / k                       # thermal half-life
tau = 1 / k                                 # time constant
```

**Constraint:** The professor requires separation of variables (NOT numerical methods like Euler/RK4).

---

## 4. HARD CONSTRAINTS

These rules must never be violated:

| # | Constraint |
|---|-----------|
| 1 | **Dual deployment**: Code must work in BOTH local Streamlit AND stlite/Pyodide (WebAssembly). Test API compatibility before shipping. |
| 2 | **session_state is single source of truth** for all widget values. Never pass `value=` to a `number_input` that also has a `key=` in session_state. Initialize defaults via `if key not in st.session_state: st.session_state[key] = default`. |
| 3 | **Use `use_container_width=True`** instead of `width="stretch"` for `st.dataframe` (stlite compatibility). |
| 4 | **All user-facing text in Spanish.** Code identifiers in English. |
| 5 | **156 tests must pass** before any push. Run `python -m pytest tests/ -v`. |
| 6 | **Separation of variables only** — no numerical ODE solvers. |
| 7 | **OOP architecture** — logic in solver.py, visualization in charts.py, data in models.py. app.py is a thin UI layer. |
| 8 | **CSS contrast** — all custom-styled elements must have explicit `color` properties with `!important` for dark mode safety. |

---

## 5. TEST SUITE

```bash
python -m pytest tests/ -v    # 156 tests, ~6 seconds
```

| File | Count | What it tests |
|------|-------|---------------|
| tests/test_models.py | 18 | CoolingParameters validation (13 cases), properties (4), CoolingResults construction (1) |
| tests/test_solver.py | 117 | k calculation, T(t), t(T), half-life, tau, solve() across 4 scenarios, table/curve generation, classify_k, edge cases |
| tests/test_charts.py | 8 | CoolingVisualizer returns valid matplotlib Figures with correct axes/labels |
| tests/test_app_visual.py | 13 | Streamlit integration: loads without error, renders 5 metrics, 10+ LaTeX blocks, dataframe, presets work |

---

## 6. CI/CD

GitHub Actions workflow (`.github/workflows/deploy.yml`):
1. On push to `main`: run tests on ubuntu with Python 3.12
2. If tests pass: deploy to GitHub Pages via `actions/deploy-pages@v4`
3. `enablement: true` on `configure-pages` to auto-create the Pages site

---

## 7. DEPENDENCIES

```
streamlit>=1.30.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
pytest>=8.0.0
```

---

## 8. COMMON TASKS

### Add a new scenario
1. Add entry to `ESCENARIOS` dict in `scenarios.py`
2. Tests auto-parametrize over `ESCENARIOS` — new scenario gets tested automatically

### Add a new chart
1. Add method to `CoolingVisualizer` in `charts.py` returning a `Figure`
2. Call it in `app.py` with `st.pyplot(viz.new_method(t_max))` + `plt.close()`
3. Add test in `tests/test_charts.py`

### Modify solver logic
1. Edit `NewtonCoolingSolver` in `solver.py`
2. Update/add tests in `tests/test_solver.py`
3. Run `python -m pytest tests/ -v` before committing

### Deploy
```bash
git push origin main   # triggers CI: test → deploy to Pages automatically
```

---

## 9. KNOWN LIMITATIONS

- stlite initial load takes 10-15 seconds (downloads Pyodide/WebAssembly)
- Streamlit's built-in `min_value`/`max_value` error messages are in English (cannot be overridden)
- `st.download_button` works locally but may have browser-specific behavior in stlite
- Matplotlib renders as static images (no hover/zoom — would need Plotly for interactivity)
