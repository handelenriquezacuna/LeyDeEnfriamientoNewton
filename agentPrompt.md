# PROMPT PARA CLAUDE CODE — Simulador de Ley de Enfriamiento de Newton
# Copiar y pegar este prompt completo en Claude Code para generar la app

"""
Necesito que construyas una aplicación completa en Streamlit para simular y explicar
paso a paso la Ley de Enfriamiento de Newton aplicada a disipación térmica en GPUs.

## CONTEXTO
Es un proyecto universitario de Ecuaciones Diferenciales (MA-106, Universidad Fidélitas).
La profesora exige:
- Desarrollo matemático con separación de variables (NO métodos numéricos como Euler/RK4)
- Problemas de aplicación propios con datos hipotéticos realistas
- Gráficos y diagramas
- Formato académico

## ESTRUCTURA DE LA APP

### Sidebar (parámetros editables)
- T₀: Temperatura inicial de la GPU (default: 92°C, rango 30-200)
- Tₐ: Temperatura ambiente (default: 24°C, rango 0-50)
- t₁: Tiempo de medición conocido (default: 5 min)
- T(t₁): Temperatura medida en t₁ (default: 68°C)
- t₂: Tiempo a evaluar (default: 15 min)
- T*: Temperatura objetivo (default: 30°C)
- Escenarios predefinidos con selectbox:
  - GPU gaming (T₀=95, Tₐ=28, Tm=72, t₁=4, t₂=12, T*=35)
  - GPU servidor data center (T₀=85, Tₐ=20, Tm=55, t₁=6, t₂=20, T*=25)
  - Batería EV post carga rápida (T₀=52, Tₐ=25, Tm=42, t₁=10, t₂=30, T*=28)
  - CPU laptop uso normal (T₀=78, Tₐ=22, Tm=58, t₁=3, t₂=10, T*=28)

### Contenido principal (en este orden exacto)

1. **Planteamiento del problema**
   - Box estilizado que muestra el problema con los valores actuales
   - Se actualiza dinámicamente al cambiar parámetros

2. **Resultados rápidos**
   - 3 metric cards: (a) k en min⁻¹, (b) T(t₂) en °C, (c) tiempo para T*
   - 2 métricas extra: vida media térmica y constante de tiempo τ

3. **Desarrollo matemático paso a paso** (8 pasos con st.latex)
   - Paso 1: Escribir la EDO → dT/dt = -k(T - Tₐ)
   - Paso 2: Separar variables → dT/(T-Tₐ) = -k dt
   - Paso 3: Integrar ambos miembros → ln|T-Tₐ| = -kt + C
   - Paso 4: Solución general → T(t) = Tₐ + (T₀-Tₐ)·e^(-kt)
   - Paso 5: Verificar condición inicial T(0) = T₀
   - Paso 6(a): Encontrar k usando T(t₁) = Tm (mostrar TODOS los pasos algebraicos)
   - Paso 7(b): Evaluar T(t₂) (mostrar sustitución completa)
   - Paso 8(c): Despejar t cuando T = T* (mostrar todos los pasos con ln)
   
   CADA PASO debe tener:
   - Label coloreado (ej: "PASO 1 — ECUACIÓN DIFERENCIAL")
   - Descripción breve en texto
   - Ecuación en st.latex() con los valores numéricos sustituidos
   - Los resultados finales de (a), (b), (c) resaltados en un box verde

4. **Gráfico 1: Curva de enfriamiento** (matplotlib, figsize 12x6)
   - Curva T(t) en azul con fill_between suave
   - Línea horizontal punteada para Tₐ
   - 4 puntos anotados: T₀, T(t₁), T(t₂), T* con colores distintos
   - Líneas punteadas desde los puntos a los ejes
   - Leyenda, grid suave, spines top/right ocultos

5. **Gráfico 2: Comparación de k** (matplotlib, figsize 12x5)
   - 3 curvas superpuestas: k/2 (ventilación pobre), k actual, 2k (refrigeración líquida)
   - Estilos de línea distintos (dashed, solid, dash-dot)
   - Demuestra visualmente el impacto del tipo de refrigeración

6. **Gráfico 3: Representación semilogarítmica** (matplotlib, figsize 12x4.5)
   - Graficar ln(T - Tₐ) vs t
   - Debe verse como una línea recta (prueba de que el modelo es correcto)
   - Anotar la pendiente = -k

7. **Tabla de valores** con st.dataframe
   - Columnas: t (min), T(t) °C, T-Tₐ °C, % enfriado

8. **Análisis de resultados**
   - Box con fondo amarillo suave
   - Interpreta k (alta/moderada/baja según valor)
   - Vida media y constante de tiempo
   - Implicación para diseño de disipadores
   - Referencia a Singh et al. (2024)

9. **Ecuación resumen final** en st.latex

10. **Footer** con créditos del grupo

## MATEMÁTICAS CORE
```python
k = -np.log((Tm - Ta) / (T0 - Ta)) / t1
T2 = Ta + (T0 - Ta) * np.exp(-k * t2)
t_goal = -np.log((Tgoal - Ta) / (T0 - Ta)) / k
half_life = np.log(2) / k
tau = 1 / k
```

## VALIDACIÓN
- Verificar que T₀ > Tm > Tₐ antes de calcular
- Mostrar st.error si los parámetros son inválidos
- Todos los números mostrados deben estar redondeados (no mostrar floats largos)

## ESTILO
- CSS personalizado con st.markdown(unsafe_allow_html=True)
- Pasos matemáticos con border-left coloreado y fondo suave
- Resultados en boxes verdes resaltados
- Gráficos con fondo #fafafa, sin spines superiores/derecho
- Font profesional, colores: azul #4361ee, rojo #e63946, verde #2a9d8f, morado #7209b7

## DEPENDENCIAS
streamlit, numpy, matplotlib, pandas

## ARCHIVOS A GENERAR
1. app.py (la aplicación completa)
2. requirements.txt
3. README.md con instrucciones de ejecución

Genera todo el código completo, funcional, y listo para ejecutar con `streamlit run app.py`.
"""