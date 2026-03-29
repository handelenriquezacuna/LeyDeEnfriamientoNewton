# 🌡️ Ley de Enfriamiento de Newton — Simulador Interactivo

**Aplicación de EDO: Disipación Térmica en GPU**  
Universidad Fidélitas · MA-106 Ecuaciones Diferenciales · Grupo 3

## Inicio rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar la aplicación
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`

## ¿Qué incluye?

- **Parámetros editables** en el sidebar (temperaturas, tiempos, temperatura objetivo)
- **Escenarios predefinidos** (GPU gaming, servidor, batería EV, CPU laptop)
- **Desarrollo matemático paso a paso** con LaTeX renderizado
- **3 gráficos profesionales:**
  - Curva de enfriamiento con puntos clave anotados
  - Comparación del efecto de k (ventilación pobre vs actual vs refrigeración líquida)
  - Representación semilogarítmica (linealización para verificar el modelo)
- **Tabla de valores** exportable
- **Análisis de resultados** que se actualiza automáticamente

## Para la presentación

1. Ejecutar `streamlit run app.py`
2. Mostrar el problema con los valores por defecto (GPU a 92°C)
3. Cambiar valores en vivo para demostrar el impacto
4. Usar los escenarios predefinidos para comparar contextos
5. El gráfico semilogarítmico demuestra que el modelo es correcto (línea recta)
