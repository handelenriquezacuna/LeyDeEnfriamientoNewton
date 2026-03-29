"""
Tests visuales / de integración para la app de Streamlit.
Ley de Enfriamiento de Newton — Simulador Interactivo.

Utiliza streamlit.testing.v1.AppTest para verificar que la app
renderiza correctamente todos sus componentes.
"""

import sys
import os
import pytest

# Agregar el directorio raíz del proyecto al path para que los imports funcionen
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)

from streamlit.testing.v1 import AppTest

# Ruta al archivo principal de la app (relativa al PROJECT_DIR)
APP_FILE = os.path.join(PROJECT_DIR, "app.py")


# ─── Fixtures ───

@pytest.fixture
def app():
    """Fixture que carga la app con parámetros por defecto y la ejecuta."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    return at


# ─── 1. Test carga inicial ───

def test_carga_inicial(app):
    """Verifica que la app carga sin excepciones con los parámetros por defecto."""
    assert not app.exception, (
        f"La app lanzó excepciones al cargar: {app.exception}"
    )


# ─── 2. Test título ───

def test_titulo_contiene_ley_de_enfriamiento(app):
    """Verifica que el título principal contiene 'Ley de Enfriamiento de Newton'."""
    # El título se renderiza con st.markdown("# ..."), no con st.title,
    # así que buscamos en los elementos markdown.
    textos_markdown = [m.value for m in app.markdown]
    texto_completo = " ".join(textos_markdown)
    assert "Ley de Enfriamiento de Newton" in texto_completo, (
        "No se encontró 'Ley de Enfriamiento de Newton' en los elementos markdown de la app."
    )


# ─── 3. Test métricas renderizadas ───

def test_metricas_renderizadas(app):
    """Verifica que se muestran exactamente 5 métricas (k, T2, t_goal, half_life, tau)."""
    cantidad_metricas = len(app.metric)
    assert cantidad_metricas == 5, (
        f"Se esperaban 5 métricas, pero se encontraron {cantidad_metricas}. "
        f"Valores: {[m.label for m in app.metric]}"
    )

    # Verificar que las etiquetas contienen las palabras clave esperadas
    etiquetas = [m.label for m in app.metric]
    etiquetas_texto = " ".join(etiquetas)

    assert "k" in etiquetas_texto, "No se encontró la métrica de la constante k"
    assert "Vida media" in etiquetas_texto, "No se encontró la métrica de vida media térmica"
    assert "τ" in etiquetas_texto or "tau" in etiquetas_texto.lower(), (
        "No se encontró la métrica de constante de tiempo tau"
    )


# ─── 4. Test LaTeX renderizado ───

def test_latex_renderizado(app):
    """Verifica que hay ecuaciones LaTeX (al menos 10 llamadas a st.latex)."""
    cantidad_latex = len(app.latex)
    assert cantidad_latex >= 10, (
        f"Se esperaban al menos 10 elementos LaTeX, pero se encontraron {cantidad_latex}. "
        "La app debería mostrar el desarrollo matemático completo paso a paso."
    )


# ─── 5. Test tabla de datos ───

def test_tabla_de_datos(app):
    """Verifica que hay un dataframe con las columnas correctas."""
    assert len(app.dataframe) >= 1, (
        "No se encontró ningún dataframe en la app."
    )

    # Obtener el primer dataframe renderizado
    df = app.dataframe[0].value

    columnas_esperadas = ["t (min)", "T(t) °C", "T − Tₐ (°C)", "% enfriado"]
    for col in columnas_esperadas:
        assert col in df.columns, (
            f"La columna '{col}' no se encontró en el dataframe. "
            f"Columnas presentes: {list(df.columns)}"
        )

    # Verificar que la tabla tiene filas de datos
    assert len(df) > 0, "El dataframe está vacío, debería tener filas de datos."


# ─── 6. Test parámetros inválidos ───

def test_parametros_invalidos_tm_mayor_que_t0():
    """Cambiar Tm > T0 y verificar que se muestra un mensaje de error."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()

    # Parámetros por defecto: T0=92, Tm=68
    # Cambiar Tm a un valor mayor que T0 para provocar error de validación.
    # Tm es el cuarto number_input del sidebar (índice 3).
    # Orden en sidebar: T0 (0), Ta (1), t1 (2), Tm (3), t2 (4), Tgoal (5)
    at.number_input(key="input_Tm").set_value(150.0).run()

    # Verificar que no hay excepción no controlada
    assert not at.exception, (
        f"La app lanzó una excepción no controlada: {at.exception}"
    )

    # Verificar que se muestra el mensaje de error
    assert len(at.error) > 0, (
        "No se mostró ningún mensaje de error cuando Tm > T0. "
        "La app debería mostrar st.error cuando los parámetros son inválidos."
    )

    # Verificar el contenido del error
    texto_error = at.error[0].value
    assert "inválidos" in texto_error.lower() or "Parámetros" in texto_error, (
        f"El mensaje de error no contiene el texto esperado. Mensaje: {texto_error}"
    )


# ─── 7. Test cambio de parámetros ───

def test_cambio_de_parametros():
    """Cambiar T0 a 95 y verificar que los resultados cambian respecto a los valores por defecto."""
    # Ejecutar con valores por defecto
    at_default = AppTest.from_file(APP_FILE, default_timeout=30)
    at_default.run()

    # Guardar valores de métricas por defecto
    metricas_default = [m.value for m in at_default.metric]

    # Ejecutar con T0 cambiado a 95
    at_modificado = AppTest.from_file(APP_FILE, default_timeout=30)
    at_modificado.run()
    at_modificado.number_input(key="input_T0").set_value(95.0).run()

    assert not at_modificado.exception, (
        f"La app lanzó una excepción tras cambiar T0: {at_modificado.exception}"
    )

    # Verificar que las métricas cambiaron
    metricas_modificadas = [m.value for m in at_modificado.metric]
    assert metricas_default != metricas_modificadas, (
        "Las métricas no cambiaron al modificar T0 de 92 a 95. "
        f"Default: {metricas_default}, Modificado: {metricas_modificadas}"
    )


# ─── 8. Test escenarios predefinidos ───

# Importar los escenarios para parametrizar el test
from scenarios import ESCENARIOS

@pytest.mark.parametrize("nombre_escenario", list(ESCENARIOS.keys()))
def test_escenarios_predefinidos(nombre_escenario):
    """Seleccionar cada escenario predefinido y verificar que carga sin error."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()

    # Seleccionar el escenario en el selectbox del sidebar
    at.selectbox(key="preset_select").set_value(nombre_escenario).run()

    assert not at.exception, (
        f"La app lanzó una excepción al cargar el escenario '{nombre_escenario}': "
        f"{at.exception}"
    )

    # Verificar que no se muestra error de validación
    assert len(at.error) == 0, (
        f"Se mostró un error al cargar el escenario '{nombre_escenario}': "
        f"{at.error[0].value if at.error else 'N/A'}"
    )

    # Verificar que las métricas se renderizan
    assert len(at.metric) == 5, (
        f"El escenario '{nombre_escenario}' no renderizó las 5 métricas esperadas. "
        f"Se encontraron {len(at.metric)}."
    )


# ─── 9. Test gráficos ───

def test_graficos_renderizados(app):
    """Verificar que se renderizan figuras matplotlib (sin excepciones tras run)."""
    # Streamlit AppTest no expone directamente st.pyplot como un elemento accesible,
    # pero podemos verificar que la app se ejecutó completamente sin errores,
    # lo cual implica que los 3 gráficos (curva, comparación k, semilog) se generaron.
    assert not app.exception, (
        f"La app lanzó excepciones al renderizar gráficos: {app.exception}"
    )

    # Verificar que los encabezados de las secciones de gráficos están presentes
    textos_markdown = [m.value for m in app.markdown]
    texto_completo = " ".join(textos_markdown)

    assert "Curva de enfriamiento" in texto_completo, (
        "No se encontró la sección 'Curva de enfriamiento' en la app."
    )
    assert "Efecto de la constante k" in texto_completo, (
        "No se encontró la sección de comparación de k en la app."
    )
    assert "semilogarítmica" in texto_completo or "Linealización" in texto_completo.lower() or "linealización" in texto_completo, (
        "No se encontró la sección de representación semilogarítmica en la app."
    )


# ─── 10. Test footer ───

def test_footer_universidad_fidelitas(app):
    """Verificar que el footer con 'Universidad Fidélitas' aparece en la app."""
    textos_markdown = [m.value for m in app.markdown]
    texto_completo = " ".join(textos_markdown)

    assert "Universidad Fidélitas" in texto_completo, (
        "No se encontró 'Universidad Fidélitas' en el footer de la app. "
        "Textos markdown encontrados: " + str(textos_markdown[-3:])
    )
