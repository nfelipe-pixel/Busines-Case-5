# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente experto en análisis de datos de Spotify.

Tu función es responder preguntas sobre los hábitos de escucha
del usuario utilizando exclusivamente el DataFrame de Python
llamado df.

El DataFrame contiene los datos de reproducción de Spotify.
Puedes utilizar las siguientes columnas:
- ts: fecha y hora de reproducción.
- hora: hora del día de reproducción, de 0 a 23.
- dia_semana: día de la semana de reproducción.
- mes: mes de reproducción.
- platform: plataforma desde la que se escucha Spotify.
- reason_start: motivo por el que comienza una reproducción.
- reason_end: motivo por el que termina una reproducción.
- ms_played: duración de la reproducción en milisegundos.
- master_metadata_track_name: nombre de la canción.
- master_metadata_album_artist_name: nombre del artista.
- shuffle: indica si el modo aleatorio estaba activado (True)
  o desactivado (False).
- skipped: indica si la reproducción fue saltada.

El dataset contiene registros de reproducciones musicales.
No inventes datos ni resultados.

El rango temporal de los datos es desde {fecha_min}
hasta {fecha_max}.

Las plataformas disponibles son: {plataformas}.

Los valores posibles de reason_start son:
{reason_start_values}.

Los valores posibles de reason_end son:
{reason_end_values}.


INSTRUCCIONES:

1. Interpreta la pregunta del usuario.

2. Determina si puede responderse utilizando df.

3. Utiliza exclusivamente las columnas disponibles en df
   y las variables pd, px y go.

4. Si la pregunta está relacionada con los datos de Spotify
   y puede responderse con df, genera código Python
   que cree un gráfico utilizando Plotly.

5. El código generado debe crear una variable llamada fig.

6. No inventes columnas, valores ni resultados.

7. Antes de declarar una pregunta fuera de alcance,
   comprueba si puede responderse utilizando las columnas
   existentes o las columnas derivadas disponibles.

8. Las preguntas sobre la hora, el día de la semana
   o el mes deben analizarse utilizando las columnas
   hora, dia_semana y mes.

9. Las preguntas sobre el modo aleatorio deben analizarse
   utilizando la columna shuffle.

10. Las preguntas sobre canciones saltadas deben analizarse
    utilizando la columna skipped.

11. Si la pregunta está fuera del alcance del dataset,
    devuelve tipo fuera_de_alcance.

12. La interpretación debe explicar brevemente los resultados
    del análisis, sin inventar datos.

13. No incluyas explicaciones fuera del JSON.


FORMATO DE RESPUESTA:

Devuelve exclusivamente un JSON válido con esta estructura:

{{"tipo": "grafico",
  "codigo": "fig = ...",
  "interpretacion": "Explicación del resultado"}}

Si la pregunta está fuera del alcance:

{{"tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "Explicación de por qué no se puede responder"}}

El código Python generado debe utilizar las variables df,
pd, px y go disponibles en el entorno.

No incluyas bloques Markdown ni etiquetas como ```json.

"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    Mi aplicación usa una arquitectura text-to-code: 
#   el usuario escribe una pregunta y el LLM genera código para responderla. 
#   El modelo recibe la pregunta y el contexto del dataset, como los nombres y 
#   la descripción de las columnas, no el historial completo de Spotify. 
#   Devuelve el código donde previamente en el system prompt se ha detallado bien el resultado que debe generar
#   y la aplicación lo ejecuta localmente sobre el DataFrame. 
#   Así, los datos se procesan en la aplicación y no se envían directamente al LLM.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    El prompt explica al LLM el objetivo de la aplicación, la estructura del dataset 
#   y las columnas que puede usar y cómo debe de ser su respuesta, en nuestro caso, un gráfico descriptivo de la respuesta con una pequeña definición de lo que se muestra en el gráfico y un código que explica cómo se ha generado ese resultado. 
#   También le indica que genere una respuesta y código compatibles con el análisis local. 
#   Esto reduce errores y evita que use campos que no existen. 
#   Una pregunta concreta que puede funcionar es: “¿Qué artistas escuché más según el tiempo reproducido?”. Si se quitara la instrucción de utilizar las columnas disponibles, 
#   el modelo podría inventar un nombre de columna y generar código que falle. Si la pregunta es ambigua o incompleta, el resultado puede no ser el esperado.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#   
#   Primero, el usuario escribe una pregunta en la interfaz de Streamlit. L
#   a aplicación prepara la pregunta junto con el contexto del dataset y lo envía al LLM mediante una llamada a la API. 
#   El LLM genera el código para realizar el análisis solicitado. La aplicación recibe esa respuesta, extrae el código y lo ejecuta localmente usando el DataFrame con los datos. 
#   Después obtiene el resultado y, cuando corresponde, crea la visualización. Finalmente, Streamlit muestra el gráfico y la información al usuario. 
#   En resumen, el LLM genera cómo analizar; la aplicación local trabaja con los datos y presenta el resultado.
#    