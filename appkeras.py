import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Configuración de la página
st.set_page_config(page_title="Reconocimiento de Números", layout="centered")

st.title("Reconocimiento de Números Escritos a Mano")
st.write("Dibuja un número del 0 al 9 en el recuadro y presiona el botón **Predecir**.")

# Cargar el modelo en caché para evitar recargarlo en cada interacción
@st.cache_resource
def cargar_modelo():
    # REEMPLAZA 'modelo.h5' POR EL NOMBRE EXACTO DE TU ARCHIVO DE MODELO
    return tf.keras.models.load_model("numerosC2.keras")

# Intentar cargar el modelo
try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar el modelo. Asegúrate de que el archivo del modelo esté en el repositorio. Error: {e}")
    st.stop()

# Componente Canvas para dibujar
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=15,
    stroke_color="#FFFFFF",      # Trazo blanco
    background_color="#000000",  # Fondo negro
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True,      # Habilita la devolución de image_data
)

# Botón para ejecutar la predicción
if st.button("Predecir"):
    if canvas_result is not None and canvas_result.image_data is not None:
        # 1. Convertir la matriz RGBA a escala de grises ('L')
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        img_array = np.array(img)

        # 2. Validar si el canvas está vacío
        if np.max(img_array) == 0:
            st.warning("Por favor, dibuja un número antes de predecir.")
        else:
            # 3. Redimensionar a 28x28 píxeles
            img_resized = img.resize((28, 28))
            img_array_resized = np.array(img_resized)

            # NOTA: Si entrenaste tu modelo con trazo negro y fondo blanco, descomenta esta línea:
            # img_array_resized = 255 - img_array_resized

            # 4. Normalizar (0.0 a 1.0) y formatear a (1, 28, 28, 1)
            input_data = (img_array_resized / 255.0).reshape((1, 28, 28, 1))

            # 5. Mostrar la imagen procesada de 28x28
            st.image(img_resized, caption="Imagen procesada (28x28)", width=140)

            # 6. Predecir
            prediction = modelo.predict(input_data)
            predicted_class = np.argmax(prediction)
            predicted_probability = np.max(prediction)

            # 7. Mostrar resultados
            st.success(f"La predicción es: **{predicted_class}** (Confianza: {predicted_probability:.2%})")
    else:
        st.warning("Por favor, dibuja un número antes de predecir.")
