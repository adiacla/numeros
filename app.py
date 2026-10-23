# pip install tensorflow==2.17.0
# pip install streamlit
# pip install streamlit-drawable-canvas
# pip install pillow pandas

import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model

# Mostrar logo
st.image("logo.gif", width='stretch')

# Título
st.title("🧮 Reconocimiento de números escritos a mano - Alfredo Díaz Claro 2026- UNAB")

# Lista de modelos disponibles
modelos_disponibles = ["numerosD1.keras", "numerosC2.keras", "numerosC3.keras"]

# Función para cargar modelos con caché (más eficiente)
@st.cache_resource
def load_model_from_file(modelo_path):
    if not os.path.exists(modelo_path):
        st.error(f"❌ Archivo no encontrado: {modelo_path}")
        return None
    try:
        modelobien = load_model(modelo_path)
        return modelobien
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo '{modelo_path}': {e}")
        return None

# Cargar los modelos
modelo_d1 = load_model_from_file("numerosD1.keras")
modelo_c2 = load_model_from_file("numerosC2.keras")
modelo_c3 = load_model_from_file("numerosC3.keras")

# Verificar si todos los modelos se cargaron correctamente
if not all([modelo_d1, modelo_c2, modelo_c3]):
    st.stop()

# Lienzo para dibujar
st.subheader("🖌️ Dibuja un número")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=18,             # Un grosor algo mayor ayuda al escalar a 28x28
    stroke_color="#FFFFFF",      # Trazo blanco (MNIST)
    background_color="#000000",  # Fondo negro (MNIST)
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True,
)

# Mostrar mensaje personalizado según la probabilidad
def mostrar_mensaje(probabilidad, modelo_nombre):
    if probabilidad < 0.8:
        return f" ({modelo_nombre}): ❗ No identificado adecuadamente, intenta nuevamente"
    else:
        return f" ({modelo_nombre}): ✅ Alta confianza ({probabilidad:.2f})"

# Botón de predicción
if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # 1. Obtener datos RGBA del canvas
        img_rgba = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
        
        # 2. Pegar sobre un fondo negro sólido usando la máscara Alfa
        bg_negro = Image.new("RGB", img_rgba.size, (0, 0, 0))
        bg_negro.paste(img_rgba, mask=img_rgba.split()[3])
        
        # 3. Redimensionar a 28x28 y convertir a escala de grises
        img_resized = bg_negro.resize((28, 28), Image.Resampling.LANCZOS)
        img_gray = img_resized.convert("L")
        
        # 4. Convertir a array numpy (SIN invertir valores)
        img_array = np.array(img_gray)
        
        # Mostrar la imagen procesada tal como la verá el modelo
        st.image(img_array, caption="Imagen procesada para el modelo (28x28 - Fondo negro, trazo blanco)", use_container_width=False)
        
        # 5. Normalizar datos entre 0 y 1 para Keras/TensorFlow
        img_array = img_array.reshape((1, 28, 28, 1)) / 255.0

        with st.spinner("🔍 Realizando predicciones..."):
            # Modelo D1
            prediction_d1 = modelo_d1.predict(img_array)
            predicted_class_d1 = np.argmax(prediction_d1)
            predicted_probability_d1 = np.max(prediction_d1)

            # Modelo C2
            prediction_c2 = modelo_c2.predict(img_array)
            predicted_class_c2 = np.argmax(prediction_c2)
            predicted_probability_c2 = np.max(prediction_c2)

            # Modelo C3
            prediction_c3 = modelo_c3.predict(img_array)
            predicted_class_c3 = np.argmax(prediction_c3)
            predicted_probability_c3 = np.max(prediction_c3)

        # Mostrar resultados en columnas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🧠 Modelo D1")
            st.write(f"Predicción: {predicted_class_d1}" + mostrar_mensaje(predicted_probability_d1, "D1"))

        with col2:
            st.subheader("🧠 Modelo C2")
            st.write(f"Predicción: {predicted_class_c2}" + mostrar_mensaje(predicted_probability_c2, "C2"))

        with col3:
            st.subheader("🧠 Modelo C3")
            st.write(f"Predicción: {predicted_class_c3}" + mostrar_mensaje(predicted_probability_c3, "C3"))

        # Mostrar resumen en tabla
        resultados = pd.DataFrame({
            "Modelo": ["D1", "C2", "C3"],
            "Predicción": [predicted_class_d1, predicted_class_c2, predicted_class_c3],
            "Confianza": [f"{predicted_probability_d1:.4f}", f"{predicted_probability_c2:.4f}", f"{predicted_probability_c3:.4f}"]
        })

        st.subheader("📊 Resumen de predicciones")
        st.table(resultados)

    else:
        st.warning("⚠️ Por favor, dibuja un número antes de predecir.")




