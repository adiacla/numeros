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
st.title("🧮 Reconocimiento de números escritos a mano - Alfredo Díaz Claro 2025- UNAB")

# Inicializar clave del lienzo en session_state para poder reiniciarlo
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Función para cargar modelos con caché
@st.cache_resource
def load_model_from_file(modelo_path):
    if not os.path.exists(modelo_path):
        st.error(f"❌ Archivo no encontrado: {modelo_path}")
        return None
    try:
        return load_model(modelo_path)
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo '{modelo_path}': {e}")
        return None

# Cargar los modelos
modelo_d1 = load_model_from_file("numerosD1.keras")
modelo_c2 = load_model_from_file("numerosC2.keras")
modelo_c3 = load_model_from_file("numerosC3.keras")

if not all([modelo_d1, modelo_c2, modelo_c3]):
    st.stop()

# Lienzo para dibujar
st.subheader("🖌️ Dibuja un número")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=18,
    stroke_color="#FFFFFF",      # Trazo blanco (MNIST)
    background_color="#000000",  # Fondo negro (MNIST)
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",  # Key dinámica
    return_image_data=True,
)

# Mensaje según la probabilidad
def mostrar_mensaje(probabilidad, modelo_nombre):
    if probabilidad < 0.8:
        return f" ({modelo_nombre}): ❗ No identificado adecuadamente, intenta nuevamente"
    else:
        return f" ({modelo_nombre}): ✅ Alta confianza ({probabilidad:.2f})"

# Botones de control
col_pred, col_clear = st.columns(2)

with col_clear:
    if st.button("🗑️ Limpiar lienzo", use_container_width=True):
        st.session_state.canvas_key += 1
        st.rerun()

with col_pred:
    prediccion_solicitada = st.button("🔍 Predecir", type="primary", use_container_width=True)

# Lógica de predicción
if prediccion_solicitada:
    if canvas_result.image_data is not None and canvas_result.image_data.any():
        # 1. Obtener datos RGBA del canvas
        img_rgba = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
        
        # 2. Fondo negro sólido para evitar transparencia
        bg_negro = Image.new("RGB", img_rgba.size, (0, 0, 0))
        bg_negro.paste(img_rgba, mask=img_rgba.split()[3])
        
        # 3. Redimensionar a 28x28 y convertir a escala de grises
        img_resized = bg_negro.resize((28, 28), Image.Resampling.LANCZOS)
        img_gray = img_resized.convert("L")
        
        # 4. Array numpy sin invertir
        img_array = np.array(img_gray)
        
        # Mostrar la imagen que ve el modelo
        st.image(img_array, caption="Imagen procesada para el modelo (28x28)", use_container_width=False)
        
        # 5. Normalizar para TensorFlow
        img_array = img_array.reshape((1, 28, 28, 1)) / 255.0

        with st.spinner("Realizando predicciones..."):
            prediction_d1 = modelo_d1.predict(img_array)
            predicted_class_d1 = np.argmax(prediction_d1)
            predicted_probability_d1 = np.max(prediction_d1)

            prediction_c2 = modelo_c2.predict(img_array)
            predicted_class_c2 = np.argmax(prediction_c2)
            predicted_probability_c2 = np.max(prediction_c2)

            prediction_c3 = modelo_c3.predict(img_array)
            predicted_class_c3 = np.argmax(prediction_c3)
            predicted_probability_c3 = np.max(prediction_c3)

        # Mostrar resultados en columnas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🧠 Modelo D1")
            st.write(f"Predicción: {predicted_class_d1}" + mostrar_mensaje(predicted_probability_d1, "
