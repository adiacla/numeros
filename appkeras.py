import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=10,
    stroke_color="#FFFFFF",      # Trazo blanco
    background_color="#000000",  # Fondo negro
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True,
)

if st.button("Predecir"):
    if canvas_result is not None and canvas_result.image_data is not None:
        # 1. Convertir matriz RGBA a imagen PIL en escala de grises
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        img_array = np.array(img)

        # 2. Validar si el canvas está vacío (si el valor máximo es 0, no hay dibujo)
        if np.max(img_array) == 0:
            st.warning("Por favor, dibuja un número antes de predecir.")
        else:
            # 3. Redimensionar a 28x28
            img_resized = img.resize((28, 28))
            img_array_resized = np.array(img_resized)

            # NOTA: Descomenta la línea de abajo SOLO si tu modelo fue entrenado con trazo negro y fondo blanco
            # img_array_resized = 255 - img_array_resized

            # 4. Normalizar (0.0 a 1.0) y ajustar dimensiones a (1, 28, 28, 1)
            input_data = (img_array_resized / 255.0).reshape((1, 28, 28, 1))

            # 5. Mostrar la imagen procesada
            st.image(img_resized, caption="Imagen procesada (28x28)", width=140)

            # 6. Realizar la predicción
            prediction = modelo.predict(input_data)
            predicted_class = np.argmax(prediction)
            predicted_probability = np.max(prediction)

            st.success(f"La predicción es: **{predicted_class}** (Confianza: {predicted_probability:.2%})")
