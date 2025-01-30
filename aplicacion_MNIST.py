import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle

# Diccionario para asignar nombres a las clases (ajustado para 0-9)
class_names = [str(i) for i in range(10)]

def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar la imagen a 28x28
    image_array = img_to_array(image) / 255.0  # Normalizar los valores de píxeles entre 0 y 1
    image_array = image_array.flatten().reshape(1, -1)  # Aplanar para KNN y asegurarse de que sea 2D (1, 784)
    return image_array

def load_model():
    filename = "model_trained_classifier.pkl.gz"  # Nombre del archivo del modelo entrenado
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)  # Cargar el modelo entrenado
    return model

def main():
    # Título con color
    st.markdown(
        '<h1 style="color: #4CAF50; text-align: center;">Clasificación de la base de datos MNIST</h1>',
        unsafe_allow_html=True
    )
    st.markdown("Sube una imagen de un dígito manuscrito para clasificar")

    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown("### Imagen original:")
        st.image(image, caption="Imagen subida", use_container_width=True)

        # Preprocesar la imagen
        preprocessed_image = preprocess_image(image)

        # Mostrar las imágenes lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen Original", use_container_width=True)
        with col2:
            st.image(
                preprocessed_image[0].reshape(28, 28),
                caption="Imagen Preprocesada",
                use_container_width=True
            )

        if st.button("Clasificar imagen"):
            model = load_model()
            prediction = model.predict(preprocessed_image)  # Realizar predicción
            class_id = np.argmax(prediction)  # Obtener índice de la clase predicha
            class_name = class_names[class_id]  # Obtener nombre de la clase

            st.markdown(f"### La imagen fue clasificada como: **{class_name}**")

if __name__ == "__main__":
    main()

