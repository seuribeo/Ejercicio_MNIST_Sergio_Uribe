import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gzip
import pickle

# Diccionario de clases para MNIST (0-9)
class_names = [str(i) for i in range(10)]

def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))
    image_array = img_to_array(image) / 255.0  # Normalizar
    image_array = image_array.reshape(1, 28, 28, 1)  # Formato correcto para CNN
    return image_array

def load_model():
    filename = "model_trained_classifier.pkl.gz"  # Archivo del mejor modelo
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    st.markdown(
        '<h1 style="color: #4CAF50; text-align: center;">Clasificación de la base de datos MNIST</h1>',
        unsafe_allow_html=True
    )
    st.markdown("Sube una imagen de un dígito manuscrito para clasificar")

    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_container_width=True)

        preprocessed_image = preprocess_image(image)
        st.image(
            preprocessed_image[0].reshape(28, 28),
            caption="Imagen Preprocesada",
            use_container_width=True
        )

        if st.button("Clasificar imagen"):
            model = load_model()
            prediction = model.predict(preprocessed_image)
            class_id = np.argmax(prediction)
            class_name = class_names[class_id]

            st.markdown(f"### La imagen fue clasificada como: **{class_name}**")

if __name__ == "__main__":
    main()
