import streamlit as st
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Función para entrenar y guardar el modelo
def train_and_save_model():
    """Entrena un modelo RandomForest y lo guarda en un archivo."""
    # Cargar el conjunto de datos de Boston desde OpenML
    boston = fetch_openml(name="boston", version=1)
    X = boston.data
    y = boston.target

    # Preprocesar los datos (escalar)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar el modelo RandomForest
    model = RandomForestRegressor()
    model.fit(X_scaled, y)

    # Guardar el modelo entrenado
    with open('model_trained_regressor.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Modelo entrenado y guardado con éxito.")

# Función para cargar el modelo preentrenado
def load_model():
    """Cargar el modelo y sus pesos desde el archivo model_trained_regressor.pkl."""
    with open('model_trained_regressor.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Función para hacer predicciones con el modelo
def predict_price(model, features):
    """Realiza la predicción del precio de la casa."""
    price = model.predict([features])
    return price[0]

# Función principal de la aplicación Streamlit
def main():
    # Entrenamiento del modelo si no existe el archivo
    try:
        load_model()
    except FileNotFoundError:
        st.warning("Modelo no encontrado. Entrenando y guardando el modelo...")
        train_and_save_model()
        st.success("Modelo entrenado y guardado exitosamente.")
    
    # Título de la aplicación
    st.title("Predicción del Precio de una Casa - Boston Housing")

    # Descripción
    st.markdown("""
    Ingresa las características de una casa y obtén la predicción del precio de la casa según el modelo entrenado.
    """)

    # Características de la casa
    st.subheader("Introduce las características de la casa:")
    
    # Entradas de texto para las 13 características
    CRIM = st.number_input("CRIM - Tasa de criminalidad", value=0.1)
    ZN = st.number_input("ZN - Proporción de terrenos residenciales zonificados", value=0.0)
    INDUS = st.number_input("INDUS - Proporción de acres de negocios no minoristas", value=10.0)
    CHAS = st.number_input("CHAS - Proporción de zonas cercanas al río Charles (0 o 1)", value=0)
    NOX = st.number_input("NOX - Concentración de óxidos de nitrógeno", value=0.5)
    RM = st.number_input("RM - Número promedio de habitaciones por casa", value=6.0)
    AGE = st.number_input("AGE - Proporción de casas construidas antes de 1940", value=50.0)
    DIS = st.number_input("DIS - Distancia ponderada a los centros de empleo", value=5.0)
    RAD = st.number_input("RAD - Índice de accesibilidad a carreteras radiales", value=4)
    TAX = st.number_input("TAX - Tasa de impuestos sobre la propiedad", value=300)
    PTRATIO = st.number_input("PTRATIO - Relación alumno/profesor", value=18)
    B = st.number_input("B - Proporción de residentes de origen africano", value=400)
    LSTAT = st.number_input("LSTAT - Porcentaje de población de estatus bajo", value=12.0)

    # Crear una lista con las características ingresadas
    features = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]
    
    # Cargar el modelo preentrenado
    model = load_model()

    # Botón para realizar la predicción
    if st.button("Predecir precio de la casa"):
        # Preprocesar los datos (escalar)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform([features])

        # Realizar la predicción
        predicted_price = predict_price(model, features_scaled[0])

        # Mostrar el resultado
        st.write(f"El precio estimado de la casa es: ${predicted_price:,.2f}")

if __name__ == "__main__":
    main()
