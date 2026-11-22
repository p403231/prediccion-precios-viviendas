import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Predicción de precios de viviendas",
    page_icon="🏠"
)

st.title("Predicción de precios de viviendas")
st.write("Introduce las características de la vivienda.")

# Cargar modelo
modelo = joblib.load("modelo_viviendas.pkl")

# Entradas
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input(
        "Número de habitaciones",
        min_value=0,
        value=3
    )

    sqft_m2 = st.number_input(
        "Superficie habitable (m²)",
        min_value=1.0,
        value=185.0
    )

with col2:
    bathrooms = st.number_input(
        "Número de baños",
        min_value=0.0,
        value=2.0,
        step=0.25
    )

    yr_built = st.number_input(
        "Año de construcción",
        min_value=1900,
        max_value=2026,
        value=2000
    )

sqft_living = sqft_m2 * 10.7639

st.caption(
    f"Equivalente a {sqft_living:,.2f} pies cuadrados"
)

grade = st.slider(
    "Calidad de construcción",
    min_value=1,
    max_value=13,
    value=7
)

if grade <= 3:
    calidad = "Muy baja"
elif grade <= 6:
    calidad = "Baja o básica"
elif grade == 7:
    calidad = "Promedio"
elif grade <= 9:
    calidad = "Buena"
elif grade <= 11:
    calidad = "Muy buena"
else:
    calidad = "Excepcional o de lujo"

st.caption(f"Calidad seleccionada: {calidad}")

vivienda = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "yr_built": yr_built,
    "grade": grade
}])

if st.button("Predecir precio"):

    precio = modelo.predict(vivienda)[0]

    st.divider()

    st.metric(
        label="Precio estimado de la vivienda",
        value=f"${precio:,.2f} USD"
    )

    st.caption(
        "Estimación realizada mediante Random Forest optimizado. "
        "El precio real puede variar."
    )

    st.divider()

with st.expander("Información del modelo"):

    st.subheader("Random Forest Optimizado")

    st.write(
        "Este modelo utiliza técnicas de Machine Learning "
        "para estimar el precio de una vivienda a partir "
        "de sus características."
    )

    st.write("### Rendimiento del modelo en Test")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", "$130,901")

    with col2:
        st.metric("RMSE", "$201,428")

    with col3:
        st.metric("R²", "0.7095")

    st.write("### Características utilizadas")

    st.write(
        "- Número de habitaciones\n"
        "- Número de baños\n"
        "- Superficie habitable\n"
        "- Año de construcción\n"
        "- Calidad de construcción"
    )

    st.info(
        "El modelo fue entrenado con datos de viviendas "
        "del condado de King, Washington, Estados Unidos. "
        "Las predicciones están expresadas en dólares estadounidenses."
    )


with st.expander("Rendimiento del modelo"):

    st.write(
        "Resultados obtenidos con el modelo "
        "Random Forest optimizado sobre el conjunto Test."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="MAE",
            value="$130,901.22 USD"
        )

        st.metric(
            label="MSE",
            value="40,573,244,392.06 USD²"
        )

    with col2:
        st.metric(
            label="RMSE",
            value="$201,428.01 USD"
        )

        st.metric(
            label="R²",
            value="70.95%"
        )

    st.caption(
        "Estas métricas corresponden a la evaluación "
        "del modelo sobre datos de prueba no utilizados "
        "durante el entrenamiento."
    )

with st.expander("Rendimiento del modelo"):

    st.write(
        "Resultados obtenidos con el modelo "
        "Random Forest optimizado sobre el conjunto Test."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="MAE",
            value="$130,901.22 USD"
        )

        st.metric(
            label="MSE",
            value="40,573,244,392.06 USD²"
        )

    with col2:
        st.metric(
            label="RMSE",
            value="$201,428.01 USD"
        )

        st.metric(
            label="R²",
            value="70.95%"
        )

    st.caption(
        "Estas métricas corresponden a la evaluación "
        "del modelo sobre datos de prueba no utilizados "
        "durante el entrenamiento."
    )
