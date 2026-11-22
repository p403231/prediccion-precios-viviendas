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
            value="4.0573 × 10¹⁰ USD²"
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



# 1. Información del dataset
with st.expander("Información del dataset"):
    st.subheader("KC House Data")

    st.write("""
    Base de datos de viviendas del condado de King,
    Washington, Estados Unidos.

    - Total de registros: 21,613
    - Variable objetivo: price
    - Tipo de problema: Regresión
    """)

    st.write("Variables utilizadas:")

    variables = pd.DataFrame({
        "Variable": [
            "bedrooms", "bathrooms", "sqft_living",
            "yr_built", "grade", "price"
        ],
        "Tipo de dato": [
            "int64", "float64", "int64",
            "int64", "int64", "float64"
        ],
        "Descripción": [
            "Número de habitaciones",
            "Número de baños",
            "Superficie habitable en pies cuadrados",
            "Año de construcción",
            "Calidad de construcción",
            "Precio de la vivienda (USD)"
        ]
    })

    st.dataframe(variables, hide_index=True)

    st.caption(
        "Inspección inicial: 21,613 observaciones. "
        "Se seleccionaron cinco variables predictoras "
        "para estimar el precio."
    )


# 2. Modelos implementados
with st.expander("Modelos implementados"):
    st.write("""
    Durante el desarrollo del proyecto se evaluaron
    los siguientes modelos:

    1. Dummy Regressor (modelo de referencia)
    2. Linear Regression
    3. Ridge Regression
    4. Lasso Regression
    5. Random Forest Regressor
    6. Support Vector Regression (SVR)

    Se optimizaron Ridge, Lasso, Random Forest y SVR
    mediante búsqueda de hiperparámetros y
    validación cruzada.
    """)


# 3. Comparación de modelos optimizados
with st.expander("Comparación de modelos optimizados"):

    st.write(
        "Resultados obtenidos mediante validación "
        "cruzada sobre el conjunto de entrenamiento."
    )

    comparacion = pd.DataFrame({
        "Modelo": [
            "Random Forest",
            "SVR",
            "Ridge",
            "Lasso"
        ],
        "MAE": [
            130789.1884,
            129179.4076,
            145139.1894,
            145424.3328
        ],
        "MSE": [
            4.397923e10,
            4.729115e10,
            5.193870e10,
            5.194081e10
        ],
        "RMSE": [
            209653.1956,
            216983.6323,
            227694.3785,
            227703.6734
        ],
        "R2": [
            0.6687,
            0.6470,
            0.6106,
            0.6106
        ]
    })

    st.dataframe(
        comparacion,
        hide_index=True,
        use_container_width=True,
        column_config={
            "MAE": st.column_config.NumberColumn(
                "MAE (USD)", format="%.2f"
            ),
            "MSE": st.column_config.NumberColumn(
                "MSE (USD²)", format="%.3e"
            ),
            "RMSE": st.column_config.NumberColumn(
                "RMSE (USD)", format="%.2f"
            ),
            "R2": st.column_config.NumberColumn(
                "R²", format="%.4f"
            )
        }
    )

    st.write("""
    **Conclusión:** Random Forest optimizado obtuvo
    el menor RMSE y el mayor R², mientras que SVR
    obtuvo el menor MAE.

    Se seleccionó Random Forest como modelo final
    por su mejor desempeño en RMSE y R².
    """)



