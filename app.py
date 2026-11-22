
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Predicción de precios de viviendas",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE_DIR / "modelo_viviendas.pkl")


# ==========================================
# MENÚ LATERAL
# ==========================================

st.sidebar.title("🏠 Proyecto Viviendas")

seccion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Inicio",
        "Dataset",
        "Análisis descriptivo",
        "Modelos implementados",
        "Comparación de modelos",
        "Predicción de viviendas",
        "Rendimiento del modelo"
    ]
)


# ==========================================
# 1. INICIO
# ==========================================

if seccion == "Inicio":

    st.title("Predicción de precios de viviendas")

    st.header("Sistema de Machine Learning")

    st.write("""
    Este proyecto tiene como objetivo desarrollar
    un sistema capaz de estimar el precio de una
    vivienda a partir de sus características.

    Para ello se utilizó el dataset KC House Data,
    correspondiente a viviendas del condado de
    King, Washington, Estados Unidos.

    Se implementaron diferentes algoritmos de
    regresión y se seleccionó Random Forest
    optimizado como modelo final.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Viviendas registradas", "21,613")

    with col2:
        st.metric("Variables predictoras", "5")

    with col3:
        st.metric("R² en Test", "70.95%")

    st.header("Objetivos del proyecto")

    st.markdown("""
    - Analizar las características de las viviendas.
    - Estudiar la relación entre variables y precio.
    - Comparar diferentes modelos de regresión.
    - Optimizar hiperparámetros mediante validación cruzada.
    - Evaluar el rendimiento sobre datos de prueba.
    - Desarrollar una aplicación para estimar precios.
    """)


# ==========================================
# 2. DATASET
# ==========================================

elif seccion == "Dataset":

    st.title("Información del dataset")

    st.subheader("KC House Data")

    st.write("""
    Base de datos de viviendas del condado de
    King, Washington, Estados Unidos.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Registros", "21,613")
    col2.metric("Variables predictoras", "5")
    col3.metric("Tipo de problema", "Regresión")

    st.subheader("Variables utilizadas")

    variables = pd.DataFrame({
        "Variable": [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "yr_built",
            "grade",
            "price"
        ],
        "Tipo de dato": [
            "int64",
            "float64",
            "int64",
            "int64",
            "int64",
            "float64"
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

    st.dataframe(
        variables,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Inspección inicial")

    st.write("""
    El dataset contiene 21,613 observaciones.

    Para el desarrollo del modelo se seleccionaron
    cinco variables independientes y una variable
    objetivo denominada price.
    """)


# ==========================================
# 3. ANÁLISIS DESCRIPTIVO
# ==========================================

elif seccion == "Análisis descriptivo":

    st.title("Análisis descriptivo de variables")

    with st.expander(
        "Distribución de variables",
        expanded=True
    ):

        st.subheader("Variables independientes")

        st.write("""
        Los histogramas permiten observar la
        distribución de las cinco características
        utilizadas para entrenar el modelo.
        """)

        imagen = (
            BASE_DIR /
            "graficas/distribucion_variables.png"
        )

        if imagen.exists():
            st.image(
                str(imagen),
                use_container_width=True
            )
        else:
            st.warning("No se encontró la imagen.")

        st.subheader("Variable objetivo: Precio")

        imagen_precio = (
            BASE_DIR /
            "graficas/distribucion_precios.png"
        )

        if imagen_precio.exists():
            st.image(
                str(imagen_precio),
                use_container_width=True
            )
        else:
            st.warning("No se encontró la imagen.")


    with st.expander("Correlación de variables"):

        st.subheader("Matriz de correlación de Pearson")

        st.write("""
        La siguiente matriz muestra las correlaciones
        entre las variables predictoras y el precio.
        """)

        imagen_corr = (
            BASE_DIR / "graficas/matriz_correlacion.png"
        )

        if imagen_corr.exists():
            st.image(
                str(imagen_corr),
                use_container_width=650
            )
        else:
            st.warning("No se encontró la matriz.")

        st.subheader("Correlación con el precio")

    correlaciones = pd.DataFrame({
        "Variable": [
            "Superficie habitable",
            "Calidad de construcción",
            "Número de baños",
            "Número de habitaciones",
            "Año de construcción"
        ],
        "Correlación": [
            0.702035,
            0.667434,
            0.525138,
            0.308350,
            0.054012
        ]
    })

    st.dataframe(
        correlaciones,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Correlación": st.column_config.NumberColumn(
                "Correlación (r)",
                format="%.4f"
            )
        }
    )

    st.subheader("Interpretación de resultados")

    st.write("""
    **Hallazgos principales:**

    - La superficie habitable presenta la mayor
      correlación positiva con el precio (0.7020).

    - La calidad de construcción también presenta
      una correlación positiva importante (0.6674).

    - El número de baños presenta una correlación
      positiva moderada (0.5251).

    - El número de habitaciones presenta una
      correlación positiva más débil (0.3084).

    - El año de construcción presenta una
      correlación lineal muy débil (0.0540).

    Estas correlaciones no implican causalidad
    ni representan directamente la importancia
    de las variables en Random Forest.
    """)


# ==========================================
# 4. COMPARACION DE MODELOS
# ==========================================

elif seccion == "Comparación de modelos":

    st.title("Comparación de modelos optimizados")

    st.write("""
    Resultados obtenidos mediante validación
    cruzada sobre el conjunto de entrenamiento.
    """)

    comparacion = pd.DataFrame({
        "Modelo": [
            "Random Forest Optimizado",
            "SVR Optimizado",
            "Ridge Optimizado",
            "Lasso Optimizado"
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


    st.subheader("Comparación de métricas")

    metrica = st.segmented_control(
        "Selecciona la métrica que deseas comparar:",
        options=["MAE", "MSE", "RMSE", "R²"],
        default="RMSE",
        selection_mode="single"
    )


    columna = "R2" if metrica == "R²" else metrica


    datos_grafica = comparacion[
        ["Modelo", columna]
    ].copy()


    datos_grafica = datos_grafica.sort_values(
        by=columna,
        ascending=(columna != "R2")
    )

    st.bar_chart(
        datos_grafica.set_index("Modelo"),
        y=columna,
        x_label="Modelo",
        y_label=metrica,
        use_container_width=True
    )

    if columna == "R2":
        st.info(
            "En R², los valores más altos indican "
            "un mejor desempeño del modelo."
        )
    else:
        st.info(
            f"En {metrica}, los valores más bajos "
            "indican un menor error de predicción."
        )

    st.subheader("Conclusión")

    st.write("""
    Random Forest optimizado obtuvo el menor
    RMSE y el mayor R².

    SVR optimizado obtuvo el menor MAE.

    Se seleccionó Random Forest como modelo final
    debido a su mejor desempeño en RMSE y R².
    """)


# ==========================================
# 5. PREDICCIÓN DE VIVIENDAS
# ==========================================

elif seccion == "Predicción de viviendas":

    st.title("Predicción de precios de viviendas")

    st.write("Introduce las características de la vivienda.")

    modelo = cargar_modelo()

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


# ==========================================
# 6. RENDIMIENTO DEL MODELO
# ==========================================

elif seccion == "Rendimiento del modelo":

    st.title("Rendimiento del modelo final")

    st.write("""
    Resultados obtenidos con Random Forest
    optimizado sobre el conjunto Test.
    """)

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

    st.subheader("Interpretación de resultados")

    st.write("""
    **MAE:** El modelo presenta un error absoluto
    promedio de aproximadamente 130,901 dólares.

    **RMSE:** El error cuadrático medio expresado
    en unidades monetarias es de aproximadamente
    201,428 dólares. Esta métrica penaliza
    especialmente los errores grandes.

    **MSE:** Representa el promedio de los errores
    al cuadrado.

    **R²:** El modelo explica aproximadamente el
    70.95% de la variabilidad del precio en el
    conjunto de prueba.
    """)