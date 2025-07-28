
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo entrenado
modelo = joblib.load("modelo_funcional_vasopresores.pkl")
UMBRAL = 0.3

# Lista de variables esperadas por el modelo
variables = [
    "GCS_first", "GCS_last", "SysABP_first", "DiasABP_first",
    "HR_first", "HR_last", "Temp_first", "Temp_last",
    "PaO2_first", "PaO2_last", "FiO2_first", "FiO2_last",
    "Platelets_first", "Platelets_last", "WBC_first", "WBC_last",
    "Na_first", "Na_last", "K_first", "K_last",
    "Creatinine_first", "Creatinine_last", "Age"
]

st.title("Predicción de uso de vasopresores")
st.markdown("Ingrese los datos clínicos del paciente. Valores faltantes se imputan automáticamente.")

# Formulario interactivo
inputs = {}
with st.form("formulario"):
    for var in variables:
        inputs[var] = st.number_input(var, format="%.2f")
    submitted = st.form_submit_button("Predecir")

# Procesar predicción
if submitted:
    df_input = pd.DataFrame([inputs], columns=variables)

    try:
        prob = modelo.predict_proba(df_input)[0][1]
        pred = int(prob > UMBRAL)
        st.write(f"Probabilidad de requerir vasopresores: **{prob:.2%}**")

        if pred == 1:
            st.error("⚠️ El modelo predice que el paciente SÍ requerirá vasopresores.")
        else:
            st.success("✅ El modelo predice que el paciente NO requerirá vasopresores.")
    except Exception as e:
        st.error(f"❌ Error al procesar predicción: {str(e)}")
