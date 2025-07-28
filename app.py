
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo y herramientas
modelo = joblib.load("modelo.pkl")
imputador = joblib.load("imputador.pkl")
escalador = joblib.load("escalador.pkl")

# Etiquetas de los campos en español
etiquetas = {
    "DiasABP_first": "Presión diastólica inicial",
    "SysABP_first": "Presión sistólica inicial",
    "SAPS-I": "Escala SAPS-I",
    "GCS_lowest": "Glasgow más bajo",
    "PaO2_first": "PaO2 inicial",
    "CSRU": "Unidad de recuperación cardíaca (0 = No, 1 = Sí)",
    "HR_first": "Frecuencia cardíaca inicial",
    "GCS_first": "Glasgow al ingreso",
    "GCS_last": "Último Glasgow",
    "Temp_first": "Temperatura inicial",
    "FiO2_first": "FiO2 inicial",
    "Creatinine_first": "Creatinina inicial",
    "Lactate_first": "Lactato inicial",
    "K_first": "Potasio",
    "Na_first": "Sodio",
    "WBC_first": "Leucocitos"
}

campos = list(etiquetas.keys())

# Título
st.set_page_config(page_title="Predicción de Vasopresores", layout="centered")
st.title("🩸 Predicción de uso de vasopresores")

st.write("Llena los datos del paciente para calcular la probabilidad de requerir vasopresores.")

# Formulario
valores_usuario = {}
for campo in campos:
    if campo == "CSRU":
        valores_usuario[campo] = st.number_input(etiquetas[campo], min_value=0, max_value=1, step=1, key=campo)
    else:
        valores_usuario[campo] = st.number_input(etiquetas[campo], step=0.1, format="%.2f", key=campo)

if st.button("Calcular riesgo"):
    if any(valor is None or (isinstance(valor, (int, float)) and valor == 0 and campo != "CSRU") for campo, valor in valores_usuario.items()):
        st.warning("⚠️ Por favor, llena todos los campos del formulario antes de continuar.")
    else:
        df_usuario = pd.DataFrame([valores_usuario])
        df_usuario_imputado = imputador.transform(df_usuario)
        df_usuario_escalado = escalador.transform(df_usuario_imputado)
        probabilidad = modelo.predict_proba(df_usuario_escalado)[0][1]

        if probabilidad >= 0.75:
            color = "red"
        elif probabilidad >= 0.5:
            color = "orange"
        elif probabilidad >= 0.25:
            color = "yellow"
        else:
            color = "green"

        st.markdown(f"<h2 style='color:{color}'>Probabilidad de requerir vasopresores: {probabilidad:.2f}</h2>", unsafe_allow_html=True)
