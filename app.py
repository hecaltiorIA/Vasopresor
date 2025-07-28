
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Predicción de uso de vasopresores", layout="centered")

st.title("🧠 Predicción de necesidad de vasopresores")

st.markdown("Ingresa las variables clínicas del paciente para estimar el riesgo.")

# Campos de entrada
def user_input_features():
    GCS_first = st.slider("Glasgow al ingreso", 3, 15, 15)
    GCS_last = st.slider("Último Glasgow", 3, 15, 15)
    SysABP_first = st.number_input("Presión sistólica al ingreso", value=100)
    DiasABP_first = st.number_input("Presión diastólica al ingreso", value=60)
    HR_first = st.number_input("FC al ingreso", value=85)
    HR_last = st.number_input("Última FC", value=90)
    Temp_first = st.number_input("Temperatura al ingreso", value=36.5)
    Temp_last = st.number_input("Última temperatura", value=36.7)
    PaO2_first = st.number_input("PaO2 al ingreso", value=90.0)
    PaO2_last = st.number_input("Último PaO2", value=85.0)
    FiO2_first = st.number_input("FiO2 al ingreso", value=0.4)
    FiO2_last = st.number_input("Última FiO2", value=0.4)
    Platelets_first = st.number_input("Plaquetas al ingreso", value=150.0)
    Platelets_last = st.number_input("Últimas plaquetas", value=160.0)
    WBC_first = st.number_input("Leucocitos al ingreso", value=8.0)
    WBC_last = st.number_input("Últimos leucocitos", value=9.0)
    Na_first = st.number_input("Sodio al ingreso", value=140.0)
    Na_last = st.number_input("Último sodio", value=139.0)
    K_first = st.number_input("Potasio al ingreso", value=4.2)
    K_last = st.number_input("Último potasio", value=4.3)
    Creatinine_first = st.number_input("Creatinina al ingreso", value=1.0)
    Creatinine_last = st.number_input("Última creatinina", value=1.2)
    Age = st.number_input("Edad", value=60)

    data = {
        "GCS_first": GCS_first,
        "GCS_last": GCS_last,
        "SysABP_first": SysABP_first,
        "DiasABP_first": DiasABP_first,
        "HR_first": HR_first,
        "HR_last": HR_last,
        "Temp_first": Temp_first,
        "Temp_last": Temp_last,
        "PaO2_first": PaO2_first,
        "PaO2_last": PaO2_last,
        "FiO2_first": FiO2_first,
        "FiO2_last": FiO2_last,
        "Platelets_first": Platelets_first,
        "Platelets_last": Platelets_last,
        "WBC_first": WBC_first,
        "WBC_last": WBC_last,
        "Na_first": Na_first,
        "Na_last": Na_last,
        "K_first": K_first,
        "K_last": K_last,
        "Creatinine_first": Creatinine_first,
        "Creatinine_last": Creatinine_last,
        "Age": Age,
        "WBC_diff": WBC_last - WBC_first,
        "HR_diff": HR_last - HR_first,
        "Creatinine_diff": Creatinine_last - Creatinine_first,
        "PaO2_diff": PaO2_last - PaO2_first,
        "Temp_diff": Temp_last - Temp_first
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

if st.button("Predecir"):
    modelo = joblib.load("modelo_final_vasopresores.pkl")
    prediccion = modelo.predict_proba(input_df)[0][1]
    st.subheader("Resultado:")
    st.write(f"Riesgo estimado de requerir vasopresores: **{prediccion * 100:.1f}%**")
