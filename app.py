
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de vasopresores", layout="centered")
st.title("🩺 Predicción de uso de vasopresores")
st.write("Llena los datos del paciente para estimar el riesgo de requerir vasopresores.")

modelo = joblib.load("modelo.pkl")
imputador = joblib.load("imputador.pkl")
escalador = joblib.load("escalador.pkl")

campos = {    "SysABP_first": st.number_input("Sysabp first", step=0.1),
    "SysABP_last": st.number_input("Sysabp last", step=0.1),
    "SysABP_lowest": st.number_input("Sysabp lowest", step=0.1),
    "SysABP_highest": st.number_input("Sysabp highest", step=0.1),
    "DiasABP_first": st.number_input("Diasabp first", step=0.1),
    "DiasABP_last": st.number_input("Diasabp last", step=0.1),
    "DiasABP_lowest": st.number_input("Diasabp lowest", step=0.1),
    "DiasABP_highest": st.number_input("Diasabp highest", step=0.1),
    "GCS_first": st.number_input("Gcs first", step=0.1),
    "GCS_last": st.number_input("Gcs last", step=0.1),
    "GCS_lowest": st.number_input("Gcs lowest", step=0.1),
    "PaO2_first": st.number_input("Pao2 first", step=0.1),
    "PaO2_last": st.number_input("Pao2 last", step=0.1),
    "HR_first": st.number_input("Hr first", step=0.1),
    "HR_last": st.number_input("Hr last", step=0.1),
    "Temp_first": st.number_input("Temp first", step=0.1),
    "Temp_last": st.number_input("Temp last", step=0.1),
    "FiO2_first": st.number_input("Fio2 first", step=0.1),
    "FiO2_last": st.number_input("Fio2 last", step=0.1),
    "Creatinine_first": st.number_input("Creatinine first", step=0.1),
    "Creatinine_last": st.number_input("Creatinine last", step=0.1),
    "Lactate_first": st.number_input("Lactate first", step=0.1),
    "Lactate_last": st.number_input("Lactate last", step=0.1),
    "K_first": st.number_input("K first", step=0.1),
    "K_last": st.number_input("K last", step=0.1),
    "Na_first": st.number_input("Na first", step=0.1),
    "Na_last": st.number_input("Na last", step=0.1),
    "WBC_first": st.number_input("Wbc first", step=0.1),
    "WBC_last": st.number_input("Wbc last", step=0.1),
    "Platelets_first": st.number_input("Platelets first", step=0.1),
    "Platelets_last": st.number_input("Platelets last", step=0.1),
    "Age": st.number_input("Age", step=0.1)
}

if st.button("Calcular riesgo"):
    valores = np.array([list(campos.values())])
    valores_imp = imputador.transform(valores)
    valores_esc = escalador.transform(valores_imp)
    prob = modelo.predict_proba(valores_esc)[0][1]

    color = "green" if prob < 0.45 else "yellow" if prob < 0.65 else "orange" if prob < 0.85 else "red"
    st.markdown(f"<h2 style='color:{color}; background-color:black; padding:10px'>Probabilidad de uso de vasopresores: {prob:.2%}</h2>", unsafe_allow_html=True)
