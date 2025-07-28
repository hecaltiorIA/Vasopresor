
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

campos = {    "Presión sistólica al ingreso": st.number_input("Sysabp first", step=0.1),
    "Última presión sistólica": st.number_input("Sysabp last", step=0.1),
    "Presión sistólica más baja": st.number_input("Sysabp lowest", step=0.1),
    "Presión sistólica más alta": st.number_input("Sysabp highest", step=0.1),
    "Presión diastólica al ingreso": st.number_input("Diasabp first", step=0.1),
    "Última presión diastólica": st.number_input("Diasabp last", step=0.1),
    "Presión diastólica más baja": st.number_input("Diasabp lowest", step=0.1),
    "Presión diastólica más alta": st.number_input("Diasabp highest", step=0.1),
    "Glasgow al ingreso": st.number_input("Gcs first", step=0.1),
    "Último Glasgow": st.number_input("Gcs last", step=0.1),
    "Glasgow más bajo": st.number_input("Gcs lowest", step=0.1),
    "PaO2_first": st.number_input("Pao2 first", step=0.1),
    "PaO2_last": st.number_input("Pao2 last", step=0.1),
    "Frecuencia cardíaca al ingreso": st.number_input("Hr first", step=0.1),
    "Última frecuencia cardíaca": st.number_input("Hr last", step=0.1),
    "Temperatura al ingreso": st.number_input("Temp first", step=0.1),
    "Última temperatura": st.number_input("Temp last", step=0.1),
    "FiO2 al ingreso": st.number_input("Fio2 first", step=0.1),
    "Última FiO2": st.number_input("Fio2 last", step=0.1),
    "Creatinina al ingreso": st.number_input("Creatinine first", step=0.1),
    "Última creatinina": st.number_input("Creatinine last", step=0.1),
    "Lactate_first": st.number_input("Lactate first", step=0.1),
    "Lactate_last": st.number_input("Lactate last", step=0.1),
    "Potasio al ingreso": st.number_input("K first", step=0.1),
    "Último potasio": st.number_input("K last", step=0.1),
    "Sodio al ingreso": st.number_input("Na first", step=0.1),
    "Último sodio": st.number_input("Na last", step=0.1),
    "Leucocitos al ingreso": st.number_input("Wbc first", step=0.1),
    "Últimos leucocitos": st.number_input("Wbc last", step=0.1),
    "Plaquetas al ingreso": st.number_input("Platelets first", step=0.1),
    "Últimas plaquetas": st.number_input("Platelets last", step=0.1),
    "Edad": st.number_input("Edad", step=0.1)
}

if st.button("Calcular riesgo"):
    valores = np.array([list(campos.values())])
    valores_imp = imputador.transform(valores)
    valores_esc = escalador.transform(valores_imp)
    prob = modelo.predict_proba(valores_esc)[0][1]

    color = "green" if prob < 0.45 else "yellow" if prob < 0.65 else "orange" if prob < 0.85 else "red"
    st.markdown(f"<h2 style='color:{color}; background-color:black; padding:10px'>Probabilidad de uso de vasopresores: {prob:.2%}</h2>", unsafe_allow_html=True)
