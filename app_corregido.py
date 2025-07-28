import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

# Configuración de la página

st.set_page_config(
page_title="Prediccion de vasopresores",
layout="centered",
initial_sidebar_state="collapsed"
)

# Título y descripción

st.title("🩺 Prediccion de uso de vasopresores")
st.write("Llena los datos del paciente para estimar el riesgo de requerir vasopresores.")

# Función para cargar o crear modelo dummy

@st.cache_resource
def cargar_modelos():
    try:
        modelo = joblib.load("modelo.pkl")
        imputador = joblib.load("imputador.pkl")
        escalador = joblib.load("escalador.pkl")
        return modelo, imputador, escalador
    except FileNotFoundError:
        st.warning("⚠️ No se encontraron los archivos del modelo. Usando modelo de demostración.")
        np.random.seed(42)
        X_dummy = np.random.randn(1000, 31)
        y_dummy = np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_dummy, y_dummy)
        imputador = SimpleImputer(strategy='median')
        imputador.fit(X_dummy)
        escalador = StandardScaler()
        escalador.fit(X_dummy)
        return modelo, imputador, escalador
try:
modelo = joblib.load("modelo.pkl")
imputador = joblib.load("imputador.pkl")
escalador = joblib.load("escalador.pkl")
return modelo, imputador, escalador
except FileNotFoundError:
st.warning("⚠️ No se encontraron los archivos del modelo. Usando modelo de demostracion.")



# Cargar modelos

modelo, imputador, escalador = cargar_modelos()

# Organizar campos en columnas para mejor presentación

st.subheader("📊 Datos del paciente")

col1, col2 = st.columns(2)

with col1:
st.markdown("**🩸 Presion arterial sistolica**")
SysABP_first = st.number_input("Primera medicion (mmHg)", key="sys_first", step=0.1, min_value=0.0, max_value=300.0)
SysABP_last = st.number_input("Ultima medicion (mmHg)", key="sys_last", step=0.1, min_value=0.0, max_value=300.0)
SysABP_lowest = st.number_input("Mas baja (mmHg)", key="sys_low", step=0.1, min_value=0.0, max_value=300.0)
SysABP_highest = st.number_input("Mas alta (mmHg)", key="sys_high", step=0.1, min_value=0.0, max_value=300.0)



with col2:
st.markdown("**❤️ Frecuencia cardiaca**")
HR_first = st.number_input("Primera medicion (lpm)", key="hr_first", step=0.1, min_value=0.0, max_value=300.0)
HR_last = st.number_input("Ultima medicion (lpm)", key="hr_last", step=0.1, min_value=0.0, max_value=300.0)



# Botón de cálculo centrado

st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
calcular = st.button("🔍 Calcular riesgo", use_container_width=True)

if calcular:
# Recopilar todos los valores
campos = [
SysABP_first, SysABP_last, SysABP_lowest, SysABP_highest,
DiasABP_first, DiasABP_last, DiasABP_lowest, DiasABP_highest,
GCS_first, GCS_last, GCS_lowest,
PaO2_first, PaO2_last,
HR_first, HR_last,
Temp_first, Temp_last,
FiO2_first, FiO2_last,
Creatinine_first, Creatinine_last,
Lactate_first, Lactate_last,
K_first, K_last,
Na_first, Na_last,
WBC_first, WBC_last,
Platelets_first, Platelets_last,
Age
]



# Información adicional

with st.expander("ℹ️ Informacion sobre el modelo"):
st.write("""
**Acerca de esta herramienta:**
- Este modelo predictivo esta diseñado para asistir en la toma de decisiones clinicas
- Los resultados deben interpretarse en el contexto clinico completo del paciente
- No sustituye el juicio clinico profesional
- Siempre consulte con el equipo medico antes de tomar decisiones terapeuticas
# Footer
st.markdown("—")
st.markdown(
"<p style='text-align: center; color: gray;'>🏥 Herramienta de apoyo clinico - Siempre consulte con profesionales medicos</p>",
unsafe_allow_html=True
)
"""