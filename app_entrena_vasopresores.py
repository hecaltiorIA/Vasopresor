
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

st.title("Predicción de uso de vasopresores")

@st.cache_data
def cargar_datos():
    return pd.read_csv("X_train_2025.csv")

df = cargar_datos()

# Variables seleccionadas
variables = [
    "GCS_first", "GCS_last", "SysABP_first", "DiasABP_first",
    "HR_first", "HR_last", "Temp_first", "Temp_last",
    "PaO2_first", "PaO2_last", "FiO2_first", "FiO2_last",
    "Platelets_first", "Platelets_last", "WBC_first", "WBC_last",
    "Na_first", "Na_last", "K_first", "K_last",
    "Creatinine_first", "Creatinine_last", "Age"
]

# Preparar DataFrame
df_modelo = df[variables].copy()
tam = (2 * df["DiasABP_last"] + df["SysABP_last"]) / 3
df_modelo["vasopresores"] = (tam < 65).astype(int)

# Balanceo de clases
df_mayoria = df_modelo[df_modelo["vasopresores"] == 0]
df_minoria = df_modelo[df_modelo["vasopresores"] == 1]
df_minoria_upsampled = resample(df_minoria, replace=True, n_samples=len(df_mayoria), random_state=42)
df_balanceado = pd.concat([df_mayoria, df_minoria_upsampled])

# Entrenamiento
X = df_balanceado.drop(columns=["vasopresores"])
y = df_balanceado["vasopresores"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

modelo = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
modelo.fit(X_train, y_train)

st.markdown("### Ingrese los datos del paciente")
inputs = {}
with st.form("formulario"):
    for var in variables:
        inputs[var] = st.number_input(var, format="%.2f", value=np.nan)
    submitted = st.form_submit_button("Predecir")

if submitted:
    df_input = pd.DataFrame([inputs])
    prob = modelo.predict_proba(df_input)[0][1]
    pred = int(prob > 0.3)
    st.write(f"Probabilidad de requerir vasopresores: **{prob:.2%}**")
    if pred == 1:
        st.error("⚠️ El modelo predice que el paciente SÍ requerirá vasopresores.")
    else:
        st.success("✅ El modelo predice que el paciente NO requerirá vasopresores.")
