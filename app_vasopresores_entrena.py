
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("Predicción de uso de vasopresores - Entrenamiento en tiempo real")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("X_train_2025.csv")
    return df

# Cargar datos
df = cargar_datos()

# Variables seleccionadas
variables = [
    "GCS_first", "GCS_last", "GCS_lowest",
    "SysABP_first", "SysABP_last", "SysABP_highest", "SysABP_lowest",
    "DiasABP_first", "DiasABP_last", "DiasABP_highest", "DiasABP_lowest",
    "HR_first", "HR_last", "Temp_first", "Temp_last",
    "PaO2_first", "PaO2_last", "FiO2_first", "FiO2_last",
    "Platelets_first", "Platelets_last", "WBC_first", "WBC_last",
    "Na_first", "Na_last", "K_first", "K_last",
    "Creatinine_first", "Creatinine_last", "Age"
]

# Preparar datos
df_modelo = df[variables].copy()
tam = (2 * df["DiasABP_last"] + df["SysABP_last"]) / 3
df_modelo["vasopresores"] = (tam < 65).astype(int)
df_modelo = df_modelo.fillna(df_modelo.median(numeric_only=True))

X = df_modelo.drop(columns=["vasopresores"])
y = df_modelo["vasopresores"]

# Dividir y entrenar
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
modelo = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
modelo.fit(X_train, y_train)

# Formulario para predicción
st.markdown("### Ingrese datos del paciente")
inputs = {}
with st.form("formulario"):
    for var in variables:
        inputs[var] = st.number_input(var, format="%.2f", value=np.nan)
    submitted = st.form_submit_button("Predecir")

# Medianas para imputación
medianas = X_train.median().to_dict()

# Hacer predicción si se envió el formulario
if submitted:
    df_input = pd.DataFrame([inputs])
    for col in df_input.columns:
        if pd.isna(df_input[col].values[0]):
            df_input[col] = medianas[col]

    prob = modelo.predict_proba(df_input)[0][1]
    pred = int(prob > 0.3)
    st.write(f"Probabilidad de requerir vasopresores: **{prob:.2%}**")
    if pred == 1:
        st.error("⚠️ El modelo predice que el paciente SÍ requerirá vasopresores.")
    else:
        st.success("✅ El modelo predice que el paciente NO requerirá vasopresores.")
