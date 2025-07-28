
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Función para generar dataset si no existe
@st.cache_data
def generar_dataset():
    df = pd.read_csv("X_train_2025.csv")

    columnas = [
        "GCS_first", "GCS_last", "SysABP_first", "DiasABP_first",
        "HR_first", "HR_last", "Temp_first", "Temp_last",
        "PaO2_first", "PaO2_last", "FiO2_first", "FiO2_last",
        "Platelets_first", "Platelets_last", "WBC_first", "WBC_last",
        "Na_first", "Na_last", "K_first", "K_last",
        "Creatinine_first", "Creatinine_last", "Age"
    ]

    df = df[columnas].copy()

    # Calcular PAM
    df["PAM"] = (df["SysABP_first"] + 2 * df["DiasABP_first"]) / 3
    df["vasopresores"] = (df["PAM"] < 65).astype(int)

    # Nuevas variables
    df["WBC_diff"] = df["WBC_last"] - df["WBC_first"]
    df["HR_diff"] = df["HR_last"] - df["HR_first"]
    df["Creatinine_diff"] = df["Creatinine_last"] - df["Creatinine_first"]
    df["PaO2_diff"] = df["PaO2_last"] - df["PaO2_first"]
    df["Temp_diff"] = df["Temp_last"] - df["Temp_first"]

    df.drop(columns=["PAM"], inplace=True)

    df.to_csv("dataset.csv", index=False)
    return df

# Función para cargar y entrenar
@st.cache_data
def cargar_y_entrenar():
    if not os.path.exists("dataset.csv"):
        df = generar_dataset()
    else:
        df = pd.read_csv("dataset.csv")

    X = df.drop("vasopresores", axis=1)
    y = df["vasopresores"]

    imputer = SimpleImputer()
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    X_res, y_res = SMOTE().fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    return modelo, X.columns.tolist(), X_test, y_test, y_pred, y_proba

# Streamlit
st.set_page_config(page_title="Predicción de Vasopresores", layout="centered")
st.title("🧠 Predicción de Uso de Vasopresores")
st.markdown("Esta app predice si un paciente usará vasopresores a partir de variables clínicas.")

modelo, columnas, X_test, y_test, y_pred, y_proba = cargar_y_entrenar()

st.subheader("Ingresa los valores del paciente:")

valores = []
for col in columnas:
    val = st.number_input(col, value=70.0)
    valores.append(val)

if st.button("Predecir necesidad de vasopresores"):
    resultado = modelo.predict([valores])[0]
    prob = modelo.predict_proba([valores])[0][1]
    st.write(f"**Probabilidad:** {prob:.2%}")
    if resultado == 1:
        st.error("🩺 El modelo predice que este paciente probablemente requerirá vasopresores.")
    else:
        st.success("✅ El modelo predice que probablemente NO requerirá vasopresores.")

st.subheader("📊 Métricas del modelo")
st.write("ROC AUC:", roc_auc_score(y_test, y_proba))

fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
st.pyplot(fig)

cm = confusion_matrix(y_test, y_pred)
st.write("Matriz de confusión:")
st.dataframe(pd.DataFrame(cm, columns=["No VP", "Sí VP"], index=["No VP", "Sí VP"]))

