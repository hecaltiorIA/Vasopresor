
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Predicción de uso de vasopresores", layout="centered")
st.title("🧠 Predicción de necesidad de vasopresores")

@st.cache_data
def cargar_datos_entrenar_modelo():
    df = pd.read_csv("dataset.csv")  # Asegúrate de subir este archivo en la misma carpeta
    df["WBC_diff"] = df["WBC_last"] - df["WBC_first"]
    df["HR_diff"] = df["HR_last"] - df["HR_first"]
    df["Creatinine_diff"] = df["Creatinine_last"] - df["Creatinine_first"]
    df["PaO2_diff"] = df["PaO2_last"] - df["PaO2_first"]
    df["Temp_diff"] = df["Temp_last"] - df["Temp_first"]

    X = df.drop(columns=["vasopresores"])
    y = df["vasopresores"]

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    modelo = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    modelo.fit(X_bal, y_bal)
    return modelo

modelo = cargar_datos_entrenar_modelo()

def user_input_features():
    features = {}
    campos = [
        ("GCS_first", 3, 15),
        ("GCS_last", 3, 15),
        ("SysABP_first", 60, 180),
        ("DiasABP_first", 30, 100),
        ("HR_first", 40, 140),
        ("HR_last", 40, 140),
        ("Temp_first", 34.0, 40.0),
        ("Temp_last", 34.0, 40.0),
        ("PaO2_first", 40.0, 200.0),
        ("PaO2_last", 40.0, 200.0),
        ("FiO2_first", 0.21, 1.0),
        ("FiO2_last", 0.21, 1.0),
        ("Platelets_first", 20.0, 500.0),
        ("Platelets_last", 20.0, 500.0),
        ("WBC_first", 1.0, 40.0),
        ("WBC_last", 1.0, 40.0),
        ("Na_first", 120.0, 160.0),
        ("Na_last", 120.0, 160.0),
        ("K_first", 2.0, 6.5),
        ("K_last", 2.0, 6.5),
        ("Creatinine_first", 0.3, 7.0),
        ("Creatinine_last", 0.3, 7.0),
        ("Age", 18, 100)
    ]
    for nombre, minv, maxv in campos:
        features[nombre] = st.slider(nombre, minv, maxv, float((minv+maxv)/2))

    # derivadas
    features["WBC_diff"] = features["WBC_last"] - features["WBC_first"]
    features["HR_diff"] = features["HR_last"] - features["HR_first"]
    features["Creatinine_diff"] = features["Creatinine_last"] - features["Creatinine_first"]
    features["PaO2_diff"] = features["PaO2_last"] - features["PaO2_first"]
    features["Temp_diff"] = features["Temp_last"] - features["Temp_first"]

    return pd.DataFrame(features, index=[0])

df_input = user_input_features()

if st.button("Predecir"):
    pred_proba = modelo.predict_proba(df_input)[0][1]
    st.subheader("Resultado:")
    st.write(f"Probabilidad de requerir vasopresores: **{pred_proba*100:.2f}%**")
