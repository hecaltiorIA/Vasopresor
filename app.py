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
layout=“centered”,
initial_sidebar_state=“collapsed”
)

# Título y descripción

st.title(“🩺 Prediccion de uso de vasopresores”)
st.write(“Llena los datos del paciente para estimar el riesgo de requerir vasopresores.”)

# Función para cargar o crear modelo dummy

@st.cache_resource
def cargar_modelos():
try:
modelo = joblib.load(“modelo.pkl”)
imputador = joblib.load(“imputador.pkl”)
escalador = joblib.load(“escalador.pkl”)
return modelo, imputador, escalador
except FileNotFoundError:
st.warning(“⚠️ No se encontraron los archivos del modelo. Usando modelo de demostracion.”)

```
    # Crear modelo dummy para demostración
    np.random.seed(42)
    X_dummy = np.random.randn(1000, 31)  # 31 características
    y_dummy = np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    
    # Entrenar modelos dummy
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_dummy, y_dummy)
    
    imputador = SimpleImputer(strategy='median')
    imputador.fit(X_dummy)
    
    escalador = StandardScaler()
    escalador.fit(X_dummy)
    
    return modelo, imputador, escalador
```

# Cargar modelos

modelo, imputador, escalador = cargar_modelos()

# Organizar campos en columnas para mejor presentación

st.subheader(“📊 Datos del paciente”)

col1, col2 = st.columns(2)

with col1:
st.markdown(”**🩸 Presion arterial sistolica**”)
SysABP_first = st.number_input(“Primera medicion (mmHg)”, key=“sys_first”, step=0.1, min_value=0.0, max_value=300.0)
SysABP_last = st.number_input(“Ultima medicion (mmHg)”, key=“sys_last”, step=0.1, min_value=0.0, max_value=300.0)
SysABP_lowest = st.number_input(“Mas baja (mmHg)”, key=“sys_low”, step=0.1, min_value=0.0, max_value=300.0)
SysABP_highest = st.number_input(“Mas alta (mmHg)”, key=“sys_high”, step=0.1, min_value=0.0, max_value=300.0)

```
st.markdown("**💓 Presion arterial diastolica**")
DiasABP_first = st.number_input("Primera medicion (mmHg)", key="dias_first", step=0.1, min_value=0.0, max_value=200.0)
DiasABP_last = st.number_input("Ultima medicion (mmHg)", key="dias_last", step=0.1, min_value=0.0, max_value=200.0)
DiasABP_lowest = st.number_input("Mas baja (mmHg)", key="dias_low", step=0.1, min_value=0.0, max_value=200.0)
DiasABP_highest = st.number_input("Mas alta (mmHg)", key="dias_high", step=0.1, min_value=0.0, max_value=200.0)

st.markdown("**🧠 Escala de Glasgow**")
GCS_first = st.number_input("Primera medicion", key="gcs_first", step=0.1, min_value=3.0, max_value=15.0)
GCS_last = st.number_input("Ultima medicion", key="gcs_last", step=0.1, min_value=3.0, max_value=15.0)
GCS_lowest = st.number_input("Mas baja", key="gcs_low", step=0.1, min_value=3.0, max_value=15.0)

st.markdown("**🫁 Oxigenacion**")
PaO2_first = st.number_input("PaO2 primera (mmHg)", key="pao2_first", step=0.1, min_value=0.0, max_value=600.0)
PaO2_last = st.number_input("PaO2 ultima (mmHg)", key="pao2_last", step=0.1, min_value=0.0, max_value=600.0)
FiO2_first = st.number_input("FiO2 primera (%)", key="fio2_first", step=0.1, min_value=0.0, max_value=100.0)
FiO2_last = st.number_input("FiO2 ultima (%)", key="fio2_last", step=0.1, min_value=0.0, max_value=100.0)
```

with col2:
st.markdown(”**❤️ Frecuencia cardiaca**”)
HR_first = st.number_input(“Primera medicion (lpm)”, key=“hr_first”, step=0.1, min_value=0.0, max_value=300.0)
HR_last = st.number_input(“Ultima medicion (lpm)”, key=“hr_last”, step=0.1, min_value=0.0, max_value=300.0)

```
st.markdown("**🌡️ Temperatura**")
Temp_first = st.number_input("Primera medicion (°C)", key="temp_first", step=0.1, min_value=30.0, max_value=45.0)
Temp_last = st.number_input("Ultima medicion (°C)", key="temp_last", step=0.1, min_value=30.0, max_value=45.0)

st.markdown("**🧪 Laboratorios**")
Creatinine_first = st.number_input("Creatinina primera (mg/dL)", key="creat_first", step=0.1, min_value=0.0, max_value=20.0)
Creatinine_last = st.number_input("Creatinina ultima (mg/dL)", key="creat_last", step=0.1, min_value=0.0, max_value=20.0)
Lactate_first = st.number_input("Lactato primero (mmol/L)", key="lact_first", step=0.1, min_value=0.0, max_value=30.0)
Lactate_last = st.number_input("Lactato ultimo (mmol/L)", key="lact_last", step=0.1, min_value=0.0, max_value=30.0)

st.markdown("**⚡ Electrolitos**")
K_first = st.number_input("Potasio primero (mEq/L)", key="k_first", step=0.1, min_value=0.0, max_value=10.0)
K_last = st.number_input("Potasio ultimo (mEq/L)", key="k_last", step=0.1, min_value=0.0, max_value=10.0)
Na_first = st.number_input("Sodio primero (mEq/L)", key="na_first", step=0.1, min_value=100.0, max_value=200.0)
Na_last = st.number_input("Sodio ultimo (mEq/L)", key="na_last", step=0.1, min_value=100.0, max_value=200.0)

st.markdown("**🩸 Hemograma**")
WBC_first = st.number_input("Leucocitos primeros (K/μL)", key="wbc_first", step=0.1, min_value=0.0, max_value=100.0)
WBC_last = st.number_input("Leucocitos ultimos (K/μL)", key="wbc_last", step=0.1, min_value=0.0, max_value=100.0)
Platelets_first = st.number_input("Plaquetas primeras (K/μL)", key="plt_first", step=0.1, min_value=0.0, max_value=1000.0)
Platelets_last = st.number_input("Plaquetas ultimas (K/μL)", key="plt_last", step=0.1, min_value=0.0, max_value=1000.0)

st.markdown("**👤 Datos demograficos**")
Age = st.number_input("Edad (años)", key="age", step=0.1, min_value=0.0, max_value=120.0)
```

# Botón de cálculo centrado

st.markdown(”<br>”, unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
calcular = st.button(“🔍 Calcular riesgo”, use_container_width=True)

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

```
# Procesar datos
valores = np.array([campos])
valores_imp = imputador.transform(valores)
valores_esc = escalador.transform(valores_imp)
prob = modelo.predict_proba(valores_esc)[0][1]

# Determinar color y mensaje según probabilidad
if prob < 0.25:
    color = "#28a745"  # Verde
    mensaje = "RIESGO BAJO"
    icono = "✅"
elif prob < 0.45:
    color = "#ffc107"  # Amarillo
    mensaje = "RIESGO MODERADO-BAJO"
    icono = "⚠️"
elif prob < 0.65:
    color = "#fd7e14"  # Naranja
    mensaje = "RIESGO MODERADO"
    icono = "⚠️"
elif prob < 0.85:
    color = "#dc3545"  # Rojo
    mensaje = "RIESGO ALTO"
    icono = "🚨"
else:
    color = "#6f42c1"  # Morado
    mensaje = "RIESGO MUY ALTO"
    icono = "🆘"

# Mostrar resultado
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style='
        background: linear-gradient(135deg, {color}20, {color}10);
        border: 2px solid {color};
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    '>
        <h2 style='color: {color}; margin: 0;'>
            {icono} {mensaje}
        </h2>
        <h1 style='color: {color}; margin: 10px 0; font-size: 3em;'>
            {prob:.1%}
        </h1>
        <p style='color: {color}; margin: 0; font-size: 1.2em;'>
            Probabilidad de requerir vasopresores
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Barra de progreso visual
st.progress(prob)

# Interpretación clínica
st.markdown("### 📋 Interpretacion clinica")

if prob < 0.25:
    st.success("**Riesgo bajo:** El paciente tiene baja probabilidad de requerir vasopresores. Continuar monitoreo habitual.")
elif prob < 0.45:
    st.warning("**Riesgo moderado-bajo:** Mantener vigilancia estrecha de signos vitales y parametros hemodinamicos.")
elif prob < 0.65:
    st.warning("**Riesgo moderado:** Considerar optimizacion de volemia y monitoreo hemodinamico avanzado.")
elif prob < 0.85:
    st.error("**Riesgo alto:** Preparar para posible inicio de vasopresores. Evaluar causas de inestabilidad hemodinamica.")
else:
    st.error("**Riesgo muy alto:** Considerar inicio inmediato de vasopresores y evaluacion urgente por intensivista.")
```

# Información adicional

with st.expander(“ℹ️ Informacion sobre el modelo”):
st.write(”””
**Acerca de esta herramienta:**
- Este modelo predictivo esta diseñado para asistir en la toma de decisiones clinicas
- Los resultados deben interpretarse en el contexto clinico completo del paciente
- No sustituye el juicio clinico profesional
- Siempre consulte con el equipo medico antes de tomar decisiones terapeuticas

```
**Variables incluidas:**
- Presion arterial sistolica y diastolica (primera, ultima, minima, maxima)
- Escala de Glasgow (primera, ultima, minima)
- Parametros respiratorios (PaO2, FiO2)
- Signos vitales (frecuencia cardiaca, temperatura)
- Laboratorios (creatinina, lactato, electrolitos, hemograma)
- Edad del paciente
""")
```

# Footer

st.markdown(”—”)
st.markdown(
“<p style='text-align: center; color: gray;'>🏥 Herramienta de apoyo clinico - Siempre consulte con profesionales medicos</p>”,
unsafe_allow_html=True
)
