
import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Cargar el archivo original
df = pd.read_csv("X_train_2025.csv")

# Variables a conservar
vars_modelo = [
    "GCS_first", "GCS_last", "SysABP_first", "DiasABP_first",
    "HR_first", "HR_last", "Temp_first", "Temp_last",
    "PaO2_first", "PaO2_last", "FiO2_first", "FiO2_last",
    "Platelets_first", "Platelets_last", "WBC_first", "WBC_last",
    "Na_first", "Na_last", "K_first", "K_last",
    "Creatinine_first", "Creatinine_last", "Age"
]
df = df[vars_modelo].dropna()

# Etiqueta: uso de vasopresores si TAM < 65
MAP = (2 * df["DiasABP_first"] + df["SysABP_first"]) / 3
df["vasopresores"] = (MAP < 65).astype(int)

# Balanceo
df_0 = df[df["vasopresores"] == 0]
df_1 = df[df["vasopresores"] == 1]
df_0_down = resample(df_0, replace=False, n_samples=len(df_1), random_state=42)
df_bal = pd.concat([df_0_down, df_1])

X = df_bal.drop(columns=["vasopresores"])
y = df_bal["vasopresores"]

# Variables derivadas
X["WBC_diff"] = X["WBC_last"] - X["WBC_first"]
X["HR_diff"] = X["HR_last"] - X["HR_first"]
X["Creatinine_diff"] = X["Creatinine_last"] - X["Creatinine_first"]
X["PaO2_diff"] = X["PaO2_last"] - X["PaO2_first"]
X["Temp_diff"] = X["Temp_last"] - X["Temp_first"]

# SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Exportar
df_final = X_res.copy()
df_final["vasopresores"] = y_res
df_final.to_csv("dataset.csv", index=False)
print("✅ Dataset generado correctamente como dataset.csv")
