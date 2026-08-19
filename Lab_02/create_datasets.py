import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

print("Generating 3 CSV Dataset Files in Workspace...")

# ---------------------------------------------------------
# DATASET 1: Breast Cancer Wisconsin (Diagnostic)
# ---------------------------------------------------------
raw_bc = load_breast_cancer(as_frame=True)
df_bc = raw_bc.data.copy()
df_bc["malignant"] = (raw_bc.target == 0).astype(int) # 1 = Malignant, 0 = Benign
df_bc.to_csv("breast_cancer_dataset.csv", index=False)
print(f"1. Saved breast_cancer_dataset.csv (Shape: {df_bc.shape})")

# ---------------------------------------------------------
# DATASET 2: UCI Heart Disease (Cleveland)
# ---------------------------------------------------------
url_hd = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
hd_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
try:
    df_hd = pd.read_csv(url_hd, names=hd_cols, na_values="?")
    df_hd["target"] = (df_hd["target"] > 0).astype(int) # 1 = Heart Disease Present
    df_hd.to_csv("heart_disease_dataset.csv", index=False)
    print(f"2. Saved heart_disease_dataset.csv (Shape: {df_hd.shape})")
except Exception as e:
    print(f"Note loading online Heart Disease dataset: {e}")

# ---------------------------------------------------------
# DATASET 3: Early Stage Diabetes Risk Prediction
# ---------------------------------------------------------
url_db = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"
try:
    df_db = pd.read_csv(url_db)
    df_db_encoded = df_db.copy()
    for col in df_db_encoded.columns:
        if col != "Age" and col != "class":
            df_db_encoded[col] = (df_db_encoded[col] == "Yes").astype(int)
    df_db_encoded["class"] = (df_db_encoded["class"] == "Positive").astype(int)
    df_db_encoded.to_csv("early_stage_diabetes_dataset.csv", index=False)
    print(f"3. Saved early_stage_diabetes_dataset.csv (Shape: {df_db_encoded.shape})")
except Exception as e:
    print(f"Note loading online Diabetes dataset: {e}")

print("Dataset CSV creation complete!")
