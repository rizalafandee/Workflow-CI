import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import shutil
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. TANGKAP PARAMETER
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 2
if max_depth == 0:
    max_depth = None

# Set Experiment Name (Lokal)
mlflow.set_experiment("Liver_Prediction_Docker")

# 2. LOAD DATA
train_path = "indian_liver_prepocessing/train_df.csv"
test_path = "indian_liver_prepocessing/test_df.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Pisahkan Fitur (X) dan Target (y)
X_train = train_df.drop("is_patient", axis=1)
y_train = train_df["is_patient"]
X_test = test_df.drop("is_patient", axis=1)
y_test = test_df["is_patient"]

# Contoh input untuk log model
input_example = X_train.iloc[:5]

# 3. TRAINING & EVALUASI
with mlflow.start_run():
    # --- A. Train Model ---
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # --- B. Predict ---
    y_pred = model.predict(X_test)
    
    # --- C. Hitung Metrik Lengkap ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # --- D. Log Metrics & Params ke MLflow ---
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # --- E. ARTIFACTS (Simpan File Fisik) ---
    os.makedirs("temp_artifacts", exist_ok=True)

    # 1. Simpan Metrics Lengkap ke TXT
    with open("temp_artifacts/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    mlflow.log_artifact("temp_artifacts/metrics.txt")

    # 2. Simpan Confusion Matrix (Gambar)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"CM - Acc: {acc:.2f} | F1: {f1:.2f}")
    plt.savefig("temp_artifacts/confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("temp_artifacts/confusion_matrix.png")

    # 3. Simpan Classification Report (HTML)
    report = classification_report(y_test, y_pred, output_dict=True)
    html_content = f"<html><body><h3>Classification Report</h3><pre>{json.dumps(report, indent=2)}</pre></body></html>"
    with open("temp_artifacts/estimator.html", "w") as f:
        f.write(html_content)
    mlflow.log_artifact("temp_artifacts/estimator.html")

    # 5. Log Model (Penting untuk Docker Build nanti)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    # Bersih-bersih folder temp
    if os.path.exists("temp_artifacts"):
        shutil.rmtree("temp_artifacts")

ini modelling, sekarang sesuaikan semua file yang ada, dan apakah susah sesuai untuk py to drive nanti
