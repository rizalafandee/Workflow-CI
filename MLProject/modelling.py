import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import shutil
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. KONFIGURASI DAGSHUB & MLFLOW
dagshub.init(repo_owner='rizalafandee', repo_name='Liver_Prediction', mlflow=True)
mlflow.set_experiment("Liver_Prediction_CI") 

# 2. LOAD DATA (PATH RELATIF BARU)
train_df = pd.read_csv('indian_liver_preprocessing/train_df.csv')
test_df = pd.read_csv('indian_liver_preprocessing/test_df.csv')

X_train = train_df.drop("is_patient", axis=1)
y_train = train_df["is_patient"]
X_test = test_df.drop("is_patient", axis=1)
y_test = test_df["is_patient"]

input_example = X_train.iloc[:5]

# 3. HYPERPARAMETER TUNING LOOP
param_grid = {
    "n_estimators": [50, 100],      
    "max_depth": [None, 10],        
    "min_samples_split": [2, 5] 
}

for params in ParameterGrid(param_grid):
    
    # Nama run disesuaikan parameter
    nama_run = f"Run_Est-{params['n_estimators']}_Depth-{params['max_depth']}_Split-{params['min_samples_split']}"

    with mlflow.start_run(run_name=nama_run):
        # A. Train Model
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        
        # B. Predict
        y_pred = model.predict(X_test)
        
        # C. Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # D. Log ke DagsHub
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # D. ARTIFACTS
        os.makedirs("temp_artifacts", exist_ok=True)

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"CM - F1: {f1:.2f}")
        plt.savefig("temp_artifacts/confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("temp_artifacts/confusion_matrix.png")

        # 2. HTML Report
        report = classification_report(y_test, y_pred, output_dict=True)
        html_content = f"<html><body><pre>{json.dumps(report, indent=2)}</pre></body></html>"
        with open("temp_artifacts/estimator.html", "w") as f:
            f.write(html_content)
        mlflow.log_artifact("temp_artifacts/estimator.html")

        # 3. Log Model & Schema
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Bersih-bersih folder temp
        if os.path.exists("temp_artifacts"):
            shutil.rmtree("temp_artifacts")
