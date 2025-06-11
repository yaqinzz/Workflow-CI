import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
X_train_scaled = pd.read_csv('../Lung_Cancer_preprocessing/X_train_scaled.csv')
X_test_scaled = pd.read_csv('../Lung_Cancer_preprocessing/X_test_scaled.csv')
y_train = pd.read_csv('../Lung_Cancer_preprocessing/y_train.csv')
y_test = pd.read_csv('../Lung_Cancer_preprocessing/y_test.csv')

# Convert target to 1D array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Lung_Cancer_CI_Run")
mlflow.sklearn.autolog()

# Buat folder output jika belum ada
os.makedirs("output", exist_ok=True)

try:
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        # Save run_id to file for GitHub Actions
        with open("mlflow_run_id.txt", "w") as f:
            f.write(run_id)
        print("Run ID saved to mlflow_run_id.txt")

        # Define & train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Log additional metrics (autolog already does this)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log model explicitly (optional but useful)
        mlflow.sklearn.log_model(sk_model= model, artifact_path="model",)

        print("Model trained and logged to MLflow.")
        print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1 Score: {f1}")

        # Save metrics to JSON
        metrics_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

        with open("model_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print("Metrics saved to model_metrics.json")

        # Log metrics JSON as artifact
        mlflow.log_artifact("model_metrics.json", artifact_path="metrics")

except Exception as e:
    print(f"[ERROR] Something went wrong: {e}")
    raise

# Optional: check file exists after run
if not os.path.exists("mlflow_run_id.txt"):
    raise FileNotFoundError("mlflow_run_id.txt was not created.")
