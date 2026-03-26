import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

THRESHOLD = 0.85

# Set MLflow tracking server
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

# Read run ID from file
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

# Fetch run from MLflow
run = client.get_run(run_id)

# Read accuracy metric
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("Error: accuracy metric not found in MLflow.")
    sys.exit(1)

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print("Model accuracy is below threshold. Deployment failed.")
    sys.exit(1)

print("Model accuracy meets threshold. Deployment can continue.")