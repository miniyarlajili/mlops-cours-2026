"""Lab 0 — test de fumée MLflow : un run, une métrique, un artefact."""
import os
import tempfile
from pathlib import Path

import mlflow

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("smoke-test")

with mlflow.start_run(run_name="hello-mlops") as run:
    mlflow.log_param("lab", "m0-setup")
    mlflow.log_metric("accuracy", 0.42)
    with tempfile.TemporaryDirectory() as dossier:
        fichier = Path(dossier) / "hello.txt"
        fichier.write_text("hello mlops\n", encoding="utf-8")
        mlflow.log_artifact(str(fichier))   # envoyé à S3 via le serveur MLflow
    print("run créé :", run.info.run_id)
