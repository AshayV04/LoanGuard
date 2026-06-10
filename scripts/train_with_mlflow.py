# scripts/train_with_mlflow.py
"""
Loan Default Prediction — XGBoost Binary Classification
=========================================================
Runs inside a SageMaker Training Job.
Reads loans.csv from S3, validates schema, trains XGBoost classifier,
logs params + metrics (AUC-ROC, F1) to MLflow, saves model to S3.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Import our data validator — fails fast if the dataset has quality issues
try:
    from validate_data import validate as validate_dataset
    _HAS_VALIDATOR = True
except ImportError:
    _HAS_VALIDATOR = False
    print("[WARN] validate_data not found — skipping schema validation")

TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
MODEL_DIR  = os.environ.get("SM_MODEL_DIR",    "/opt/ml/model")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_csv_path(train_dir: str) -> str:
    csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    if not csv_files:
        raise RuntimeError(f"No CSV found in {train_dir}")
    return os.path.join(train_dir, csv_files[0])


def try_mlflow_setup(uri: str):
    if not uri:
        return None
    try:
        import mlflow
        import mlflow.xgboost
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("loan-default-prediction")
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow setup failed, continuing without it. Error: {e}")
        return None


# ── Load & validate data ──────────────────────────────────────────────────────

csv_path = get_csv_path(TRAIN_DIR)

if _HAS_VALIDATOR:
    df = validate_dataset(csv_path)   # exits non-zero if schema fails
else:
    df = pd.read_csv(csv_path)

print(f"[train] Dataset shape: {df.shape}")

# ── Feature / target split ────────────────────────────────────────────────────
# TARGET must be the first column (SageMaker convention)
TARGET_COL = "TARGET"

y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

# Encode any remaining object columns that slipped through prep
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print(f"[train] Features: {X.shape[1]}  |  Default rate: {y.mean():.1%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Class imbalance — scale_pos_weight ───────────────────────────────────────
# Home Credit dataset is ~8% default. We tell XGBoost to weight positives higher.
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
print(f"[train] scale_pos_weight: {scale_pos_weight:.2f}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "max_depth":        6,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": scale_pos_weight,
    "random_state":     42,
    "tree_method":      "hist",   # fast histogram method
}


def train_and_evaluate(params, X_train, X_test, y_train, y_test):
    """Train model, return (model, metrics_dict)."""
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred, zero_division=0)

    print(classification_report(y_test, y_pred, target_names=["Repaid", "Default"]))
    print(f"[train] AUC-ROC: {auc:.4f}  |  F1: {f1:.4f}")

    return model, {"auc": float(auc), "f1": float(f1)}


def save_artifacts(model, metrics: dict, feature_columns: list):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))

    # Save the exact feature columns the model was trained on.
    # inference.py loads this file so it can reorder/align columns correctly.
    columns_path = os.path.join(MODEL_DIR, "feature_columns.json")
    with open(columns_path, "w") as f:
        json.dump(list(feature_columns), f)
    print(f"[train] Saved {len(feature_columns)} feature columns → feature_columns.json")

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    return metrics_path


# ── Train (with or without MLflow) ───────────────────────────────────────────
mlflow = try_mlflow_setup(MLFLOW_URI)

if mlflow:
    try:
        with mlflow.start_run():
            mlflow.log_params(params)
            model, metrics = train_and_evaluate(params, X_train, X_test, y_train, y_test)
            mlflow.log_metric("auc",      metrics["auc"])
            mlflow.log_metric("f1_score", metrics["f1"])
            metrics_path = save_artifacts(model, metrics, X.columns.tolist())
            mlflow.log_artifact(metrics_path)
            mlflow.xgboost.log_model(model, artifact_path="model")
        print("[OK] Trained + logged to MLflow")
    except Exception as e:
        print(f"[WARN] MLflow logging failed mid-run, saving model only. Error: {e}")
        model, metrics = train_and_evaluate(params, X_train, X_test, y_train, y_test)
        save_artifacts(model, metrics, X.columns.tolist())
else:
    model, metrics = train_and_evaluate(params, X_train, X_test, y_train, y_test)
    save_artifacts(model, metrics, X.columns.tolist())
    print("[OK] Trained model saved (MLflow disabled)")
