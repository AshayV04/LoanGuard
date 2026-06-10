"""
SageMaker Inference Handler — Loan Default Prediction
======================================================
This file is bundled inside model.tar.gz during deployment.
SageMaker calls these four functions to serve predictions.
"""
import joblib
import json
import os
import numpy as np
import pandas as pd


def model_fn(model_dir):
    """
    Load the model AND the feature column list saved during training.
    The column list is essential for aligning API input to training features,
    especially after one-hot encoding creates variable column counts.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    columns_path = os.path.join(model_dir, "feature_columns.json")
    if os.path.exists(columns_path):
        with open(columns_path) as f:
            feature_columns = json.load(f)
        print(f"[inference] Loaded {len(feature_columns)} feature columns")
    else:
        feature_columns = None
        print("[inference] WARN: feature_columns.json not found — column alignment disabled")

    return {"model": model, "feature_columns": feature_columns}


def input_fn(request_body, content_type):
    """
    Parse the CSV payload sent by the FastAPI app into a DataFrame.

    Expected format (header + values on separate lines):
        AMT_INCOME_TOTAL,AMT_CREDIT,...
        180000,450000,...
    """
    if content_type == "text/csv":
        lines = [line.strip() for line in request_body.strip().split("\n") if line.strip()]

        if len(lines) < 2:
            raise ValueError(f"Expected header + data row, got {len(lines)} line(s)")

        headers = [h.strip() for h in lines[0].split(",")]
        rows = []
        for line in lines[1:]:
            try:
                row = [float(x.strip()) for x in line.split(",")]
                rows.append(row)
            except ValueError:
                continue   # skip non-numeric lines gracefully

        if not rows:
            raise ValueError("No valid numeric data rows found in request")

        return pd.DataFrame(rows, columns=headers)

    raise ValueError(f"Unsupported content type: {content_type}. Expected text/csv")


def predict_fn(input_df, model_bundle):
    """
    Align input columns to training feature order, then run predict_proba.

    Any columns present in training but missing from the request (e.g. one-hot
    encoded columns that are zero for this applicant) are filled with 0.0.
    """
    model           = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    if feature_columns is not None:
        # Fill missing OHE columns with 0 (applicant has none of that category)
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0
        # Reorder to exactly match training column order
        input_df = input_df[feature_columns]

    proba = model.predict_proba(input_df.values)[:, 1]  # P(default)
    return proba


def output_fn(prediction, accept):
    """Return default probabilities as comma-separated floats (0.0–1.0)."""
    return ",".join(str(round(float(p), 4)) for p in prediction)
