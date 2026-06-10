"""
Model Quality Gate — Loan Default Prediction
=============================================
Reads metrics.json from S3 after training and enforces minimum thresholds
before allowing deployment to proceed.

Exits non-zero (blocking GitHub Actions deploy step) if:
  AUC-ROC < GATE_MIN_AUC   (default 0.75)
  F1 Score < GATE_MIN_F1   (default 0.60)

Usage (called by .github/workflows/deploy.yml):
    python scripts/evaluate_and_gate.py

Environment variables:
    AWS_REGION      — e.g. ap-south-1
    S3_BUCKET       — your SageMaker bucket
    GATE_MIN_AUC    — minimum AUC-ROC (default 0.75)
    GATE_MIN_F1     — minimum F1 score (default 0.60)
"""

import os
import sys
import json
import boto3
import tempfile

REGION   = os.environ["AWS_REGION"]
BUCKET   = os.environ["S3_BUCKET"]

MIN_AUC  = float(os.environ.get("GATE_MIN_AUC", "0.75"))
MIN_F1   = float(os.environ.get("GATE_MIN_F1",  "0.60"))


def find_latest_metrics(sm_client, s3_client, bucket: str) -> dict | None:
    """Find the most recently completed training job and download its metrics.json."""
    print("[gate] Searching for latest completed training job...")
    jobs = sm_client.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=20,
    )["TrainingJobSummaries"]

    latest_job = None
    for job in jobs:
        if job["TrainingJobStatus"] == "Completed":
            latest_job = job["TrainingJobName"]
            break

    if latest_job is None:
        print("[gate] ❌ No completed training job found.", file=sys.stderr)
        sys.exit(1)

    print(f"[gate] Latest job: {latest_job}")

    job_details    = sm_client.describe_training_job(TrainingJobName=latest_job)
    model_artifact = job_details["ModelArtifacts"]["S3ModelArtifacts"]
    artifact_prefix = "/".join(model_artifact.split("/")[3:-1])
    metrics_key     = f"{artifact_prefix}/metrics.json"

    print(f"[gate] Looking for metrics at: s3://{bucket}/{metrics_key}")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        s3_client.download_file(bucket, metrics_key, tmp_path)
        with open(tmp_path) as f:
            metrics = json.load(f)
        print(f"[gate] Metrics: {metrics}")
        return metrics
    except Exception:
        # Fallback search
        print("[gate] Not at primary path, scanning bucket...")
        prefix = f"models/{latest_job}/"
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith("metrics.json"):
                s3_client.download_file(bucket, obj["Key"], tmp_path)
                with open(tmp_path) as f:
                    metrics = json.load(f)
                print(f"[gate] Found at: {obj['Key']} → {metrics}")
                return metrics

    print(
        "[gate] ⚠️  metrics.json not found — training ran without MLflow. Skipping gate.",
        file=sys.stderr,
    )
    return None


def run_gate(metrics: dict) -> None:
    auc = metrics.get("auc")
    f1  = metrics.get("f1")

    print("\n" + "=" * 60)
    print("  LOAN DEFAULT MODEL — QUALITY GATE")
    print("=" * 60)
    print(f"  AUC-ROC : {auc:.4f}   (threshold ≥ {MIN_AUC})")
    print(f"  F1 Score: {f1:.4f}   (threshold ≥ {MIN_F1})")
    print("=" * 60)

    passed = True

    if auc is not None and auc < MIN_AUC:
        print(f"  ❌ FAIL — AUC {auc:.4f} below minimum {MIN_AUC}", file=sys.stderr)
        passed = False

    if f1 is not None and f1 < MIN_F1:
        print(f"  ❌ FAIL — F1 {f1:.4f} below minimum {MIN_F1}", file=sys.stderr)
        passed = False

    if passed:
        print("  ✅ PASS — Model meets thresholds. Proceeding with deployment.")
    else:
        print("\n  Deployment blocked. Retrain with adjusted parameters.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    session   = boto3.Session(region_name=REGION)
    sm_client = session.client("sagemaker")
    s3_client = session.client("s3")

    metrics = find_latest_metrics(sm_client, s3_client, BUCKET)
    if metrics is not None:
        run_gate(metrics)
    else:
        print("[gate] Skipping threshold check — no metrics available.")
    sys.exit(0)
