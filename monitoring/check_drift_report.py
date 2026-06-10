"""
Drift Report Reader
===================
Downloads the latest Model Monitor report from S3 and prints a
human-readable summary of any data quality violations detected.

Called by:
  - .github/workflows/monitor.yml  (daily scheduled check)
  - Manually: python monitoring/check_drift_report.py

Environment variables:
    AWS_REGION  — e.g. ap-south-1
    S3_BUCKET   — your SageMaker bucket

Exit codes:
    0 — No violations found (or no report yet)
    1 — Violations detected (triggers CI failure / retraining alert)
"""

import os
import sys
import json
import boto3
from datetime import datetime, timezone

REGION        = os.environ["AWS_REGION"]
BUCKET        = os.environ["S3_BUCKET"]
REPORTS_PREFIX = "monitoring/reports/loan-default-endpoint-monitor"

VIOLATION_THRESHOLD = int(os.environ.get("VIOLATION_THRESHOLD", "5"))


def get_latest_report(s3_client) -> dict | None:
    """
    Scan S3 for the most recent monitoring execution report.
    Model Monitor organises reports by: <prefix>/<execution-id>/constraint_violations.json
    """
    print(f"[drift] Scanning: s3://{BUCKET}/{REPORTS_PREFIX}/")

    resp = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=REPORTS_PREFIX)
    objects = resp.get("Contents", [])

    if not objects:
        print("[drift] No monitoring reports found yet.")
        print("        (The first report appears ~1 hour after setup_model_monitor.py runs)")
        return None

    # Find the latest constraint_violations.json
    violation_files = [
        obj for obj in objects
        if obj["Key"].endswith("constraint_violations.json")
    ]

    if not violation_files:
        print("[drift] Reports exist but no violation files found — all clean or still processing.")
        return None

    latest = max(violation_files, key=lambda o: o["LastModified"])
    print(f"[drift] Latest report: {latest['Key']}")
    print(f"[drift] Generated at:  {latest['LastModified'].strftime('%Y-%m-%d %H:%M UTC')}")

    with open("/tmp/violations.json", "wb") as f:
        s3_client.download_fileobj(BUCKET, latest["Key"], f)

    with open("/tmp/violations.json") as f:
        return json.load(f)


def print_report(report: dict) -> int:
    """
    Pretty-print the violation report.
    Returns the number of violations found.
    """
    violations = report.get("violations", [])
    n = len(violations)

    print("\n" + "=" * 65)
    print("  LOANDEFAULT MODEL — DRIFT MONITORING REPORT")
    print("=" * 65)

    if n == 0:
        print("  ✅ No violations — production data matches training baseline")
        print("=" * 65)
        return 0

    print(f"  ⚠️  {n} violation(s) detected\n")

    for v in violations:
        feature      = v.get("feature_name", "unknown")
        constraint   = v.get("constraint_check_type", "unknown")
        description  = v.get("description", "")
        print(f"  Feature  : {feature}")
        print(f"  Check    : {constraint}")
        print(f"  Detail   : {description}")
        print()

    print("=" * 65)

    if n > VIOLATION_THRESHOLD:
        print(f"\n  🚨 Violations ({n}) exceed threshold ({VIOLATION_THRESHOLD})")
        print("  Recommendation: Trigger retraining pipeline")
    else:
        print(f"\n  Minor drift ({n} violations ≤ threshold {VIOLATION_THRESHOLD}) — monitor closely")

    return n


if __name__ == "__main__":
    session   = boto3.Session(region_name=REGION)
    s3_client = session.client("s3")

    report = get_latest_report(s3_client)

    if report is None:
        print("[drift] Nothing to check.")
        sys.exit(0)

    n_violations = print_report(report)

    # Exit 1 if violations exceed threshold — causes GitHub Actions to flag the run
    if n_violations > VIOLATION_THRESHOLD:
        sys.exit(1)
    else:
        sys.exit(0)
