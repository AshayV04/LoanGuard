"""
SageMaker Model Monitor — Setup Script
=======================================
Configures production monitoring on the live loan-default-endpoint:

  1. Enables data capture  — logs every request/response to S3
  2. Creates a baseline    — establishes "normal" distributions from training data
  3. Schedules monitoring  — hourly job comparing live traffic against baseline
  4. Creates CloudWatch alarms — alerts when drift violations exceed threshold

Run once after deployment:
    python monitoring/setup_model_monitor.py

Environment variables:
    AWS_REGION          — e.g. ap-south-1
    S3_BUCKET           — your SageMaker bucket
    SAGEMAKER_ROLE_ARN  — SageMaker execution role ARN
    SNS_ALERT_EMAIL     — (optional) email for CloudWatch alarm notifications

Architecture after setup:
    Prediction Request
          ↓
    SageMaker Endpoint  (data capture: ON)
          ↓
    s3://<bucket>/monitoring/captured-data/
          ↓
    Model Monitor Job   (runs hourly)
          ↓
    s3://<bucket>/monitoring/reports/
          ↓
    CloudWatch Alarm    (if violations > threshold)
          ↓
    SNS Email Alert     (optional)
"""

import os
import boto3
import sagemaker
from sagemaker.model_monitor import (
    DefaultModelMonitor,
    DataCaptureConfig,
    CronExpressionGenerator,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat

# ── Config ────────────────────────────────────────────────────────────────────
REGION        = os.environ["AWS_REGION"]
BUCKET        = os.environ["S3_BUCKET"]
ROLE_ARN      = os.environ["SAGEMAKER_ROLE_ARN"]
SNS_EMAIL     = os.environ.get("SNS_ALERT_EMAIL", "")

ENDPOINT_NAME = "loan-default-endpoint"

# S3 paths for monitoring artifacts
CAPTURE_URI   = f"s3://{BUCKET}/monitoring/captured-data"
BASELINE_URI  = f"s3://{BUCKET}/monitoring/baseline"
REPORTS_URI   = f"s3://{BUCKET}/monitoring/reports"
BASELINE_DATA = f"s3://{BUCKET}/data/loans.csv"   # training data as baseline reference

boto_session = boto3.Session(region_name=REGION)
sm_session   = sagemaker.Session(boto_session=boto_session)
sm_client    = boto_session.client("sagemaker")
cw_client    = boto_session.client("cloudwatch")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Enable Data Capture on the endpoint
# ─────────────────────────────────────────────────────────────────────────────

def enable_data_capture():
    """
    Update the endpoint to capture 100% of requests and responses.
    Captured data is stored in S3 and used by the monitoring job.
    """
    print("\n[1/4] Enabling data capture on endpoint...")

    # Get current endpoint config name
    ep_desc     = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    config_name = ep_desc["EndpointConfigName"]
    config_desc = sm_client.describe_endpoint_config(EndpointConfigName=config_name)

    # Create a new config with data capture enabled
    new_config_name = f"{ENDPOINT_NAME}-capture-config"

    try:
        sm_client.create_endpoint_config(
            EndpointConfigName=new_config_name,
            ProductionVariants=config_desc["ProductionVariants"],
            DataCaptureConfig={
                "EnableCapture":           True,
                "InitialSamplingPercentage": 100,   # capture 100% of traffic
                "DestinationS3Uri":        CAPTURE_URI,
                "CaptureOptions": [
                    {"CaptureMode": "Input"},   # log what goes IN
                    {"CaptureMode": "Output"},  # log what comes OUT
                ],
                "CaptureContentTypeHeader": {
                    "CsvContentTypes":  ["text/csv"],
                    "JsonContentTypes": ["application/json"],
                },
            },
        )
    except sm_client.exceptions.ResourceInUse:
        print(f"  Config {new_config_name} already exists — skipping creation")

    # Update the live endpoint to use the new config
    sm_client.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=new_config_name,
    )

    # Wait for the endpoint update to complete before proceeding
    print("  Waiting for endpoint update to complete (this takes ~5 minutes)...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=ENDPOINT_NAME,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},
    )

    print(f"  ✅ Data capture enabled → {CAPTURE_URI}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Create baseline from training data
# ─────────────────────────────────────────────────────────────────────────────

def create_baseline():
    """
    Run a SageMaker baseline job on the training dataset.
    This establishes the 'normal' statistical distributions for each feature.
    The monitoring job will compare live data against this baseline.
    """
    print("\n[2/4] Creating baseline from training data...")

    monitor = DefaultModelMonitor(
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=sm_session,
    )

    monitor.suggest_baseline(
        baseline_dataset=BASELINE_DATA,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=BASELINE_URI,
        wait=True,
        logs=False,
    )

    print(f"  ✅ Baseline statistics saved → {BASELINE_URI}/statistics.json")
    print(f"  ✅ Baseline constraints saved → {BASELINE_URI}/constraints.json")
    return monitor


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Schedule hourly monitoring job
# ─────────────────────────────────────────────────────────────────────────────

def schedule_monitoring(monitor: DefaultModelMonitor):
    """
    Schedule a recurring monitoring job that runs every hour.
    Compares captured production data against the baseline.
    Results (violation reports) are written to S3.
    """
    print("\n[3/4] Scheduling hourly monitoring job...")

    monitor.create_monitoring_schedule(
        monitor_schedule_name=f"{ENDPOINT_NAME}-monitor",
        endpoint_input=ENDPOINT_NAME,
        output_s3_uri=REPORTS_URI,
        statistics=f"{BASELINE_URI}/statistics.json",
        constraints=f"{BASELINE_URI}/constraints.json",
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,   # violations → CloudWatch metrics
    )

    print(f"  ✅ Monitoring schedule created (hourly)")
    print(f"  Reports will appear at: {REPORTS_URI}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: CloudWatch alarm on monitoring violations
# ─────────────────────────────────────────────────────────────────────────────

def create_cloudwatch_alarm():
    """
    Create a CloudWatch alarm that fires when the monitoring job reports
    more than 5 data quality violations in a single hour.
    """
    print("\n[4/4] Creating CloudWatch alarm for drift violations...")

    alarm_name = f"{ENDPOINT_NAME}-drift-alarm"
    alarm_actions = []

    # Optionally create SNS topic for email alerts
    if SNS_EMAIL:
        sns = boto_session.client("sns")
        topic_name = f"{ENDPOINT_NAME}-alerts"
        try:
            topic_arn = sns.create_topic(Name=topic_name)["TopicArn"]
            sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=SNS_EMAIL)
            alarm_actions.append(topic_arn)
            print(f"  SNS topic created — confirm subscription email sent to {SNS_EMAIL}")
        except Exception as e:
            print(f"  [WARN] Could not create SNS topic: {e}")

    cw_client.put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription=(
            "Fires when loan-default model detects > 5 data quality violations in 1 hour. "
            "Indicates input data distribution has drifted from the training baseline — "
            "model retraining may be needed."
        ),
        Namespace="aws/sagemaker/Endpoints/data-metrics",
        MetricName="feature_baseline_drift_check.violations",
        Dimensions=[{"Name": "Endpoint", "Value": ENDPOINT_NAME}],
        Period=3600,           # 1 hour window
        EvaluationPeriods=1,
        Threshold=5,
        ComparisonOperator="GreaterThanThreshold",
        Statistic="Sum",
        TreatMissingData="notBreaching",
        AlarmActions=alarm_actions,
    )

    print(f"  ✅ CloudWatch alarm created: {alarm_name}")
    print("  Fires when violations > 5 in a 1-hour window")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  LoanGuard — Model Monitor Setup")
    print(f"  Endpoint : {ENDPOINT_NAME}")
    print(f"  Bucket   : {BUCKET}")
    print(f"  Region   : {REGION}")
    print("=" * 60)

    enable_data_capture()
    monitor = create_baseline()
    schedule_monitoring(monitor)
    create_cloudwatch_alarm()

    print("\n" + "=" * 60)
    print("  ✅ Monitoring fully configured")
    print(f"  Capture  : {CAPTURE_URI}")
    print(f"  Baseline : {BASELINE_URI}")
    print(f"  Reports  : {REPORTS_URI}")
    print("=" * 60)
