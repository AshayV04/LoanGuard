import os
import time
import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

REGION = os.environ["AWS_REGION"]
BUCKET = os.environ["S3_BUCKET"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]

DATA_S3 = f"s3://{BUCKET}/data/loans.csv"

# Minimal names
JOB_NAME      = f"loan-default-{int(time.time())}"
ENDPOINT_NAME = "loan-default-endpoint"

sm_session = sagemaker.Session(boto3.session.Session(region_name=REGION))
print(f"Using dataset: {DATA_S3}")
print(f"Training job: {JOB_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")

# Use SageMaker built-in XGBoost container (super minimal)
xgb = XGBoost(
    entry_point=None,
    framework_version="1.7-1",
    py_version="py3",
    instance_type="ml.m5.large",
    instance_count=1,
    role=ROLE_ARN,
    sagemaker_session=sm_session,
)

# Hyperparameters (minimal)
xgb.set_hyperparameters(
    objective="reg:squarederror",
    num_round=200,
    max_depth=5,
    eta=0.2,
    subsample=0.8,
    colsample_bytree=0.8
)

# IMPORTANT: For built-in XGBoost CSV, SageMaker expects label as first column.
# This dataset usually has quality as last column.
# So we DO one tiny conversion step using a Processing job would be proper,
# but you asked "minimum". We'll do a quick in-job conversion by creating a new S3 object.

s3 = boto3.client("s3", region_name=REGION)
tmp_local = "/tmp/loans.csv"
s3.download_file(BUCKET, "data/loans.csv", tmp_local)

# Convert: move last column to first column
import pandas as pd
df = pd.read_csv(tmp_local)
label_col = df.columns[-1]
cols = [label_col] + [c for c in df.columns if c != label_col]
df = df[cols]

converted_local = "/tmp/loans_converted.csv"
df.to_csv(converted_local, index=False)

converted_key = "data/loans_converted.csv"
s3.upload_file(converted_local, BUCKET, converted_key)
converted_s3 = f"s3://{BUCKET}/{converted_key}"
print(f"Converted dataset uploaded: {converted_s3}")

train_input = TrainingInput(
    s3_data=converted_s3,
    content_type="text/csv"
)

# Start training
xgb.fit({"train": train_input}, job_name=JOB_NAME, wait=True)

# Deploy endpoint
predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=ENDPOINT_NAME,
    wait=True
)

print("SUCCESS ✅")
print(f"Endpoint InService: {ENDPOINT_NAME}")
