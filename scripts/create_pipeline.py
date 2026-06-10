# scripts/create_pipeline.py

import os
import boto3
import sagemaker

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep

REGION = os.environ["AWS_REGION"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["S3_BUCKET"]
MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")

PIPELINE_NAME = "loan-default-pipeline"

sess = sagemaker.Session(boto3.Session(region_name=REGION))

train_s3_uri = f"s3://{BUCKET}/data/loans.csv"

estimator = SKLearn(
    entry_point="train_with_mlflow.py",
    source_dir="scripts",
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sess,
    environment={"MLFLOW_TRACKING_URI": MLFLOW_URI},
    output_path=f"s3://{BUCKET}/models/"
)

step_train = TrainingStep(
    name="TrainLoanDefaultModel",
    estimator=estimator,
    inputs={"train": TrainingInput(s3_data=train_s3_uri, content_type="text/csv")}
)

pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[step_train],
    sagemaker_session=sess
)

pipeline.upsert(role_arn=ROLE_ARN)
execution = pipeline.start()
print("Pipeline started:", execution.arn)
