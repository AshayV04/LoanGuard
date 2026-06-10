"""
Deploy loan default prediction model using boto3 directly.
Bypasses SageMaker Python SDK serving constraints entirely.

Steps:
  1. Find latest completed training job
  2. Download model.tar.gz from S3, inject inference.py + requirements.txt
  3. Re-upload repackaged tar to S3
  4. Clean up any existing endpoint / config / model
  5. Create new SageMaker Model → EndpointConfig → Endpoint
"""
import boto3
import os
import time
import tarfile
import tempfile
import shutil

region        = os.environ["AWS_REGION"]
role          = os.environ["SAGEMAKER_ROLE_ARN"]
endpoint_name = "loan-default-endpoint"
model_name    = "loan-default-model"

boto_session = boto3.Session(region_name=region)
sm = boto_session.client("sagemaker")
s3 = boto_session.client("s3")

# sklearn 1.2 container — supports XGBoost + scikit-learn + pandas via requirements.txt
IMAGE_URI = f"720646828776.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

print(f"Region: {region}")
print(f"Role:   {role}")

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Find latest completed training job
# ═══════════════════════════════════════════════════════════════════
print("\n[1/5] Finding latest completed training job...")

jobs = sm.list_training_jobs(
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
    raise RuntimeError("No completed training job found. Run the training pipeline first.")

print(f"  Training job: {latest_job}")

job_details    = sm.describe_training_job(TrainingJobName=latest_job)
model_artifact = job_details["ModelArtifacts"]["S3ModelArtifacts"]
print(f"  Model artifact: {model_artifact}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Repackage model.tar.gz with inference code inside
# ═══════════════════════════════════════════════════════════════════
print("\n[2/5] Repackaging model with inference code...")

bucket = model_artifact.split("/")[2]
key    = "/".join(model_artifact.split("/")[3:])

tmpdir      = tempfile.mkdtemp()
original_tar = os.path.join(tmpdir, "original.tar.gz")
extract_dir  = os.path.join(tmpdir, "contents")
new_tar      = os.path.join(tmpdir, "model.tar.gz")

s3.download_file(bucket, key, original_tar)

os.makedirs(extract_dir, exist_ok=True)
with tarfile.open(original_tar, "r:gz") as tar:
    tar.extractall(extract_dir)
print(f"  Extracted: {os.listdir(extract_dir)}")

# Remove any stale code/ directory from a previous repackage
code_dir = os.path.join(extract_dir, "code")
if os.path.exists(code_dir):
    shutil.rmtree(code_dir)
os.makedirs(code_dir)

# ── Copy inference.py from scripts/ ──────────────────────────────────────────
# inference.py is a standalone file (scripts/inference.py) — NOT written inline.
# This avoids triple-quote escaping issues and makes the file independently testable.
this_dir      = os.path.dirname(os.path.abspath(__file__))
inference_src = os.path.join(this_dir, "scripts", "inference.py")

if not os.path.exists(inference_src):
    raise FileNotFoundError(
        f"scripts/inference.py not found at: {inference_src}\n"
        "This file must exist in your repo alongside deploy_latest_model.py."
    )

shutil.copy(inference_src, os.path.join(code_dir, "inference.py"))
print("  Copied scripts/inference.py → code/inference.py")

# ── requirements.txt — installed inside the SageMaker container ───────────────
with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
    f.write(
        "xgboost>=2.0.0\n"
        "scikit-learn>=1.2.0\n"
        "joblib>=1.3.0\n"
        "pandas>=2.0.0\n"
    )
print("  Wrote code/requirements.txt")

print(f"  code/ contents: {os.listdir(code_dir)}")

# Repack and upload
with tarfile.open(new_tar, "w:gz") as tar:
    for item in os.listdir(extract_dir):
        tar.add(os.path.join(extract_dir, item), arcname=item)

new_key       = f"models/deploy/model-{int(time.time())}.tar.gz"
s3.upload_file(new_tar, bucket, new_key)
new_model_uri = f"s3://{bucket}/{new_key}"
print(f"  Uploaded: {new_model_uri}")

shutil.rmtree(tmpdir)

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Cleanup ALL existing resources
# ═══════════════════════════════════════════════════════════════════
print("\n[3/5] Cleaning up existing resources...")

# Delete endpoint (wait if in-progress)
try:
    ep     = sm.describe_endpoint(EndpointName=endpoint_name)
    status = ep["EndpointStatus"]
    print(f"  Endpoint status: {status}")

    while status in ("Creating", "Updating", "RollingBack", "Deleting"):
        print(f"  Endpoint is {status}, waiting 30s...")
        time.sleep(30)
        try:
            status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        except sm.exceptions.ClientError:
            status = "Gone"
            break

    if status != "Gone":
        sm.delete_endpoint(EndpointName=endpoint_name)
        print("  Delete requested, waiting for removal...")
        while True:
            try:
                time.sleep(15)
                sm.describe_endpoint(EndpointName=endpoint_name)
            except sm.exceptions.ClientError:
                print("  Endpoint deleted")
                break

except sm.exceptions.ClientError:
    print("  No existing endpoint")

# Delete endpoint configs matching our name prefix
try:
    configs = sm.list_endpoint_configs(NameContains="loan-default")
    for cfg in configs.get("EndpointConfigs", []):
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print(f"  Deleted config: {cfg['EndpointConfigName']}")
except Exception:
    pass

# Delete old model(s)
try:
    sm.delete_model(ModelName=model_name)
    print(f"  Deleted model: {model_name}")
except Exception:
    pass

try:
    models = sm.list_models(SortBy="CreationTime", SortOrder="Descending", MaxResults=20)
    for m in models["Models"]:
        mn = m["ModelName"]
        if "loan-default" in mn.lower():
            try:
                sm.delete_model(ModelName=mn)
                print(f"  Deleted model: {mn}")
            except Exception:
                pass
except Exception:
    pass

print("  Waiting 15s for cleanup to settle...")
time.sleep(15)

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Create SageMaker Model
# ═══════════════════════════════════════════════════════════════════
print("\n[4/5] Creating SageMaker model...")

sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": new_model_uri,
        "Environment": {
            "SAGEMAKER_PROGRAM":             "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY":    "/opt/ml/model/code",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION":              region,
        },
    },
    ExecutionRoleArn=role,
)
print(f"  Model created: {model_name}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Create endpoint config + endpoint, wait for InService
# ═══════════════════════════════════════════════════════════════════
print("\n[5/5] Creating endpoint...")

config_name = f"{endpoint_name}-config-{int(time.time())}"

sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[
        {
            "VariantName":          "primary",
            "ModelName":            model_name,
            "InstanceType":         "ml.m5.large",
            "InitialInstanceCount": 1,
        }
    ],
)
print(f"  Endpoint config: {config_name}")

sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=config_name,
)
print(f"  Endpoint creation started: {endpoint_name}")

print("  Waiting for endpoint to be InService (up to 20 min)...")
waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(
    EndpointName=endpoint_name,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40},
)

final_status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
print(f"\n{'='*60}")
print(f"  ENDPOINT STATUS : {final_status}")
print(f"  ENDPOINT NAME   : {endpoint_name}")
print(f"{'='*60}")