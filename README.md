# 🛡️ LoanGuard: End-to-End Loan Default Prediction & MLOps Platform

[![Train Pipeline](https://github.com/AshayV04/LoanGuard/actions/workflows/train.yml/badge.svg)](https://github.com/AshayV04/LoanGuard/actions/workflows/train.yml)
[![Deploy Model](https://github.com/AshayV04/LoanGuard/actions/workflows/deploy.yml/badge.svg)](https://github.com/AshayV04/LoanGuard/actions/workflows/deploy.yml)
[![Daily Monitor](https://github.com/AshayV04/LoanGuard/actions/workflows/monitor.yml/badge.svg)](https://github.com/AshayV04/LoanGuard/actions/workflows/monitor.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![AWS](https://img.shields.io/badge/Cloud-AWS_SageMaker-orange?logo=amazonaws)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue?logo=mlflow)

An **enterprise-grade MLOps platform** designed to predict loan default risk, automatically deploy production endpoints, track experiments, and monitor live traffic for data/model drift. 

Instead of a simple offline notebook, LoanGuard implements the **entire MLOps lifecycle** on AWS, bridging the gap between statistical modeling and production engineering.

---

## 📐 Architecture & Data Flow

```
                                  GitHub Actions CI/CD (OIDC Federated Auth)
         ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
         │                                                                                             │
         │   ┌─────────────────┐       ┌─────────────────┐       ┌───────────────────────────────┐     │
         │   │    train.yml    │──────▶│   deploy.yml    │──────▶│    Quality Gate Evaluator     │     │
         │   │ (Train Pipeline)│       │ (Deploy Model)  │       │ (AUC >= 0.75 & F1 >= 0.60)    │     │
         │   └─────────────────┘       └─────────────────┘       └──────────────┬────────────────┘     │
         └──────────────────────────────────────────────────────────────────────┼──────────────────────┘
                                                                                │ pass
                                                                                ▼
 ┌──────────┐    ┌─────────────────┐    ┌────────────────────┐    ┌──────────────────────────────┐
 │  S3      │    │  SageMaker      │    │ MLflow             │    │  SageMaker Endpoint          │
 │  Bucket  │───▶│  Pipeline       │───▶│ Tracking Server    │    │  loan-default-endpoint       │
 │loans.csv │    │  XGBoost Train  │    │ (Metrics/Artifacts)│    │  ml.t2.medium  (REST API)     │
 └──────────┘    └────────┬────────┘    └────────────────────┘    └──────────────┬───────────────┘
                          │                                                      │
                          ▼                                                      ├─────────────────────┐
               ┌─────────────────────┐                                           ▼                     ▼
               │ Pandera Validation  │                                    ┌─────────────┐       ┌─────────────┐
               │ (Schema & Ranges)   │                                    │ FastAPI App │       │ SageMaker   │
               └─────────────────────┘                                    │  /predict   │       │ Model       │
                                                                          │  /history   │       │ Monitor     │
                                                                          │  /stats     │       │ (Hourly)    │
                                                                          └──────┬──────┘       └──────┬──────┘
                                                                                 │                     │
                                                                        ┌────────▼────────┐   ┌────────▼────────┐
                                                                        │ Prediction Log  │   │ S3 Drift Report │
                                                                        │  (predictions)  │   │ Violations File │
                                                                        └─────────────────┘   └────────┬────────┘
                                                                                                       │ Drift?
                                                                                                       ▼
                                                                                              ┌─────────────────┐
                                                                                              │ Auto-Retrain CI │
                                                                                              │ (Trigger Train) │
                                                                                              └─────────────────┘
```

---

## ⚡ Key MLOps Features

### 1. Robust Data Validation (Pandera)
Before starting compute jobs on AWS, the training data is validated using a structured `Pandera` schema. This checks for:
- Missing value rates and data types.
- Target column distribution stability.
- Domain-specific ranges (e.g., FICO scores between 0-1, loan amounts, age between 18-100, and debt-to-income ratios).

### 2. Imbalance-Aware Classification & MLflow Tracking
* **Dataset:** Modeled on the Home Credit Default Risk dataset, featuring heavy class imbalance (~8% default rate).
* **Modeling:** XGBoost classifier using `binary:logistic` objective with `scale_pos_weight` to address classification bias.
* **Logging:** MLflow logs hyperparameters (learning rate, tree depth, gamma) and evaluation metrics (**AUC-ROC** and **F1 Score**).

### 3. CI/CD Quality Gating
The deployment workflow runs a Python gatekeeper script before updating production services. If the model's **AUC-ROC falls below 0.75** or **F1 Score falls below 0.60**, the deployment pipeline exits with a non-zero code, blocking degraded models from going live.

### 4. Real-time REST API & Fintech Dashboard
A responsive FastAPI server acts as the middleware, managing client requests, querying the SageMaker endpoint, logging predictions in a thread-safe JSONL file, and providing a clean fintech glassmorphism UI.

### 5. Production Drift Monitoring & Closed-loop Retraining
* **Data Capture:** SageMaker Model Monitor records 100% of production traffic to Amazon S3.
* **Drift Check:** An hourly scheduled processing job compares inference traffic against the statistical baseline of training data.
* **Triggered Retraining:** A scheduled cron workflow (`monitor.yml`) inspects monitoring violations daily. If data drift exceeds the threshold, it triggers the GitHub Actions training pipeline to automatically retrain and update the model.

---

## 🗂️ Project Directory Structure

```
├── .github/workflows/
│   ├── train.yml                # Runs SageMaker training pipeline on command
│   ├── deploy.yml               # Runs evaluation check → deploys endpoint
│   └── monitor.yml              # Daily cron → reads S3 drift reports → retrains if needed
│
├── data/
│   └── prepare_loan_data.py     # Aggregates & engineers raw Kaggle CSV → clean loans.csv
│
├── scripts/
│   ├── validate_data.py         # Enforces Pandera schemas on dataset
│   ├── train_with_mlflow.py     # XGBoost training script executing on SageMaker
│   ├── evaluate_and_gate.py     # Checks performance metrics against gates (AUC, F1)
│   ├── inference.py             # Custom entry point inside model tar to handle predictions
│   ├── create_pipeline.py       # Deploys the SageMaker training pipeline structure
│   ├── run_train_deploy.py      # Script to manually execute training & deployment
│   └── requirements.txt         # Dependencies for SageMaker steps
│
├── ml-api/
│   ├── app.py                   # FastAPI REST API (predict, history, stats, health)
│   ├── logger.py                # Thread-safe JSONL prediction logger
│   ├── templates/
│   │   └── index.html           # Glassmorphism Loan Assessment Dashboard UI
│   └── requirements.txt         # Dependencies for API hosting
│
├── monitoring/
│   ├── setup_model_monitor.py   # Configures endpoint baseline, data capture, and CloudWatch
│   └── check_drift_report.py    # Reads latest S3 monitoring results for pipeline validation
│
├── deploy_latest_model.py       # Retrieves latest SageMaker artifact → packages inference → deploys
└── README.md
```

---

## 📊 Model Performance

| Metric | Target / Gate | Expected Value |
|--------|---------------|----------------|
| Algorithm | XGBoost Classifier | `binary:logistic` |
| AUC-ROC | $\ge$ 0.75 | ~0.78 |
| F1 Score | $\ge$ 0.60 | ~0.62 |
| Endpoint Target | `ml.t2.medium` | On-demand hosting |
| API Uptime | Health check monitored | 99.9% |

---

## 🚀 Setup & Usage

### Prerequisites
- AWS Account with IAM permissions for SageMaker, S3, and CloudWatch.
- GitHub repository secrets configured for CI/CD:
  * `AWS_ROLE_ARN` (OIDC IAM Role)
  * `SAGEMAKER_ROLE_ARN` (SageMaker Execution Role)
  * `S3_BUCKET` (S3 bucket name)
  * `AWS_REGION` (e.g., `ap-south-1`)
  * `MLFLOW_TRACKING_URI` (MLflow server endpoint)

### 1. Data Prep and Upload
Prepare the dataset from the Kaggle Home Credit dataset:
```bash
python data/prepare_loan_data.py
aws s3 cp data/loans.csv s3://<your-s3-bucket>/data/loans.csv
```

### 2. Verify Schema Locally
Ensure the schema validation succeeds on your processed data:
```bash
pip install -r scripts/requirements.txt
python scripts/validate_data.py --csv data/loans.csv
```

### 3. Run Pipeline manually
Deploy the training pipeline configuration:
```bash
export S3_BUCKET="your-s3-bucket"
export SAGEMAKER_ROLE_ARN="your-sagemaker-role-arn"
python scripts/create_pipeline.py
```

### 4. Run the REST API locally
Connect the FastAPI gateway to your SageMaker endpoint:
```bash
pip install -r ml-api/requirements.txt
cd ml-api
uvicorn app:app --reload --port 8000
```
Visit `http://localhost:8000` to access the Loan Assessment Dashboard.

---

## 🔌 API Reference

| Route | Method | Description |
|---|---|---|
| `GET /` | GET | Serves the Loan Risk Dashboard (HTML UI) |
| `POST /predict` | POST | Receives applicant payload → returns risk probability & risk tier |
| `GET /history` | GET | Displays last N logged prediction queries |
| `GET /stats` | GET | Shows aggregated metrics (averages, distributions, logs count) |
| `GET /health` | GET | Basic endpoint health status check |

### Example Prediction Request
```bash
curl -X POST http://localhost:8000/predict \
  -F "AMT_INCOME_TOTAL=150000" \
  -F "AMT_CREDIT=450000" \
  -F "AMT_ANNUITY=22500" \
  -F "AGE_YEARS=34" \
  -F "YEARS_EMPLOYED=5" \
  -F "EXT_SOURCE_2=0.55" \
  -F "EXT_SOURCE_3=0.62" \
  -F "NAME_EDUCATION_TYPE=Higher education" \
  -F "CODE_GENDER=M" \
  -F "FLAG_OWN_CAR=Y"
```

Response:
```json
{
  "status": "ok",
  "probability": "18.4%",
  "risk": "Low"
}
```

---

## 🛠️ Tech Stack
* **Cloud & Infrastructure:** AWS SageMaker (Pipelines, Model Monitor), S3, CloudWatch, IAM
* **Frameworks & Libraries:** XGBoost, scikit-learn, Pandera, FastAPI, Jinja2, Boto3
* **Observability & CI/CD:** MLflow, GitHub Actions (OIDC Federated Auth)
* **Language & Styling:** Python 3.11, CSS Glassmorphism

---

## 📄 License
This project is licensed under the MIT License.