import boto3
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from logger import log_prediction, get_history, get_stats

# ── AWS config ────────────────────────────────────────────────────────────────
ENDPOINT_NAME = "loan-default-endpoint"
REGION        = "ap-south-1"

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

app = FastAPI(
    title="LoanGuard — Loan Default Prediction API",
    description="Serves default probability predictions from an XGBoost model on AWS SageMaker.",
    version="2.0.0",
)

templates = Jinja2Templates(directory="templates")

# Risk tier thresholds
def risk_tier(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    return "High"


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve the loan risk assessment dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


# ── Prediction ────────────────────────────────────────────────────────────────

@app.post("/predict")
def predict(
    AMT_INCOME_TOTAL:   float = Form(...),
    AMT_CREDIT:         float = Form(...),
    AMT_ANNUITY:        float = Form(...),
    AGE_YEARS:          float = Form(...),
    YEARS_EMPLOYED:     float = Form(...),
    EXT_SOURCE_2:       float = Form(...),
    EXT_SOURCE_3:       float = Form(...),
    DEBT_TO_INCOME:     float = Form(...),
    CREDIT_TO_INCOME:   float = Form(...),
    CNT_CHILDREN:       float = Form(...),
    CNT_FAM_MEMBERS:    float = Form(...),
):
    """
    Submit loan applicant features and receive a default probability + risk tier.

    Returns:
        probability: default probability as a percentage string (e.g. "34.2%")
        risk:        Low / Medium / High
        status:      ok / error
    """
    inputs = {
        "AMT_INCOME_TOTAL":  AMT_INCOME_TOTAL,
        "AMT_CREDIT":        AMT_CREDIT,
        "AMT_ANNUITY":       AMT_ANNUITY,
        "AGE_YEARS":         AGE_YEARS,
        "YEARS_EMPLOYED":    YEARS_EMPLOYED,
        "EXT_SOURCE_2":      EXT_SOURCE_2,
        "EXT_SOURCE_3":      EXT_SOURCE_3,
        "DEBT_TO_INCOME":    DEBT_TO_INCOME,
        "CREDIT_TO_INCOME":  CREDIT_TO_INCOME,
        "CNT_CHILDREN":      CNT_CHILDREN,
        "CNT_FAM_MEMBERS":   CNT_FAM_MEMBERS,
    }

    # Build CSV payload — header row + values row (SageMaker expects text/csv)
    header = ",".join(inputs.keys())
    values = ",".join(str(v) for v in inputs.values())
    payload = f"{header}\n{values}"

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload,
        )
        raw = response["Body"].read().decode().strip()

        # Parse — SageMaker XGBoost with binary:logistic returns a float (0–1)
        try:
            prob = float(raw.strip().split(",")[0])
        except (ValueError, IndexError):
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "predictions" in parsed:
                prob = parsed["predictions"][0].get("score", 0.5)
            else:
                prob = float(raw)

        pct_str = f"{prob * 100:.1f}%"
        tier    = risk_tier(prob)

        log_prediction(inputs, prediction=str(round(prob, 4)), error=None)
        return {"probability": pct_str, "risk": tier, "status": "ok"}

    except Exception as e:
        error_msg = str(e)
        log_prediction(inputs, prediction="N/A", error=error_msg)
        return JSONResponse(
            status_code=502,
            content={"probability": "N/A", "risk": "Unknown",
                     "error": error_msg, "status": "error"},
        )


# ── History & Observability ───────────────────────────────────────────────────

@app.get("/history")
def prediction_history(n: int = 20):
    """Return the last N loan assessments (default 20, max 100)."""
    return {"history": get_history(n)}


@app.get("/stats")
def prediction_stats():
    """Aggregate stats over all logged predictions."""
    return get_stats()


@app.get("/health")
def health():
    """Health check — used by load balancers and uptime monitors."""
    return {"status": "healthy", "endpoint": ENDPOINT_NAME, "region": REGION}
