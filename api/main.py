import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
HTML_PATH = PROJECT_ROOT / "templates" / "index.html"


# ==================================================
# LOAD MACHINE LEARNING MODEL
# ==================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Fraud model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ==================================================
# LOAD DATABASE CONNECTION
# ==================================================

try:
    from database import engine
except ImportError as exc:
    raise ImportError(
        "Could not import 'engine' from database.py. "
        "Ensure database.py exists in the project root."
    ) from exc


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Real-Time Fraud Detection System",
    description="ML-based fraud detection with SQL Server",
    version="2.0.0"
)


# ==================================================
# TRANSACTION INPUT VALIDATION
# ==================================================

class Transaction(BaseModel):
    amount: float = Field(gt=0)
    transaction_hour: int = Field(ge=0, le=23)
    transactions_last_hour: int = Field(ge=0)
    account_age_days: int = Field(ge=0)
    previous_failed_transactions: int = Field(ge=0)
    device_changed: int = Field(ge=0, le=1)
    location_changed: int = Field(ge=0, le=1)


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "Fraud Detection System",
        "model_loaded": model is not None
    }


# ==================================================
# HOME PAGE
# ==================================================

@app.get("/", response_class=HTMLResponse)
def home():
    if not HTML_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail="Frontend file templates/index.html was not found."
        )

    return HTMLResponse(
        content=HTML_PATH.read_text(encoding="utf-8")
    )


# ==================================================
# FRAUD PREDICTION
# ==================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # Prepare transaction features
    features = pd.DataFrame([{
        "amount": transaction.amount,
        "transaction_hour": transaction.transaction_hour,
        "transactions_last_hour": transaction.transactions_last_hour,
        "account_age_days": transaction.account_age_days,
        "previous_failed_transactions":
            transaction.previous_failed_transactions,
        "device_changed": transaction.device_changed,
        "location_changed": transaction.location_changed
    }])

    # Predict fraud probability
    try:
        probabilities = model.predict_proba(features)[0]

        if len(probabilities) < 2:
            raise ValueError(
                "The model does not provide two class probabilities."
            )

        fraud_probability = float(probabilities[1])

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Fraud prediction failed. Check the model and its expected features."
        ) from exc

    # Calculate risk score
    risk_score = round(fraud_probability * 100, 2)

    # Apply decision thresholds
    if fraud_probability >= 0.70:
        decision = "BLOCK"
    elif fraud_probability >= 0.30:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    # Save prediction to the database
    insert_query = text("""
        INSERT INTO transactions (
            amount,
            transaction_hour,
            transactions_last_hour,
            account_age_days,
            previous_failed_transactions,
            device_changed,
            location_changed,
            fraud_probability,
            risk_score,
            decision
        )
        VALUES (
            :amount,
            :transaction_hour,
            :transactions_last_hour,
            :account_age_days,
            :previous_failed_transactions,
            :device_changed,
            :location_changed,
            :fraud_probability,
            :risk_score,
            :decision
        )
    """)

    try:
        with engine.begin() as connection:
            connection.execute(
                insert_query,
                {
                    "amount": transaction.amount,
                    "transaction_hour": transaction.transaction_hour,
                    "transactions_last_hour":
                        transaction.transactions_last_hour,
                    "account_age_days": transaction.account_age_days,
                    "previous_failed_transactions":
                        transaction.previous_failed_transactions,
                    "device_changed": transaction.device_changed,
                    "location_changed": transaction.location_changed,
                    "fraud_probability": fraud_probability,
                    "risk_score": risk_score,
                    "decision": decision
                }
            )

        database_status = "saved"

    except SQLAlchemyError as exc:
        # Do not expose database credentials or internal error details.
        print(f"Database save failed: {type(exc).__name__}")

        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction could not be saved to the database. "
                "Check database connectivity and the transactions table."
            )
        ) from exc

    # Return prediction results
    return {
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": risk_score,
        "decision": decision,
        "database_status": database_status
    }
