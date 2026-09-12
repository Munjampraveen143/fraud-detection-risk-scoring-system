from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from pathlib import Path

import joblib
import pandas as pd

from database import engine


# ==================================================
# LOAD ML MODEL
# ==================================================

MODEL_PATH = "models/fraud_model.pkl"

model = joblib.load(MODEL_PATH)


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Real-Time Fraud Detection System",
    description="ML-based fraud detection with SQL Server",
    version="2.0.0"
)


# ==================================================
# TRANSACTION INPUT
# ==================================================

class Transaction(BaseModel):

    amount: float = Field(gt=0)

    transaction_hour: int = Field(
        ge=0,
        le=23
    )

    transactions_last_hour: int = Field(
        ge=0
    )

    account_age_days: int = Field(
        ge=0
    )

    previous_failed_transactions: int = Field(
        ge=0
    )

    device_changed: int = Field(
        ge=0,
        le=1
    )

    location_changed: int = Field(
        ge=0,
        le=1
    )


# ==================================================
# WEBSITE HOME PAGE
# ==================================================

@app.get("/", response_class=HTMLResponse)
def home():

    html_file = Path("templates/index.html")

    return html_file.read_text(
        encoding="utf-8"
    )


# ==================================================
# FRAUD PREDICTION
# ==================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # ------------------------------------------------
    # Create DataFrame
    # ------------------------------------------------

    features = pd.DataFrame([{

        "amount": transaction.amount,

        "transaction_hour":
            transaction.transaction_hour,

        "transactions_last_hour":
            transaction.transactions_last_hour,

        "account_age_days":
            transaction.account_age_days,

        "previous_failed_transactions":
            transaction.previous_failed_transactions,

        "device_changed":
            transaction.device_changed,

        "location_changed":
            transaction.location_changed

    }])


    # ------------------------------------------------
    # ML Prediction
    # ------------------------------------------------

    fraud_probability = model.predict_proba(
        features
    )[0][1]


    # ------------------------------------------------
    # Risk Score
    # ------------------------------------------------

    risk_score = round(
        float(fraud_probability) * 100,
        2
    )


    # ------------------------------------------------
    # Business Decision
    # ------------------------------------------------

    if fraud_probability >= 0.70:

        decision = "BLOCK"

    elif fraud_probability >= 0.30:

        decision = "REVIEW"

    else:

        decision = "APPROVE"


    # ------------------------------------------------
    # Save Transaction to SQL Server
    # ------------------------------------------------

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


    with engine.begin() as connection:

        connection.execute(

            insert_query,

            {

                "amount":
                    transaction.amount,

                "transaction_hour":
                    transaction.transaction_hour,

                "transactions_last_hour":
                    transaction.transactions_last_hour,

                "account_age_days":
                    transaction.account_age_days,

                "previous_failed_transactions":
                    transaction.previous_failed_transactions,

                "device_changed":
                    transaction.device_changed,

                "location_changed":
                    transaction.location_changed,

                "fraud_probability":
                    float(fraud_probability),

                "risk_score":
                    risk_score,

                "decision":
                    decision

            }

        )


    # ------------------------------------------------
    # Return Result to Website
    # ------------------------------------------------

    return {

        "fraud_probability":
            round(
                float(fraud_probability),
                4
            ),

        "risk_score":
            risk_score,

        "decision":
            decision,

        "database_status":
            "saved"

    }