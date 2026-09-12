import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")

transaction = pd.DataFrame([{
    "amount": 85000,
    "transaction_hour": 2,
    "transactions_last_hour": 15,
    "account_age_days": 100,
    "previous_failed_transactions": 7,
    "device_changed": 1,
    "location_changed": 1
}])

prediction = int(model.predict(transaction)[0])
probability = float(model.predict_proba(transaction)[0][1])

if probability >= 0.80:
    risk, decision = "HIGH", "BLOCK"
elif probability >= 0.50:
    risk, decision = "MEDIUM", "REVIEW"
else:
    risk, decision = "LOW", "APPROVE"

print("Fraud probability:", round(probability, 4))
print("Risk level:", risk)
print("Decision:", decision)
