import joblib
import numpy as np

MODEL_PATH = "models/fraud_model.pkl"

model = joblib.load(MODEL_PATH)

print("=" * 50)
print("   REAL-TIME FRAUD DETECTION SYSTEM")
print("=" * 50)

while True:

    print("\nEnter Transaction Details")
    print("-" * 30)

    amount = float(input("Transaction amount: "))

    transaction_hour = int(
        input("Transaction hour (0-23): ")
    )

    transactions_last_hour = int(
        input("Transactions in last hour: ")
    )

    account_age_days = int(
        input("Account age in days: ")
    )

    previous_failed_transactions = int(
        input("Previous failed transactions: ")
    )

    device_changed = int(
        input("Device changed? (0 = No, 1 = Yes): ")
    )

    location_changed = int(
        input("Location changed? (0 = No, 1 = Yes): ")
    )

    # Create feature array
    features = np.array([[
        amount,
        transaction_hour,
        transactions_last_hour,
        account_age_days,
        previous_failed_transactions,
        device_changed,
        location_changed
    ]])

    # ML prediction
    fraud_probability = model.predict_proba(features)[0][1]

    risk_score = round(float(fraud_probability) * 100, 2)

    # Decision
    if fraud_probability >= 0.70:
        decision = "BLOCK"
    elif fraud_probability >= 0.30:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    print("\n" + "=" * 50)
    print("             FRAUD ANALYSIS RESULT")
    print("=" * 50)

    print(f"Fraud Probability : {fraud_probability:.4f}")
    print(f"Risk Score        : {risk_score}")
    print(f"Decision           : {decision}")

    print("=" * 50)

    again = input("\nCheck another transaction? (y/n): ")

    if again.lower() != "y":
        print("\nExiting Fraud Detection System...")
        break