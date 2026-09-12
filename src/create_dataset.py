import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Number of transactions
N = 5000

# -----------------------------
# Generate transaction features
# -----------------------------

transaction_id = np.arange(100000, 100000 + N)

amount = np.round(
    np.random.lognormal(mean=3.5, sigma=1.0, size=N),
    2
)

transaction_hour = np.random.randint(0, 24, N)

transactions_last_hour = np.random.poisson(
    lam=2.5,
    size=N
)

account_age_days = np.random.randint(
    10,
    3000,
    N
)

previous_failed_transactions = np.random.poisson(
    lam=0.7,
    size=N
)

device_changed = np.random.binomial(
    1,
    0.12,
    N
)

location_changed = np.random.binomial(
    1,
    0.10,
    N
)

# -----------------------------
# Create fraud risk score
# -----------------------------

risk_score = (
    -5.0

    # Large transaction
    + 0.012 * amount

    # Late-night transactions
    + np.where(
        (transaction_hour <= 5) |
        (transaction_hour >= 23),
        1.2,
        0
    )

    # Transaction velocity
    + 0.45 * transactions_last_hour

    # Failed transactions
    + 0.9 * previous_failed_transactions

    # Device change
    + 1.8 * device_changed

    # Location change
    + 1.6 * location_changed

    # Young accounts are riskier
    + np.where(
        account_age_days < 90,
        1.5,
        0
    )

    # Very large transaction
    + np.where(
        amount > 150,
        2.0,
        0
    )

    # Multiple suspicious signals together
    + 2.0 * (
        device_changed * location_changed
    )

    + 1.5 * (
        (transactions_last_hour >= 6) *
        (previous_failed_transactions >= 2)
    )
)

# Convert risk score to probability
fraud_probability = 1 / (
    1 + np.exp(-risk_score)
)

# Generate fraud labels
is_fraud = np.random.binomial(
    1,
    fraud_probability
)

# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "transaction_id": transaction_id,
    "amount": amount,
    "transaction_hour": transaction_hour,
    "transactions_last_hour": transactions_last_hour,
    "account_age_days": account_age_days,
    "previous_failed_transactions": previous_failed_transactions,
    "device_changed": device_changed,
    "location_changed": location_changed,
    "is_fraud": is_fraud
})

# Save dataset
df.to_csv(
    "data/transactions.csv",
    index=False
)

print("Dataset created successfully!")
print(f"Total transactions: {len(df)}")
print(f"Fraud transactions: {df['is_fraud'].sum()}")
print(
    f"Fraud percentage: "
    f"{df['is_fraud'].mean() * 100:.2f}%"
)

print("\nFirst 5 rows:")
print(df.head())