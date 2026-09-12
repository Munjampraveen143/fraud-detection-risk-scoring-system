import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/transactions.csv")

# Features and target
X = df.drop(columns=["transaction_id", "is_fraud"])
y = df["is_fraud"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Improved Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Fraud probabilities
y_probability = model.predict_proba(X_test)[:, 1]

# Lower threshold to catch more fraud
threshold = 0.30
y_pred = (y_probability >= threshold).astype(int)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC:", round(roc_auc_score(y_test, y_probability), 4))

print(
    "PR-AUC:",
    round(average_precision_score(y_test, y_probability), 4)
)

# Save model
joblib.dump(model, "models/fraud_model.pkl")

print("\nSaved improved model to models/fraud_model.pkl")