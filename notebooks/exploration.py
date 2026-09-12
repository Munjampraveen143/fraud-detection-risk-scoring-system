import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/transactions.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nFraud distribution:\n", df["is_fraud"].value_counts())

df["is_fraud"].value_counts().plot(kind="bar")
plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction type (0=legitimate, 1=fraud)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
