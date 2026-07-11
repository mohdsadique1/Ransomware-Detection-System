import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, average_precision_score

os.makedirs("results", exist_ok=True)

# Load Dataset
df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

encoder = LabelEncoder()
df["Class"] = encoder.fit_transform(df["Class"])

X = df.drop(["Category", "Class"], axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load Scaler
scaler = joblib.load("models/scaler.pkl")
X_test = scaler.transform(X_test)

# Load Model
model = joblib.load("models/random_forest_model.pkl")

# Prediction Probability
y_scores = model.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_scores)

ap_score = average_precision_score(y_test, y_scores)

print("Average Precision:", ap_score)

plt.figure(figsize=(7,6))
plt.plot(recall, precision, label=f"AP = {ap_score:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results/precision_recall_curve.png")
plt.show()

print("Precision-Recall Curve Saved")