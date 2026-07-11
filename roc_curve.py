import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# ======================================
# Create Results Folder
# ======================================

os.makedirs("results", exist_ok=True)

# ======================================
# Load Dataset
# ======================================

print("Loading Dataset...")

df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

# Encode Target

encoder = LabelEncoder()
df["Class"] = encoder.fit_transform(df["Class"])

# Features

X = df.drop(["Category", "Class"], axis=1)
y = df["Class"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling

scaler = joblib.load("models/scaler.pkl")

X_test = scaler.transform(X_test)

# Load Model

model = joblib.load("models/random_forest_model.pkl")

# Probability Prediction

y_score = model.predict_proba(X_test)[:,1]

# ROC

fpr, tpr, threshold = roc_curve(y_test, y_score)

roc_auc = auc(fpr, tpr)

print("AUC Score :", roc_auc)

# Plot

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

plt.plot([0,1],[0,1],"r--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("results/roc_curve.png")

plt.show()

print("ROC Curve Saved")