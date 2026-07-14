import joblib
import shap
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================
# Load Dataset
# ==========================

print("Loading Dataset...")

df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

encoder = LabelEncoder()
df["Class"] = encoder.fit_transform(df["Class"])

X = df.drop(["Category", "Class"], axis=1)
y = df["Class"]

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Load Scaler
# ==========================

scaler = joblib.load("models/scaler.pkl")

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Load Random Forest Model
# ==========================

model = joblib.load("models/random_forest_model.pkl")

# ==========================
# SHAP Analysis
# ==========================

print("Generating SHAP Values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# ==========================
# Summary Plot
# ==========================

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns,
    show=False
)

import matplotlib.pyplot as plt

plt.tight_layout()

plt.savefig("results/shap_summary.png", dpi=300)

print("SHAP Summary Saved Successfully.")