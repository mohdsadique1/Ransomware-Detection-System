import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ======================================
# Create folders if not exist
# ======================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ======================================
# Load Dataset
# ======================================

print("Loading Dataset...")

df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

print("Dataset Shape :", df.shape)

# ======================================
# Encode Target
# ======================================

encoder = LabelEncoder()
df["Class"] = encoder.fit_transform(df["Class"])

# ======================================
# Features & Target
# ======================================

X = df.drop(["Category", "Class"], axis=1)
y = df["Class"]

# ======================================
# Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ======================================
# Feature Scaling
# ======================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================
# Random Forest Model
# ======================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed")

# ======================================
# Prediction
# ======================================

y_pred = model.predict(X_test)

# ======================================
# Evaluation
# ======================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ======================================
# Save Model
# ======================================

joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved")

# ======================================
# Save Metrics
# ======================================

metrics = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1-Score": [f1]
})

metrics.to_csv(
    "results/random_forest_metrics.csv",
    index=False
)

print("Random Forest Metrics Saved")
print("\nProgram Finished Successfully")