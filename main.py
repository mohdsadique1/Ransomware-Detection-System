import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ====================================================
# 1. Load Dataset
# ====================================================

print("Loading Dataset...")

df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

print("Dataset Loaded Successfully")
print("Shape :", df.shape)

# ====================================================
# 2. Data Cleaning
# ====================================================

print("\nMissing Values :", df.isnull().sum().sum())
print("Duplicate Rows :", df.duplicated().sum())

df = df.drop_duplicates()

print("New Shape :", df.shape)

# ====================================================
# 3. Encode Category
# ====================================================
original_category = df["Category"].copy()

encoder = LabelEncoder()
df["Category"] = encoder.fit_transform(df["Category"])
encoder = LabelEncoder()

df["Category"] = encoder.fit_transform(df["Category"])

# ====================================================
# 4. Features & Target
# ====================================================

X = df.drop(["Category", "Class"], axis=1)

y = df["Class"]

print("\nFeature Shape :", X.shape)
print("Class Distribution")
print(y.value_counts())

# ====================================================
# 5. Feature Scaling
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ====================================================
# 6. Train Test Split
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ====================================================
# 7. Category Distribution Graph
# ====================================================

plt.figure(figsize=(14,6))

original_category.value_counts().head(20).plot(
    kind="bar"
)


df["Category"].value_counts().plot(kind="bar")

plt.title("Malware Family Distribution")
plt.xlabel("Family")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("category_distribution.png")

plt.close()

# ====================================================
# 8. Class Distribution
# ====================================================

plt.figure(figsize=(6,4))

df["Class"].value_counts().plot(kind="bar")

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("class_distribution.png")

plt.close()

# ====================================================
# 9. Train Model
# ====================================================

print("\nTraining Model...")

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training Completed")

joblib.dump(model, "models/random_forest_model.pkl")

print("Model Saved Successfully")

joblib.dump(scaler, "models/scaler.pkl")



# ====================================================
# 10. Prediction
# ====================================================

print("\nPredicting...")

y_pred = model.predict(X_test)

print("Prediction Completed")

# ====================================================
# 11. Evaluation
# ====================================================

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix\n")

print(cm)

# ====================================================
# 12. Confusion Matrix Plot
# ====================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.close()

print("\nAll Graphs Saved Successfully")
print("Program Finished")

importance = pd.Series(
    model.feature_importances_,
    index=df.drop(["Category", "Class"], axis=1).columns
)

importance = importance.sort_values(ascending=False).head(20)

plt.figure(figsize=(10,6))

importance.plot(kind="barh")

plt.title("Top 20 Important Features")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig("figures/feature_importance.png", dpi=300)

plt.close()

report = classification_report(y_test, y_pred)

with open("results/classification_report.txt", "w") as f:
    f.write(report)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")

print("Model Saved Successfully")