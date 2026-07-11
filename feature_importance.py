import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Create Results Folder
# ==============================

os.makedirs("results", exist_ok=True)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

# Features
X = df.drop(["Category", "Class"], axis=1)

# Load Random Forest Model
model = joblib.load("models/random_forest_model.pkl")

# Feature Importance
importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

# Save CSV
feature_df.to_csv(
    "results/feature_importance.csv",
    index=False
)

# Top 15 Features
plt.figure(figsize=(10,6))

plt.barh(
    feature_df["Feature"][:15],
    feature_df["Importance"][:15]
)

plt.gca().invert_yaxis()

plt.title("Top 15 Important Features")

plt.xlabel("Importance Score")

plt.tight_layout()

plt.savefig("results/feature_importance.png")

plt.show()

print("Feature Importance Saved Successfully")