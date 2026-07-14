import os
import time
import psutil
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================
# Load Dataset
# ==========================

print("Loading Dataset...")

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

# ==========================
# Load Scaler
# ==========================

scaler = joblib.load("models/scaler.pkl")
X_test = scaler.transform(X_test)

# ==========================
# Load Model
# ==========================

model = joblib.load("models/random_forest_model.pkl")

process = psutil.Process(os.getpid())

cpu_before = psutil.cpu_percent(interval=1)
memory_before = process.memory_info().rss / (1024 * 1024)

start = time.time()

prediction = model.predict(X_test)

end = time.time()

cpu_after = psutil.cpu_percent(interval=1)
memory_after = process.memory_info().rss / (1024 * 1024)

print("\n========== SYSTEM PERFORMANCE ==========")
print(f"Detection Time : {end-start:.4f} seconds")
print(f"CPU Usage Before : {cpu_before:.2f}%")
print(f"CPU Usage After  : {cpu_after:.2f}%")
print(f"Memory Before : {memory_before:.2f} MB")
print(f"Memory After  : {memory_after:.2f} MB")

# ==========================
# Save Report
# ==========================

report = pd.DataFrame({
    "Detection Time (sec)": [end-start],
    "CPU Before (%)": [cpu_before],
    "CPU After (%)": [cpu_after],
    "Memory Before (MB)": [memory_before],
    "Memory After (MB)": [memory_after]
})

report.to_csv("results/system_performance.csv", index=False)

print("\nSystem Performance Report Saved Successfully.")