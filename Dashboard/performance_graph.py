import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results = os.path.join(BASE_DIR, "results")

files = [
    "rf_metrics.csv",
    "cnn_metrics.csv",
    "lstm_metrics.csv",
    "cnn_lstm_metrics.csv"
]

frames = []

for f in files:
    path = os.path.join(results, f)

    if os.path.exists(path):
        frames.append(pd.read_csv(path))

df = pd.concat(frames, ignore_index=True)

print(df)

plt.figure(figsize=(8,5))

plt.bar(df["Model"], df["Accuracy"]*100)

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy (%)")

plt.savefig(os.path.join(results,"accuracy_comparison.png"),dpi=300)

plt.show()

print("Graph Saved Successfully")