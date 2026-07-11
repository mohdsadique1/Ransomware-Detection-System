import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Model Results
# (actual values)
# ==========================================

results = {
    "Model": [
        "Random Forest",
        "CNN",
        "LSTM",
        "CNN-LSTM"
    ],

    "Accuracy": [
        99.90,      # RF actual value
        99.97,
        99.95,      # actual value
        99.98       # actual value
    ],

    "Precision": [
        99.90,
        99.97,
        99.95,
        99.98
    ],

    "Recall": [
        99.90,
        99.97,
        99.95,
        99.98
    ],

    "F1-Score": [
        99.90,
        99.97,
        99.95,
        99.98
    ]
}

# ==========================================
# DataFrame
# ==========================================

df = pd.DataFrame(results)

print("\nModel Comparison\n")
print(df)

# Save CSV
df.to_csv("model_comparison.csv", index=False)

print("\nCSV Saved Successfully")

# ==========================================
# Accuracy Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(df["Model"], df["Accuracy"])

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy (%)")

plt.ylim(99,100)

plt.tight_layout()

plt.savefig("model_accuracy_comparison.png")

plt.show()

print("Graph Saved Successfully")

import pandas as pd
import matplotlib.pyplot as plt
import os

# ====================================
# Read all Metrics Files
# ====================================

files = [
    "results/random_forest_metrics.csv",
    "results/cnn_metrics.csv",
    "results/lstm_metrics.csv",
    "results/cnn_lstm_metrics.csv"
]

dfs = []

for file in files:
    if os.path.exists(file):
        dfs.append(pd.read_csv(file))
    else:
        print(f"Warning: {file} not found")

comparison = pd.concat(dfs, ignore_index=True)

print("\n========== MODEL COMPARISON ==========\n")
print(comparison)

# Save Combined CSV
comparison.to_csv("results/model_comparison.csv", index=False)

print("\nComparison CSV Saved Successfully")

# ====================================
# Accuracy Comparison Graph
# ====================================

plt.figure(figsize=(8,5))

plt.bar(comparison["Model"], comparison["Accuracy"])

plt.title("Model Accuracy Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.ylim(0.99,1.00)

plt.grid(axis="y")

plt.tight_layout()

plt.savefig("results/model_accuracy_comparison.png")

plt.show()

print("Graph Saved Successfully")