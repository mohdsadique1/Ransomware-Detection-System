import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/model_comparison.csv")

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

for metric in metrics:

    plt.figure(figsize=(7,5))

    plt.bar(df["Model"], df[metric])

    plt.title(f"{metric} Comparison")

    plt.xlabel("Models")

    plt.ylabel(metric)

    plt.ylim(99.8,100)

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(f"results/{metric.lower()}_comparison.png")

    plt.close()

print("All Comparison Graphs Saved Successfully")