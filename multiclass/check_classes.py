import pandas as pd

df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

print(df.columns)

print("\nUnique Classes\n")

print(df["Class"].value_counts())

print("\nUnique Categories\n")

print(df["Category"].value_counts())