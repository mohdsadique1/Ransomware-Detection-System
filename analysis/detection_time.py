import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset
df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

encoder = LabelEncoder()
df["Class"] = encoder.fit_transform(df["Class"])

X = df.drop(["Category","Class"], axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = joblib.load("models/scaler.pkl")
X_test = scaler.transform(X_test)

model = joblib.load("models/random_forest_model.pkl")

start = time.time()

prediction = model.predict(X_test)

end = time.time()

print("Detection Time :", end-start, "seconds")
print("Average Detection Time :", (end-start)/len(X_test), "seconds/sample")