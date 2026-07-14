import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

# -----------------------
# Load Dataset
# -----------------------

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

scaler = joblib.load("models/scaler.pkl")
X_test = scaler.transform(X_test)

os.makedirs("static/images", exist_ok=True)

# -----------------------
# Random Forest
# -----------------------

rf = joblib.load("models/random_forest_model.pkl")
pred = rf.predict(X_test)

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Random Forest Confusion Matrix")
plt.savefig("static/images/rf_confusion_matrix.png", dpi=300)
plt.close()

# -----------------------
# CNN
# -----------------------

cnn = tf.keras.models.load_model("models/cnn_model.keras")

X_dl = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

pred = (cnn.predict(X_dl) > 0.5).astype(int)

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("CNN Confusion Matrix")
plt.savefig("static/images/cnn_confusion_matrix.png", dpi=300)
plt.close()

# -----------------------
# LSTM
# -----------------------

lstm = tf.keras.models.load_model("models/lstm_model.keras")

pred = (lstm.predict(X_dl) > 0.5).astype(int)

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("LSTM Confusion Matrix")
plt.savefig("static/images/lstm_confusion_matrix.png", dpi=300)
plt.close()

# -----------------------
# CNN-LSTM
# -----------------------

cnn_lstm = tf.keras.models.load_model("models/cnn_lstm_model.keras")

pred = (cnn_lstm.predict(X_dl) > 0.5).astype(int)

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("CNN-LSTM Confusion Matrix")
plt.savefig("static/images/cnn_lstm_confusion_matrix.png", dpi=300)
plt.close()

print("✅ All Confusion Matrix Images Generated Successfully")