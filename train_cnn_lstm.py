import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =============================
# Paths
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(BASE_DIR, "dataset", "Obfuscated-MalMem2022.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================
# Load Dataset
# =============================

print("Loading Dataset...")

df = pd.read_csv(DATASET)

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

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("Train Shape :", X_train.shape)

# =============================
# CNN-LSTM Model
# =============================

model = Sequential()

model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train.shape[1], 1)
    )
)

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(64))

model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# Train
# =============================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

# =============================
# Save Model
# =============================

model.save(
    os.path.join(
        MODEL_DIR,
        "cnn_lstm_model.keras"
    )
)

print("CNN-LSTM Model Saved")

# =============================
# Prediction
# =============================

prob = model.predict(X_test)

pred = (prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print(classification_report(y_test, pred))

print(confusion_matrix(y_test, pred))

# =============================
# Save Metrics
# =============================

metrics = pd.DataFrame({

    "Model":["CNN-LSTM"],

    "Accuracy":[accuracy],

    "Precision":[precision],

    "Recall":[recall],

    "F1-Score":[f1]

})

metrics.to_csv(

    os.path.join(

        RESULT_DIR,

        "cnn_lstm_metrics.csv"

    ),

    index=False

)

print("Metrics Saved")

# =============================
# Accuracy Graph
# =============================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"],label="Train")

plt.plot(history.history["val_accuracy"],label="Validation")

plt.legend()

plt.title("CNN-LSTM Accuracy")

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "cnn_lstm_accuracy.png"
    ),
    dpi=300
)

plt.close()

# =============================
# Loss Graph
# =============================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"],label="Train")

plt.plot(history.history["val_loss"],label="Validation")

plt.legend()

plt.title("CNN-LSTM Loss")

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "cnn_lstm_loss.png"
    ),
    dpi=300
)

plt.close()

print("Training Completed Successfully")