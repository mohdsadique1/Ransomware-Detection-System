import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Loading Dataset...")

df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

print(df.shape)

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

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# CNN Input Shape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("CNN Input Shape :", X_train.shape)

# Build CNN Model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation="relu",
           input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

print("\nTraining CNN Model...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

print("Training Completed")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy*100:.2f}%")

# Probability Prediction
y_pred_prob = model.predict(X_test)

# Convert Probability to Class
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("cnn_accuracy.png", dpi=300)
plt.show()

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("cnn_loss.png", dpi=300)
plt.show()

model.save("models/cnn_model.keras")

print("CNN Model Saved Successfully")

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

metrics = pd.DataFrame({

    "Model":["CNN"],

    "Accuracy":[accuracy],

    "Precision":[precision_score(y_test,y_pred)],

    "Recall":[recall_score(y_test,y_pred)],

    "F1-Score":[f1_score(y_test,y_pred)]

})

metrics.to_csv(
    "results/cnn_metrics.csv",
    index=False
)

print("CNN Metrics Saved")