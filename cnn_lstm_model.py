import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Load Dataset
# ======================

print("Loading Dataset...")

df = pd.read_csv("dataset/Obfuscated-MalMem2022.csv")

# Encode Class
encoder = LabelEncoder()
df["Class"] = encoder.fit_transform(df["Class"])

X = df.drop(["Category", "Class"], axis=1)
y = df["Class"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ======================
# CNN + LSTM
# ======================

model = Sequential()

model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        input_shape=(55,1)
    )
)

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(64))

model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[early]
)

loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy :", accuracy)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

model.save("models/cnn_lstm_model.keras")

print("CNN-LSTM Model Saved Successfully")

metrics = pd.DataFrame({

    "Model":["CNN-LSTM"],

    "Accuracy":[accuracy],

    "Precision":[precision_score(y_test,y_pred)],

    "Recall":[recall_score(y_test,y_pred)],

    "F1-Score":[f1_score(y_test,y_pred)]

})

metrics.to_csv(
    "results/cnn_lstm_metrics.csv",
    index=False
)