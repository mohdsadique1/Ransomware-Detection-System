# 🛡️ Ransomware Detection System Using Machine Learning and Deep Learning

## 📌 Project Overview

This project is a Ransomware Detection System developed using Machine Learning and Deep Learning techniques. The system detects whether a given sample is Benign or Malware by analyzing memory-based features from the Obfuscated-MalMem2022 dataset.

The project compares the performance of four models:

- Random Forest
- CNN
- LSTM
- CNN-LSTM (Proposed Model)

---

## 📂 Dataset

Dataset Used:

**Obfuscated-MalMem2022**

Features:
- 55 Memory-based Features
- Binary Classification
- Benign
- Malware

---

## 🛠 Technologies Used

- Python
- Scikit-learn
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Flask

---

## 🤖 Models Implemented

### 1. Random Forest

- Accuracy: 99.99%

### 2. CNN

- Accuracy: 99.97%

### 3. LSTM

- Accuracy: 99.95%

### 4. CNN-LSTM (Proposed)

- Accuracy: 99.98%

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- AUC Score
- Precision-Recall Curve
- Feature Importance

---

## 📁 Project Structure

```
Ransomware Detection System
│
├── dataset/
├── models/
├── results/
├── static/
├── templates/
├── app.py
├── random_forest.py
├── cnn_model.py
├── lstm_model.py
├── cnn_lstm_model.py
├── comparison.py
├── roc_curve.py
├── feature_importance.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/mohdsadique1/Ransomware-Detection-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Flask Application

```bash
python app.py
```

Open Browser

```
http://127.0.0.1:5000
```

---

## 📈 Results

The proposed CNN-LSTM model achieved excellent performance for ransomware detection with very high Accuracy, Precision, Recall and F1-Score.

---

## 👨‍💻 Author

**Mohd Sadique**

M.Tech Computer Science & Engineering

M.Tech Research Project
