from flask import Flask, render_template, request, send_file
import os
import time
import glob
import joblib
import psutil
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

# ==============================
# Project Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORT_FOLDER, exist_ok=True)
report_path = os.path.join(REPORT_FOLDER, "prediction_report.pdf")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# Load Models
# ==============================

rf_model = joblib.load(os.path.join(MODEL_FOLDER, "random_forest_model.pkl"))
cnn_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "cnn_model.keras"))
lstm_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "lstm_model.keras"))
cnn_lstm_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "cnn_lstm_model.keras"))
scaler = joblib.load(os.path.join(MODEL_FOLDER, "scaler.pkl"))

# ==============================
# Home Page
# ==============================

@app.route("/")
def home():
    return render_template("dashboard.html")

# ==============================
# Prediction
# ==============================

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    model_name = request.form["model"]
    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return "Please select a CSV file."

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
    uploaded_file.save(filepath)

    df = pd.read_csv(filepath)
    try:
        X = df.drop(["Category", "Class"], axis=1, errors="ignore")
        sample = scaler.transform(X)
    except Exception as e:
        return f"Dataset format error: {e}"

    if model_name == "Random Forest":
        probability = rf_model.predict_proba(sample)[:, 1]
        prediction = rf_model.predict(sample)
    elif model_name == "CNN":
        sample_dl = sample.reshape(sample.shape[0], sample.shape[1], 1)
        probability = cnn_model.predict(sample_dl).flatten()
        prediction = (probability > 0.5).astype(int)
    elif model_name == "LSTM":
        sample_dl = sample.reshape(sample.shape[0], sample.shape[1], 1)
        probability = lstm_model.predict(sample_dl).flatten()
        prediction = (probability > 0.5).astype(int)
    else:
        sample_dl = sample.reshape(sample.shape[0], sample.shape[1], 1)
        probability = cnn_lstm_model.predict(sample_dl).flatten()
        prediction = (probability > 0.5).astype(int)

    # Highest confidence among predictions
    confidence = round(np.max(probability) * 100, 2)
    detection_time = round(time.time() - start_time, 4)
    cpu = psutil.cpu_percent(interval=1)
    process = psutil.Process(os.getpid())
    memory = round(process.memory_info().rss / (1024 * 1024), 2)

    total_samples = len(prediction)
    benign_count = int(np.sum(prediction == 0))
    ransomware_count = int(np.sum(prediction == 1))
    if ransomware_count > benign_count:
     final_prediction = "Ransomware"
    elif benign_count > ransomware_count:
     final_prediction = "Benign"
    else:
     final_prediction = "Suspicious"
    if final_prediction == "Ransomware":
        if np.any(prediction == 1):
            confidence = round(float(np.mean(probability[prediction == 1])) * 100, 2)
        else:
            confidence = 0.0

    elif final_prediction == "Benign":
        if np.any(prediction == 0):
            confidence = round(float(np.mean(1 - probability[prediction == 0])) * 100, 2)
        else:
            confidence = 0.0

    else:
        confidence = 50.0

    history_file = os.path.join(LOG_FOLDER, "prediction_history.csv")
    history_df = pd.DataFrame({
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Uploaded File": [uploaded_file.filename],
        "Model": [model_name],
        "Prediction": [final_prediction],
        "Confidence": [confidence],
        "Detection Time": [detection_time],
        "CPU Usage": [cpu],
        "Memory (MB)": [memory],
        "Total Samples": [total_samples],
        "Benign Count": [benign_count],
        "Ransomware Count": [ransomware_count]
    })

    if os.path.exists(history_file):
        history_df.to_csv(history_file, mode="a", header=False, index=False)
    else:
        history_df.to_csv(history_file, index=False)

    return render_template(
    "dashboard.html",
    prediction=final_prediction,
    confidence=confidence,
    detection_time=detection_time,
    cpu=cpu,
    memory=memory,
    model=model_name,
    total_samples=total_samples,
    benign_count=int(benign_count),
    ransomware_count=int(ransomware_count)
)

@app.route("/history")
def history():
    history_file = os.path.join(LOG_FOLDER, "prediction_history.csv")
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        rows = df.to_dict(orient="records")
    else:
        rows = []
    return render_template("history.html", rows=rows)


@app.route("/download_csv")
def download_csv():
    history_file = os.path.join(LOG_FOLDER, "prediction_history.csv")
    if not os.path.exists(history_file):
        return "No prediction history found."
    return send_file(
        history_file,
        as_attachment=True,
        download_name="prediction_history.csv"
    )

@app.route("/download_pdf")
def download_pdf():

    history_file = os.path.join(
        LOG_FOLDER,
        "prediction_history.csv"
    )

    if not os.path.exists(history_file):
        return "No prediction history found."

    df = pd.read_csv(history_file)
    
    data = [list(df.columns)] + df.values.tolist()

    pdf = SimpleDocTemplate(report_path)

    table = Table(data)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,1), (-1,-1), colors.beige),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("BOTTOMPADDING", (0,0), (-1,0), 10)
    ]))

    pdf.build([table])

    return send_file(
        report_path,
        as_attachment=True,
        download_name="prediction_report.pdf"
    )
@app.route("/performance")
def performance():

    files = glob.glob(os.path.join(BASE_DIR, "results", "*metrics.csv"))

    rows = []

    for file in files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                rows.append(df.iloc[0].to_dict())
        except:
            pass
    return render_template(
        "performance.html",
        rows=rows
    )

@app.route("/confusion")
def confusion():
    return render_template("confusion_matrix.html")

@app.route("/about")
def about():
    return render_template("about.html")
# ==============================
# Run Application
# ==============================

if __name__ == "__main__":
    app.run(debug=True)


