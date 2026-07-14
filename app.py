from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "index.html",
            prediction="No File Selected"
        )

    df = pd.read_csv(file)

    # Remove unnecessary columns
    if "Category" in df.columns:
        df = df.drop(columns=["Category"])

    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    prediction = model.predict(df)

    malware = (prediction == "Malware").sum()
    benign = (prediction == "Benign").sum()

    result = f"""
    Total Samples : {len(prediction)}
    Benign : {benign}
    Malware : {malware}
    """

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)