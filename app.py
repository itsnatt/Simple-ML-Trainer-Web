from flask import Flask, render_template, request, redirect, url_for, session
import os
import pandas as pd
from model import train_and_evaluate
from utils import process_csv
from flask import send_file

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/download/<model_name>')
def download_model(model_name):
    filepath = os.path.join(MODEL_DIR, f"{model_name.lower()}.pkl")
    if not os.path.exists(filepath):
        return "Model tidak ditemukan!", 404
    return send_file(filepath, as_attachment=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Simpan path ke session
            session["filepath"] = filepath
            df = pd.read_csv(filepath)

            return render_template("index.html", columns=df.columns, uploaded=True)
    
    return render_template("index.html", uploaded=False)
@app.route("/train", methods=["POST"])
def train():
    filepath = session.get("filepath", None)
    if not filepath:
        return redirect(url_for("index"))
    
    df = pd.read_csv(filepath)
    x_column = request.form["x_column"]
    y_column = request.form["y_column"]

    # TF-IDF Parameters
    max_features = int(request.form["max_features"])
    ngram_range = tuple(map(int, request.form["ngram_range"].split(",")))

    # Model Selection & Parameters
    model_name = request.form["model"]
    model_params = {}

    if model_name == "KNN":
        model_params["n_neighbors"] = int(request.form.get("n_neighbors", 5))

    if model_name == "SVM":
        model_params["C"] = float(request.form.get("svm_c", 1.0))

    if model_name == "Random Forest":
        model_params["n_estimators"] = int(request.form.get("rf_n_estimators", 100))

    # Train model
    report, accuracy, conf_matrix, model_filename = train_and_evaluate(df, x_column, y_column, max_features, ngram_range, model_name, model_params)

    return render_template("result.html", report=report, accuracy=accuracy, conf_matrix=conf_matrix, model=model_name, model_filename=model_filename)


if __name__ == "__main__":
    app.run(debug=True)
