import base64
from pathlib import Path

import matplotlib.pyplot as plt
import ml_project_back as mpb
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, flash, redirect, render_template, request, session, url_for, send_from_directory
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

env = Environment(
    # loader=PackageLoader("ml_project_front"),
    loader=FileSystemLoader(searchpath="./"),
    autoescape=select_autoescape(),
)

app = Flask(__name__, static_folder='static')

# Configuration des dossiers pour les rapports et modèles
@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory('reports', filename)

@app.route('/ml_models/<path:filename>')
def serve_model(filename):
    return send_from_directory('ML_Models', filename)

DATA_DIR = Path("Data/")
# Set the secret key to some random bytes. Keep this really secret! This is for session data management
app.secret_key = b"H13aV1]qi$pZ"

page_options = {
    ("regression", "Régression"),
    ("classification", "Classification"),
    ("index", "Accueil"),
}

regression_models = [
    ("Linear", "Régression Linéaire"),
    ("Ridge", "Ridge"),
    ("Lasso", "Lasso"),
    ("SVM", "SVM"),
    ("RandomForest", "Random Forest"),
]
classification_models = [
    ("LogisticRegression", "Régression Logistique"),
    ("SVC", "SVC"),
    ("RandomForest", "Random Forest"),
    ("GradientBoosting", "Gradient Boosting"),
]


def get_session_id():
    if "client_id" in session:
        flash(f'Your session is identified as n° {session["client_id"]}')
    else:
        session["client_id"] = mpb.generate_request_id()

    return session["client_id"]


@app.route("/")
def index():

    # template = env.get_template("index.html")
    context = {"page_options": page_options}

    return render_template("index.html", **context)
    # return template.render(page_options)


@app.route("/classification", methods=["GET", "POST"])
def classification():
    context = {"page_options": page_options, "model_options": classification_models}
    context["model_download_display"] = "hidden"
    if request.method == "POST":
        current_client_id = get_session_id()
        classification_model_selection = request.form["modelesml"]
        target_column_name = request.form["targetcolumn"]
        uploaded_file = request.files["initialDataFile"]
        if uploaded_file.filename != "":
            client_data_file_path = (
                f"{DATA_DIR}/{uploaded_file.filename}_{current_client_id}.txt"
            )

            uploaded_file.save(client_data_file_path)
            client_id, model_file_name, metrics = mpb.process_client_request(
                current_client_id,
                "classification",
                classification_model_selection,
                client_data_file_path,
                target_column_name,
            )

            context["model_download_display"] = ""
            context["model_filename"] = model_file_name
            context["metrics_display"] = [
                ("Accuracy", metrics["accuracy"]),
                ("Precision", metrics["precision"]),
                ("Recall", metrics["recall"])
            ]
            context["figures"] = [
                ("Matrice de confusion", metrics['figures']['confusion_matrix']),
                ("Distribution des prédictions", metrics['figures']['predictions_distribution'])
            ]

    return render_template("Classification.html", **context)


@app.route("/regression", methods=["GET", "POST"])
def regression():
    context = {"page_options": page_options, "model_options": regression_models}
    context["model_download_display"] = "hidden"
    if request.method == "POST":
        current_client_id = get_session_id()
        regression_model_selection = request.form["modelesml"]
        target_column_name = request.form["targetcolumn"]
        uploaded_file = request.files["initialDataFile"]
        if uploaded_file.filename != "":
            client_data_file_path = (
                f"{DATA_DIR}/{uploaded_file.filename}_{current_client_id}.txt"
            )

            uploaded_file.save(client_data_file_path)
            client_id, model_file_name, metrics = mpb.process_client_request(
                current_client_id,
                "regression",
                regression_model_selection,
                client_data_file_path,
                target_column_name,
            )

            context["model_download_display"] = ""
            context["model_filename"] = model_file_name
            context["metrics_display"] = [
                ("RMSE", metrics["rmse"]),
                ("Mean Absolute Score", metrics["mae"]),
                ("R2 Score", metrics["r2_score"]),
            ]
            context["figures"] = [
                ("Prédictions vs Réalité", metrics['figures']['predictions']),
                ("Distribution des erreurs", metrics['figures']['error_distribution'])
            ]

    return render_template("Regression.html", **context)
