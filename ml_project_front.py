import base64
from pathlib import Path

import matplotlib.pyplot as plt
import ml_project_back as mpb
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, flash, redirect, render_template, request, session, url_for
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

env = Environment(
    # loader=PackageLoader("ml_project_front"),
    loader=FileSystemLoader(searchpath="./"),
    autoescape=select_autoescape(),
)

app = Flask(__name__)

app.add_url_rule(
    "/ML_Models/<path:filename>", endpoint="mlmodel", view_func=app.send_static_file
)

DATA_DIR = Path("Data/")
# Set the secret key to some random bytes. Keep this really secret! This is for session data management
app.secret_key = b"H13aV1]qi$pZ"

page_options = {
    ("regression", "Regression"),
    ("classification", "Classification"),
    ("index", "Accueil"),
}

regression_models = [
    ("Linear", "Regression Linéaire"),
    ("Ridge", "Ridge"),
    ("Lasso", "Lasso"),
    ("SVM", "SVM"),
]
classification_models = [
    ("LogisticRegression", "Régression Logistique"),
    ("SVC", "SVC"),
    ("RandomForest", "Random Forest"),
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
            uploaded_file.save(
                f"{DATA_DIR}/{uploaded_file.filename}_{current_client_id}.txt"
            )
            client_id, model_file_path, metrics = mpb.process_client_request(
                current_client_id,
                "classification",
                classification_model_selection,
                uploaded_file,
                target_column_name,
            )

            context["model_download_display"] = ""
            context["model_url"] = model_file_path
            context["metrics_display"] = [("accuracy", metrics["accuracy"])]
            context["fig_path"] = metrics["fig_path"]

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
            uploaded_file.save(
                f"{DATA_DIR}/{uploaded_file.filename}_{current_client_id}.txt"
            )
            client_id, model_file_path, metrics = mpb.process_client_request(
                current_client_id,
                "regression",
                regression_model_selection,
                uploaded_file,
                target_column_name,
            )

            context["model_download_display"] = ""
            context["model_url"] = model_file_path
            context["metrics_display"] = [
                ("RMSE", metrics["rmse"]),
                ("Mean Absolute Score", metrics["mae"]),
                ("R2 Score", metrics["r2_score"]),
            ]
            context["fig_path"] = metrics["fig_path"]

    return render_template("Regression.html", **context)


# local_css("style.css")
"""
st.markdown(
    "<h1 style='text-align: center; color:#841230;'>Groupe 4 - Outils de Versioning 2025 </h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='text-align: center; color:#841230;'>Gestion de données de Machine Learning</h2>",
    unsafe_allow_html=True,
)



page_options = (
    "Télécharger données collectées",
    "Collecter données",
    "Remplir formulaire",
    "Voir tableaux de bord",
)

regression_options = (
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "SVM",
)

classification_options = (
    "Logistic Regression",
    "Support Vector Classification",
    "Random Forest Classifier",
)

# st.sidebar.title("Action souhaitée")
# page = st.sidebar.selectbox("", page_options, key="action")

st.title("Action souhaitée")
page = st.selectbox("", page_options, key="action")


if page == page_options[0]:
    pass
elif page == page_options[1]:
    pass
elif page == page_options[2]:
    pass
else:
    pass

"""
