import base64

import matplotlib.pyplot as plt

# import ml_project_back as mpb
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, flash, redirect, render_template, request, url_for
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


@app.route("/")
def index():
    # template = env.get_template("index.html")
    context = {"page_options": page_options}

    return render_template("index.html", **context)
    # return template.render(page_options)


@app.route("/classification", methods=["GET", "POST"])
def classification():
    context = {"page_options": page_options, "model_options": classification_models}
    # if request.method == 'POST':

    return render_template("Classification.html", **context)


@app.route("/regression", methods=["GET", "POST"])
def regression():
    context = {"page_options": page_options, "model_options": regression_models}
    # if request.method == 'POST':
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
