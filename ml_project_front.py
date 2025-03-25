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

page_options = {
    ("index", "Accueil"),
    ("classification", "Classification"),
    ("regression", "Regression"),
}


@app.route("/")
def index():
    # template = env.get_template("index.html")
    context = {"page_options": page_options}
    # navigation = [("index", "Accueil"), ("regression", "Regression")]
    """context = {
        "index": "Accueil",
        "classification": "Classification",
        "regression": "Regression",
    }"""
    # context = {"navigation": navigation}

    return render_template("index.html", **context)
    # return template.render(page_options)


@app.route("/classification")
def classification():
    return render_template("index.html")


@app.route("/regression")
def regression():
    return render_template("index.html")


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
