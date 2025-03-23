import base64

import matplotlib.pyplot as plt
import ml_project_back as mpb
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, flash, redirect, render_template, request, url_for

app = Flask(__name__)


@app.route("/")
def index():
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
