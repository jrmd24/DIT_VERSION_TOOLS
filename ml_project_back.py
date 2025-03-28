import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Configuration
MODEL_DIR = Path("static/ml_models")
REPORT_DIR = Path("static/reports/")
LOG_DIR = Path("static/logs/")

# Configuration du logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "transactions.log",
    level=logging.INFO,
    format="%(asctime)s - %(request_id)s - %(message)s",
)
logger = logging.getLogger(__name__)

class RequestContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, "request_id", "SYSTEM")
        return True

logger.addFilter(RequestContextFilter())

def process_client_request(
    client_id, model_type, model_name, data_file_path, target_col_name
):
    """Traite la requête d'un client et génère un modèle ML avec évaluation"""
    request_id = generate_request_id()
    model_file_path = Path("")
    model_eval_results_dict = {}

    try:
        logger.info(f"Début traitement demande {request_id}", extra={"request_id": request_id})

        # Charger les données spécifiques au client
        df = load_data(data_file_path, request_id)
        X_train, X_test, y_train, y_test = preprocess_data(df, request_id, target_col_name)

        # Construire le modèle spécifié
        model = build_model(model_name)
        model.fit(X_train, y_train)

        # Sauvegarder le modèle
        model_filename = f"{model_name}_{request_id}.pkl"
        model_file_path = MODEL_DIR / model_filename
        save_artifact(model, model_file_path, request_id)

        # Évaluer le modèle
        metrics = evaluate_model(model, X_test, y_test, request_id)

        # Générer le rapport final
        report = {
            "request_id": request_id,
            "client_id": client_id,
            "model_path": str(model_file_path),
            "metrics": metrics,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
        save_artifact(report, REPORT_DIR / f"report_{request_id}.json", request_id)

        return client_id, model_file_path, metrics

    except Exception as e:
        error_report = {
            "request_id": request_id,
            "client_id": client_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        save_artifact(error_report, REPORT_DIR / f"error_{request_id}.json", request_id)
        logger.error(f"Échec traitement: {str(e)}", extra={"request_id": request_id})
        return client_id, Path(""), {"error": str(e)}

def generate_request_id():
    """Génère un ID unique basé sur la date/heure"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_artifact(data, path, request_id):
    """Sauvegarde les artefacts avec l'ID de demande"""
    path = path.with_name(f"{path.stem}_{request_id}{path.suffix}")
    path.parent.mkdir(exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False)
    elif isinstance(data, plt.Figure):
        data.savefig(path)
        plt.close()
    else:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    return path

def load_data(file_path, request_id):
    """Charge les données avec journalisation"""
    logger.info("Chargement des données", extra={"request_id": request_id})
    df = pd.read_csv(file_path)
    save_artifact(df, Path("Data") / f"raw_data_{request_id}.csv", request_id)
    return df

def preprocess_data(df, request_id, target_col):
    """Prétraitement des données avec la colonne cible dynamique"""
    logger.info("Prétraitement des données", extra={"request_id": request_id})

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    preprocessed_data = pd.concat([X_resampled, y_resampled], axis=1)
    save_artifact(preprocessed_data, Path("Data") / f"preprocessed_data_{request_id}.csv", request_id)

    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

def build_model(model_name):
    """Construit un modèle spécifique selon le choix de l'utilisateur"""
    model_map = {
        "LogisticRegression": LogisticRegression(),
        "SVC": SVC(probability=True),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "GaussianNB": GaussianNB()
    }
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model_map.get(model_name, LogisticRegression()))  # Par défaut: Regression Logistique
    ])

def evaluate_model(model, X_test, y_test, request_id):
    """Évaluation et génération des rapports"""
    logger.info("Évaluation du modèle", extra={"request_id": request_id})

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc": auc(*roc_curve(y_test, y_proba)[:2]) if len(np.unique(y_test)) > 1 else 0.0,
    }

    save_artifact(metrics, REPORT_DIR / f"metrics_{request_id}.json", request_id)

    # Visualisations
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Fraude", "Fraude"],
        yticklabels=["Non-Fraude", "Fraude"],
        ax=ax[0],
    )
    ax[0].set_title("Matrice de confusion")

    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax[1].plot(fpr, tpr, color="darkorange", lw=2, label=f'AUC = {metrics["roc_auc"]:.2f}')
    else:
        ax[1].text(0.5, 0.5, "ROC non disponible (classe unique)", ha="center")
    ax[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax[1].set_xlabel("Taux faux positifs")
    ax[1].set_ylabel("Taux vrais positifs")
    ax[1].set_title("Courbe ROC")
    ax[1].legend(loc="lower right")

    save_artifact(fig, REPORT_DIR / f"evaluation_plots_{request_id}.png", request_id)

    return metrics

if __name__ == "__main__":
    # Exemple d'utilisation pour les tests
    client_id = "client_123"
    model_type = "classification"
    model_name = "RandomForest"
    data_file = Path("Data/creditcard.csv")
    target = "Class"

    process_client_request(client_id, model_type, model_name, data_file, target)