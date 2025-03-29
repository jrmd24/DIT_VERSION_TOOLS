import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

# Configuration
MODEL_DIR = Path("static/ml_models")
REPORT_DIR = Path("static/reports/")
LOG_DIR = Path("static/logs/")
DATA_DIR = Path("Data/")

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

    try:
        logger.info(
            f"Début traitement demande {request_id}", extra={"request_id": request_id}
        )

        # Chargement et prétraitement
        df = load_data(data_file_path, request_id)
        X_train, X_test, y_train, y_test = preprocess_data(
            df, request_id, target_col_name, model_type
        )

        # Construction et entraînement
        model = build_model(model_name, model_type)
        model.fit(X_train, y_train)

        # Sauvegarde et évaluation
        model_filename = f"{model_name}_{request_id}.pkl"
        model_file_path = MODEL_DIR / model_filename
        save_artifact(model, model_file_path, request_id)

        metrics = evaluate_model(model, X_test, y_test, request_id, model_type)

        # Rapport final
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
    """Charge les données avec typage correct"""
    logger.info("Chargement des données", extra={"request_id": request_id})

    # Détection automatique du type de données
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(
            f"Erreur de chargement: {str(e)}", extra={"request_id": request_id}
        )
        raise

    save_artifact(df, DATA_DIR / f"raw_data_{request_id}.csv", request_id)
    return df


def preprocess_data(df, request_id, target_col, model_type):
    """Prétraitement adapté au type de modèle"""
    logger.info("Prétraitement des données", extra={"request_id": request_id})

    # Conversion générique des variables catégorielles
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].nunique() == 2 and set(df[col].unique()) == {"yes", "no"}:
            df[col] = df[col].map({"yes": 1, "no": 0})
        elif df[col].nunique() > 2:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Séparation features/target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Traitement spécifique au type de modèle
    if model_type == "classification":
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    # Normalisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_resampled), columns=X_resampled.columns
    )

    preprocessed_data = pd.concat([X_scaled, y_resampled], axis=1)
    save_artifact(
        preprocessed_data, DATA_DIR / f"preprocessed_data_{request_id}.csv", request_id
    )

    return train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)


def build_model(model_name, model_type):
    """Construction des modèles selon le type"""
    model_map = {
        "classification": {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVC": SVC(probability=True, kernel="rbf"),
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "GradientBoosting": GradientBoostingClassifier(),
            "DecisionTree": DecisionTreeClassifier(max_depth=5),
            "GaussianNB": GaussianNB(),
        },
        "regression": {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=0.5),
            "Lasso": Lasso(alpha=0.01),
            "SVM": SVR(kernel="rbf", C=100, gamma=0.1),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
        },
    }

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                model_map[model_type].get(
                    model_name,
                    (
                        LogisticRegression()
                        if model_type == "classification"
                        else LinearRegression()
                    ),
                ),
            ),
        ]
    )


def evaluate_model(model, X_test, y_test, request_id, model_type):
    """Évaluation adaptée au type de modèle"""
    logger.info("Évaluation du modèle", extra={"request_id": request_id})

    metrics = {}
    fig = plt.figure(figsize=(12, 8))

    if model_type == "classification":
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            metrics["roc_auc"] = auc(fpr, tpr)

            # Courbe ROC
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f'ROC curve (area = {metrics["roc_auc"]:.2f})',
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Courbe ROC")
            plt.legend(loc="lower right")

        # Matrice de confusion
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Classe 0", "Classe 1"],
            yticklabels=["Classe 0", "Classe 1"],
        )
        plt.title("Matrice de confusion")

    else:  # Régression
        y_pred = model.predict(X_test)

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred),
        }

        # Visualisation des prédictions
        plt.scatter(y_test, y_pred)
        plt.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2
        )
        plt.xlabel("Valeurs Réelles")
        plt.ylabel("Prédictions")
        plt.title("Prédictions vs Réalité")

    save_artifact(fig, REPORT_DIR / f"eval_plot_{request_id}.png", request_id)
    save_artifact(metrics, REPORT_DIR / f"metrics_{request_id}.json", request_id)

    metrics["fig_path"] = REPORT_DIR / f"eval_plot_{request_id}.png"

    return metrics


if __name__ == "__main__":
    # Exemple d'utilisation
    test_cases = [
        ("classification", "RandomForest", "Data/creditcard.csv", "Class"),
        ("regression", "RandomForest", "Data/housing.csv", "price"),
    ]

    for model_type, model_name, data_file, target in test_cases:
        client_id = f"test_{model_type}"
        process_client_request(
            client_id, model_type, model_name, Path(data_file), target
        )
