import json
import logging
import pickle
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-interactif
import matplotlib.pyplot as plt
plt.ioff()  # Désactiver le mode interactif
import numpy as np
import pandas as pd
import seaborn as sns

# from imblearn.under_sampling import RandomUnderSampler
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

# Configuration des paths
MODEL_DIR = Path("ML_Models")
REPORT_DIR = Path("reports")
LOG_DIR = Path("static/logs")
DATA_DIR = Path("Data")

label_encoder = LabelEncoder()


# Configuration du logger personnalisé
class RequestContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "SYSTEM"
        return True


def generate_request_id():
    """Génère un ID unique basé sur la date/heure"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_artifact(data, path, request_id):
    """Sauvegarde les artefacts avec l'ID de demande"""
    path = path.with_name(f"{path.stem}_{request_id}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        elif isinstance(data, plt.Figure):
            data.savefig(path, bbox_inches="tight")
            plt.close(data)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        return path
    except Exception as e:
        backend_logger.error(f"Erreur lors de la sauvegarde de l'artefact: {str(e)}")
        raise


# Création d'un logger dédié
backend_logger = logging.getLogger("ML_Backend")
backend_logger.setLevel(logging.INFO)

# Formatter personnalisé
log_format = "%(asctime)s - %(request_id)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

# Handler pour fichier log
file_handler = logging.FileHandler(LOG_DIR / "transactions.log")
file_handler.setFormatter(formatter)
file_handler.addFilter(RequestContextFilter())

# Ajout du handler au logger
backend_logger.addHandler(file_handler)

# Configuration commune
for dir_path in [MODEL_DIR, REPORT_DIR, LOG_DIR, DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def load_data(file_path, request_id):
    """Charge les données avec typage correct"""
    backend_logger.info("Chargement des données", extra={"request_id": request_id})

    try:
        df = pd.read_csv(file_path)
        save_artifact(df, DATA_DIR / f"raw_data_{request_id}.csv", request_id)
        return df
    except Exception as e:
        backend_logger.error(
            f"Erreur de chargement: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        raise


def preprocess_data(df, request_id, target_col, model_type):
    """Prétraitement des données selon le type de modèle"""
    backend_logger.info("Prétraitement en cours...", extra={"request_id": request_id})

    try:

        if model_type == "classification":
            df[target_col] = label_encoder.fit_transform(df[target_col])

        # Séparation features/target
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Conversion des variables catégorielles
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if X[col].nunique() == 2:
                X[col] = X[col].astype("category").cat.codes
            else:
                X = pd.get_dummies(X, columns=[col], prefix=col, drop_first=True)

        # Rééchantillonnage pour la classification
        """if model_type == "classification":
            rus = RandomUnderSampler(random_state=42)
            X, y = rus.fit_resample(X, y)"""

        # Normalisation
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Sauvegarde
        processed_data = pd.concat([X_scaled, y], axis=1)
        save_artifact(
            processed_data, DATA_DIR / f"processed_{request_id}.csv", request_id
        )

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    except Exception as e:
        backend_logger.error(
            f"Échec prétraitement: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        raise


def build_model(model_name, model_type):
    """Construction du pipeline de modèle"""
    model_map = {
        "classification": {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVC": SVC(probability=True),
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "GradientBoosting": GradientBoostingClassifier(),
        },
        "regression": {
            "Linear": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(alpha=0.01),
            "SVM": SVR(),
            "RandomForest": RandomForestRegressor(n_estimators=100),
        },
    }

    return Pipeline(
        [("scaler", StandardScaler()), ("model", model_map[model_type][model_name])]
    )


def evaluate_model(model, X_test, y_test, request_id, model_type):
    """Évaluation des performances du modèle"""
    backend_logger.info("Évaluation en cours...", extra={"request_id": request_id})

    metrics = {}
    
    try:
        if model_type == "classification":
            y_pred = model.predict(X_test)

            # Métriques de base
            class_report = classification_report(y_test, y_pred, output_dict=True)
            metrics.update({
                "accuracy": accuracy_score(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": class_report,
                "precision": class_report['weighted avg']['precision'],
                "recall": class_report['weighted avg']['recall']
            })

            # Créer une nouvelle figure pour la matrice de confusion
            fig_confusion = plt.figure(figsize=(10, 8))
            print(y_test.value_counts())
            class_label = label_encoder.inverse_transform(y_test.unique())
            cm = confusion_matrix(
                label_encoder.inverse_transform(y_test),
                label_encoder.inverse_transform(y_pred),
                labels=class_label,
            )

            df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
            sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
            plt.title(f"Matrice de confusion - {model_type}")
            plt.xlabel("Prédiction")
            plt.ylabel("Valeur réelle")
            
            # Sauvegarder la figure
            confusion_path = save_artifact(fig_confusion, REPORT_DIR / f"confusion_matrix_{request_id}.png", request_id)
            plt.close(fig_confusion)

            # Créer une figure pour la distribution des prédictions
            fig_dist = plt.figure(figsize=(10, 6))
            sns.countplot(x=y_pred)
            plt.title(f"Distribution des prédictions - {model_type}")
            plt.xlabel("Classes")
            plt.ylabel("Nombre d'observations")
            
            # Sauvegarder la figure
            dist_path = save_artifact(fig_dist, REPORT_DIR / f"predictions_dist_{request_id}.png", request_id)
            plt.close(fig_dist)

            # Ajouter les chemins des figures aux métriques
            metrics["figures"] = {
                "confusion_matrix": confusion_path.name,
                "predictions_distribution": dist_path.name
            }

        else:
            y_pred = model.predict(X_test)
            metrics.update({
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2_score": r2_score(y_test, y_pred),
            })
            
            # Créer une figure pour les prédictions vs réalité
            fig_pred = plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ligne parfaite')
            plt.title(f"Prédictions vs Réalité - {model_type}")
            plt.xlabel("Valeurs réelles")
            plt.ylabel("Prédictions")
            plt.legend()
            
            # Sauvegarder la figure
            pred_path = save_artifact(fig_pred, REPORT_DIR / f"predictions_{request_id}.png", request_id)
            plt.close(fig_pred)

            # Créer une figure pour la distribution des erreurs
            fig_error = plt.figure(figsize=(10, 6))
            errors = y_pred - y_test
            sns.histplot(errors, kde=True)
            plt.title(f"Distribution des erreurs - {model_type}")
            plt.xlabel("Erreur de prédiction")
            plt.ylabel("Fréquence")
            
            # Sauvegarder la figure
            error_path = save_artifact(fig_error, REPORT_DIR / f"error_dist_{request_id}.png", request_id)
            plt.close(fig_error)

            # Ajouter les chemins des figures aux métriques
            metrics["figures"] = {
                "predictions": pred_path.name,
                "error_distribution": error_path.name
            }

        return metrics

    except Exception as e:
        backend_logger.error(
            f"Erreur lors de l'évaluation: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        plt.close('all')  # S'assurer que toutes les figures sont fermées en cas d'erreur
        raise


def process_client_request(
    client_request_id, model_type, model_name, data_file_path, target_col_name
):
    """Traite la requête d'un client et génère un modèle ML avec évaluation"""
    request_id = client_request_id
    model_file_path = Path("")

    try:
        backend_logger.info(
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
        model_file_path = save_artifact(model, MODEL_DIR / model_filename, request_id)

        metrics = evaluate_model(model, X_test, y_test, request_id, model_type)

        # Rapport final
        report = {
            "request_id": request_id,
            "client_id": client_request_id,
            "model_path": str(model_file_path),
            "metrics": metrics,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
        save_artifact(report, REPORT_DIR / f"report_{request_id}.json", request_id)

        backend_logger.info(
            f"Traitement réussi {request_id}", extra={"request_id": request_id}
        )

        return request_id, str(model_file_path.name), metrics

    except Exception as e:
        error_report = {
            "request_id": request_id,
            "client_id": client_request_id,
            "status": "failed",
            "error": str(e),
            "stack": str(traceback.print_exc()),
            "timestamp": datetime.now().isoformat(),
        }
        print(error_report)
        print(traceback.print_exc())
        save_artifact(error_report, REPORT_DIR / f"error_{request_id}.json", request_id)
        backend_logger.error(
            f"Échec traitement: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        return request_id, Path(""), {"error": str(e)}


def load_model(model_path):
    """Charge un modèle sauvegardé"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        backend_logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise


if __name__ == "__main__":
    # Exécution en mode standalone
    test_cases = [
        ("classification", "RandomForest", "Data/creditcard.csv", "Class"),
        ("regression", "RandomForest", "Data/housing.csv", "price"),
    ]

    for model_type, model_name, data_file, target in test_cases:
        client_id = f"test_{model_type}"
        process_client_request(
            client_id, model_type, model_name, Path(data_file), target
        )
