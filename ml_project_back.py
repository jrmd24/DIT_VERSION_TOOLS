from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc)
from imblearn.under_sampling import RandomUnderSampler

# Configuration
DATA_PATH = Path('Data/creditcard.csv')
MODEL_DIR = Path('static/ml_models')
REPORT_DIR = Path('static/reports/')
LOG_DIR = Path('static/logs/')

# Configuration du logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR/'transactions.log',
    level=logging.INFO,
    format='%(asctime)s - %(request_id)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RequestContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'SYSTEM')
        return True

logger.addFilter(RequestContextFilter())

def generate_request_id():
    """Génère un ID unique basé sur la date/heure jusqu'à la seconde"""
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
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    return path

def load_data(file_path, request_id):
    """Charge les données avec journalisation"""
    logger.info("Chargement des données", 
                extra={'request_id': request_id})
    
    df = pd.read_csv(file_path)
    save_artifact(df, DATA_PATH.parent/'raw_data.csv', request_id)
    
    return df

def preprocess_data(df, request_id):
    """Prétraitement des données avec tracking"""
    logger.info("Prétraitement des données", 
                extra={'request_id': request_id})
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    preprocessed_data = pd.concat([X_resampled, y_resampled], axis=1)
    save_artifact(preprocessed_data, 
                 DATA_PATH.parent/'preprocessed_data.csv', 
                 request_id)
    
    return train_test_split(
        X_resampled, y_resampled, 
        test_size=0.2, 
        random_state=0
    )

def build_model(request_id):
    """Construction du modèle avec journalisation"""
    logger.info("Construction du modèle", 
               extra={'request_id': request_id})
    
    models = [
        ('LR', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('AD', DecisionTreeClassifier()),
        ('SVM', SVC(probability=True)),
        ('bayes', GaussianNB()),
        ('GB', GradientBoostingClassifier())
    ]
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('voting', VotingClassifier(models, voting='soft'))
    ])

def evaluate_model(model, X_test, y_test, request_id):
    """Évaluation et génération des rapports"""
    logger.info("Évaluation du modèle", 
               extra={'request_id': request_id})
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'roc_auc': auc(*roc_curve(y_test, y_proba)[:2])
    }
    
    save_artifact(metrics, REPORT_DIR/'metrics.json', request_id)
    
    # Visualisations
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraude', 'Fraude'],
                yticklabels=['Non-Fraude', 'Fraude'], 
                ax=ax[0])
    ax[0].set_title('Matrice de confusion')
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax[1].plot(fpr, tpr, color='darkorange', lw=2,
             label=f'AUC = {metrics["roc_auc"]:.2f}')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_xlabel('Taux faux positifs')
    ax[1].set_ylabel('Taux vrais positifs')
    ax[1].set_title('Courbe ROC')
    ax[1].legend(loc="lower right")
    
    save_artifact(fig, REPORT_DIR/'evaluation_plots.png', request_id)
    
    return metrics

def process_request():
    """Pipeline complet pour une demande"""
    request_id = generate_request_id()
    
    try:
        # Journal de début
        logger.info(f"Début traitement demande {request_id}", 
                   extra={'request_id': request_id})
        
        # Workflow principal
        df = load_data(DATA_PATH, request_id)
        X_train, X_test, y_train, y_test = preprocess_data(df, request_id)
        
        model = build_model(request_id)
        model.fit(X_train, y_train)
        
        model_path = save_artifact(
            model, 
            MODEL_DIR/'fraud_detection_model.pkl', 
            request_id
        )
        
        metrics = evaluate_model(model, X_test, y_test, request_id)
        
        # Rapport final
        report = {
            'request_id': request_id,
            'model_path': str(model_path),
            'metrics': metrics,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        save_artifact(report, REPORT_DIR/'final_report.json', request_id)
        
        logger.info(f"Traitement réussi {request_id}", 
                   extra={'request_id': request_id})
        
        return report
        
    except Exception as e:
        error_report = {
            'request_id': request_id,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        save_artifact(error_report, REPORT_DIR/'error_report.json', request_id)
        logger.error(f"Échec traitement: {str(e)}", 
                    extra={'request_id': request_id})
        return error_report

def main():
    """Point d'entrée principal"""
    result = process_request()
    
    print(f"\nRésultat du traitement [{result['request_id']}]:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()