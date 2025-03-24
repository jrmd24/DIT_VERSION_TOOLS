# Importations nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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

# Configuration des chemins
DATA_PATH = Path('Data/creditcard.csv')
MODEL_PATH = Path('ML_Models/fraud_detection_model_V2.pkl')
REPORT_PATH = Path('reports/')

def load_data(file_path):
    """Charge les données depuis le fichier CSV"""
    df = pd.read_csv(file_path)
    print("Données chargées avec succès.")
    return df

def explore_data(df):
    """Effectue l'analyse exploratoire des données"""
    print("\nAnalyse exploratoire :")
    print(f"Dimensions des données : {df.shape}")
    print("\nDistribution des classes :")
    print(df['Class'].value_counts())
    
    # Création du rapport d'exploration
    REPORT_PATH.mkdir(exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Class', data=df)
    plt.title('Distribution des transactions frauduleuses')
    plt.savefig(REPORT_PATH/'class_distribution.png')
    plt.close()

def preprocess_data(df):
    """Prétraitement des données et préparation des ensembles d'entraînement/test"""
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Sous-échantillonnage
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=0
    )
    
    return X_train, X_test, y_train, y_test

def build_model():
    """Construit le pipeline de modèle"""
    models = [
        ('LR', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('AD', DecisionTreeClassifier()),
        ('SVM', SVC(probability=True)),
        ('bayes', GaussianNB()),
        ('GB', GradientBoostingClassifier())
    ]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('voting', VotingClassifier(models, voting='soft'))
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et génère les métriques"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraude', 'Fraude'],
                yticklabels=['Non-Fraude', 'Fraude'])
    plt.title('Matrice de confusion')
    plt.savefig(REPORT_PATH/'confusion_matrix.png')
    plt.close()
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.savefig(REPORT_PATH/'roc_curve.png')
    plt.close()

def save_model(model, path):
    """Sauvegarde le modèle entraîné"""
    path.parent.mkdir(exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModèle sauvegardé à {path}")

def main():
    # Exécution du pipeline
    df = load_data(DATA_PATH)
    explore_data(df)
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    model = build_model()
    print("\nEntraînement du modèle en cours...")
    model.fit(X_train, y_train)
    
    print("\nÉvaluation du modèle :")
    evaluate_model(model, X_test, y_test)
    
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()