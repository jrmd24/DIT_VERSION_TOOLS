pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        VENV_NAME = 'ml_project_venv'
    }
    
    stages {
        stage('Préparation Environnement') {
            steps {
                script {
                    // Installation de Python et virtualenv si nécessaire
                    sh '''
                        if ! command -v python${PYTHON_VERSION} &> /dev/null; then
                            sudo apt-get update
                            sudo apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
                        fi
                    '''
                    
                    // Création et activation de l'environnement virtuel
                    sh """
                        python${PYTHON_VERSION} -m venv ${VENV_NAME}
                        . ${VENV_NAME}/bin/activate
                        python -m pip install --upgrade pip
                        pip install -r requirements.txt
                        pip install pytest pytest-cov dvc
                    """
                }
            }
        }
        
        stage('DVC Pull') {
            steps {
                script {
                    sh """
                        . ${VENV_NAME}/bin/activate
                        dvc pull
                    """
                }
            }
        }
        
        stage('Tests Backend') {
            steps {
                script {
                    sh """
                        . ${VENV_NAME}/bin/activate
                        python -m pytest ml_project_test.py -v --cov=ml_project_back
                    """
                }
            }
            post {
                always {
                    echo 'Tests backend terminés'
                }
                success {
                    echo 'Tous les tests backend ont réussi !'
                }
                failure {
                    echo 'Échec des tests backend ! Vérifiez les logs.'
                }
            }
        }

        stage('Tests Frontend') {
            steps {
                script {
                    sh """
                        . ${VENV_NAME}/bin/activate
                        python -m pytest -v --cov=ml_project_front
                    """
                }
            }
            post {
                always {
                    echo 'Tests frontend terminés'
                }
            }
        }
        
        stage('Tests Intégration') {
            steps {
                script {
                    sh """
                        . ${VENV_NAME}/bin/activate
                        python ml_project_back.py &
                        sleep 5  # Attente du démarrage du backend
                        python ml_project_front.py &
                        sleep 5  # Attente du démarrage du frontend
                        curl -f http://localhost:5000/health || exit 1  # Vérification santé
                    """
                }
            }
        }
        
        stage('Métriques DVC') {
            steps {
                script {
                    sh """
                        . ${VENV_NAME}/bin/activate
                        dvc metrics show
                        dvc plots show
                    """
                }
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline réussi ! Application prête pour le déploiement.'
        }
        failure {
            echo 'Pipeline échoué ! Vérifiez les logs.'
        }
        always {
            script {
                // Nettoyage des processus
                sh '''
                    pkill -f "python ml_project_back.py" || true
                    pkill -f "python ml_project_front.py" || true
                '''
                
                // Nettoyage de l'environnement virtuel
                sh "rm -rf ${VENV_NAME}"
            }
        }
    }
} 