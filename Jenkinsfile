pipeline {
    agent {
        dockerContainer {
            image 'ubuntu:22.04'
        }
    }

    triggers {
    githubPush()
  }

    environment {
        PYTHON_VERSION = '3.12'
        VENV_NAME = 'ml_project_venv'
    }
    
    stages {
        stage('Setup') {
            steps {
                echo 'Installing python and necessary dependencies...'
                sh """
                        if ! command -v python${PYTHON_VERSION} &> /dev/null; then
                            sudo apt-get update
                            sudo apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
                        fi
                    """

                // Création et activation de l'environnement virtuel
                sh """
                        python${PYTHON_VERSION} -m venv ${VENV_NAME}
                        . ${VENV_NAME}/bin/activate
                        python -m pip install --upgrade pip
                        pip install -r requirements.txt
                        pip install pytest pytest-cov
                    """
            }
        }
        
        stage('Test') {
            steps {
                echo 'Running tests...'
                sh "python ml_project_test.py -v"
            }
            post {
                always {
                    echo 'Test completed'
                }
                success {
                    echo 'All tests passed!'
                }
                failure {
                    echo 'Tests failed! Check logs for details.'
                }
            }
        }
        
        
        stage('Build') {
            steps {
                echo 'Building application...'
                sh 'docker build -t dit-g4-ml-app:latest .'
            }
        }
    }
    
    post {
         always {junit 'static/test-reports/*.xml'
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