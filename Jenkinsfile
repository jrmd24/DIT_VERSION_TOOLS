pipeline {
    agent any

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
                script {
                    if (isUnix()) {
                        sh """
                            if ! command -v python${PYTHON_VERSION} &> /dev/null; then
                                sudo apt-get update
                                sudo apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
                            fi
                        """
                        sh """
                            python -m venv ${VENV_NAME}
                            . ${VENV_NAME}/bin/activate
                            python -m pip install --upgrade pip
                            pip install -r requirements.txt
                        """
                    } else {
                        //# python -m venv ${VENV_NAME}
                        //#.\\${VENV_NAME}\\Scripts\\activate
                        bat """                            
                            python -m pip install --upgrade pip
                            pip install -r requirements.txt
                        """
                    }
                }
            }
        }

        stage('Test') {
            steps {
                echo 'Running tests...'
                script {
                    if (isUnix()) {
                        sh "python ml_project_test.py -v"
                    } else {
                        bat "python ml_project_test.py -v"
                    }
                }
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
                script {
                    if (isUnix()) {
                        sh 'docker build -t dit-g4-ml-app:latest .'
                    } else {
                        // Assuming Docker Desktop is installed and configured on Windows.
                        bat 'docker build -t dit-g4-ml-app:latest .'
                    }
                }
            }
        }
    }

    post {
        always {
            junit 'static/test-reports/*.xml'
            script {
                if (isUnix()) {
                    sh '''
                        pkill -f "python ml_project_back.py" || true
                        pkill -f "python ml_project_front.py" || true
                    '''
                    sh "rm -rf ${VENV_NAME}"
                } else {
                    bat '''
                        taskkill /F /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq python ml_project_back.py" || exit 0
                        taskkill /F /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq python ml_project_front.py" || exit 0
                    '''
                    //bat "deactivate"
                    //bat "rmdir /s /q ${VENV_NAME}"
                }
            }
        }
    }
}