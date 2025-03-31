pipeline {
    agent {
        docker {
            image 'python:3.9'
        }
    }
    
    stages {
        stage('Setup') {
            steps {
                echo 'Installing dependencies...'
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest pytest-cov dvc'
                sh 'dvc pull'
            }
        }
        
        stage('Backend Tests') {
            steps {
                echo 'Running backend tests...'
                sh 'python -m pytest ml_project_test.py -v --cov=ml_project_back'
            }
            post {
                always {
                    echo 'Backend tests completed'
                }
                success {
                    echo 'All backend tests passed!'
                }
                failure {
                    echo 'Backend tests failed! Check logs for details.'
                }
            }
        }

        stage('Frontend Tests') {
            steps {
                echo 'Running frontend tests...'
                sh 'python -m pytest -v --cov=ml_project_front'
            }
            post {
                always {
                    echo 'Frontend tests completed'
                }
            }
        }
        
        stage('Integration Test') {
            steps {
                echo 'Running integration tests...'
                sh 'python ml_project_back.py &'
                sh 'sleep 5'  // Wait for backend to start
                sh 'python ml_project_front.py &'
                sh 'sleep 5'  // Wait for frontend to start
                sh 'curl -f http://localhost:5000/health || exit 1'  // Basic health check
            }
        }
        
        stage('Build') {
            steps {
                echo 'Building application...'
                sh 'docker build -t ml-app:latest .'
            }
        }

        stage('DVC Metrics') {
            steps {
                echo 'Checking DVC metrics...'
                sh 'dvc metrics show'
                sh 'dvc plots show'
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline succeeded! Application ready for deployment.'
        }
        failure {
            echo 'Pipeline failed! Check the logs for details.'
        }
        always {
            sh 'pkill -f "python ml_project_back.py" || true'
            sh 'pkill -f "python ml_project_front.py" || true'
        }
    }
} 