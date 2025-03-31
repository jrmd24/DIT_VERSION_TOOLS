pipeline {
    agent {
        dockerContainer {
            image 'python:3.11'
        }
    }
    
    stages {
        stage('Setup') {
            steps {
                echo 'Installing dependencies...'
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Test') {
            steps {
                echo 'Running tests...'
                sh 'python ml_project_test.py -v'
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
         always {junit 'test-reports/*.xml'}
        
    }
} 