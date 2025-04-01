import os
import unittest
from pathlib import Path

import ml_project_back as mpb
import ml_project_front as mpf
import numpy as np
import pandas as pd

# from ml_project_front import app as ml_app


class TestMLPlatform(unittest.TestCase):
    def setUp(self):
        mpf.app.testing = True
        self.app = mpf.app.test_client()

        # Backend Tests
        # Create test datasets for both regression and classification
        self.test_data_dir = Path("Data")
        self.test_data_dir.mkdir(exist_ok=True)

        # Create regression dataset
        np.random.seed(42)
        X_reg = np.random.rand(100, 3)
        y_reg = (
            X_reg[:, 0] * 2
            + X_reg[:, 1] * 1.5
            + X_reg[:, 2] * 0.5
            + np.random.normal(0, 0.1, 100)
        )

        self.reg_df = pd.DataFrame(X_reg, columns=["feature1", "feature2", "feature3"])
        self.reg_df["target"] = y_reg

        self.reg_file_path = self.test_data_dir / "test_regression.csv"
        self.reg_df.to_csv(self.reg_file_path, index=False)

        # Create classification dataset
        X_class = np.random.rand(100, 3)
        y_class = (X_class[:, 0] + X_class[:, 1] > 1).astype(int)

        self.class_df = pd.DataFrame(
            X_class, columns=["feature1", "feature2", "feature3"]
        )
        self.class_df["target"] = y_class

        self.class_file_path = self.test_data_dir / "test_classification.csv"
        self.class_df.to_csv(self.class_file_path, index=False)

        self.request_id = "test_request"

    def test_Home(self):
        rv = self.app.get("/")
        self.assertEqual(rv.status, "200 OK")
        self.assertIn(b"Accueil", rv.data)

    def tearDown(self):
        # Clean up test files
        if self.reg_file_path.exists():
            self.reg_file_path.unlink()
        if self.class_file_path.exists():
            self.class_file_path.unlink()

        # Clean up model files
        model_dir = Path("ML_Models")
        if model_dir.exists():
            for model_file in model_dir.glob(f"{self.request_id}*"):
                model_file.unlink()

    def test_load_data(self):
        """Test if data is loaded correctly for both types"""
        # Test regression data
        df_reg = mpb.load_data(self.reg_file_path, self.request_id)
        self.assertEqual(len(df_reg), 100)
        self.assertEqual(
            list(df_reg.columns), ["feature1", "feature2", "feature3", "target"]
        )

        # Test classification data
        df_class = mpb.load_data(self.class_file_path, self.request_id)
        self.assertEqual(len(df_class), 100)
        self.assertTrue(df_class["target"].isin([0, 1]).all())

    def test_preprocess_data(self):
        """Test data preprocessing for both types"""
        # Test regression preprocessing
        df_reg = mpb.load_data(self.reg_file_path, self.request_id)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = mpb.preprocess_data(
            df_reg, self.request_id, "target", "regression"
        )

        # Test classification preprocessing
        df_class = mpb.load_data(self.class_file_path, self.request_id)
        X_train_class, X_test_class, y_train_class, y_test_class = mpb.preprocess_data(
            df_class, self.request_id, "target", "classification"
        )

        # Check split sizes and scaling for both
        for X_train in [X_train_reg, X_train_class]:
            self.assertTrue(np.abs(X_train.mean()).max() < 0.1)
            self.assertTrue(np.abs(X_train.std() - 1).max() < 0.1)

    def test_build_model(self):
        """Test model building for all supported algorithms"""
        regression_models = ["Linear", "Ridge", "Lasso", "SVM", "RandomForest"]
        classification_models = [
            "LogisticRegression",
            "SVC",
            "RandomForest",
            "GradientBoosting",
        ]

        for model_name in regression_models:
            model = mpb.build_model(model_name, "regression")
            self.assertIsNotNone(model)

        for model_name in classification_models:
            model = mpb.build_model(model_name, "classification")
            self.assertIsNotNone(model)

    def test_regression_pipeline(self):
        """Test complete regression pipeline with different models"""
        for model_name in ["Linear", "RandomForest"]:
            request_id, model_path, metrics = mpb.process_client_request(
                self.request_id, "regression", model_name, self.reg_file_path, "target"
            )

            self.assertEqual(request_id, self.request_id)
            model_file = mpb.MODEL_DIR / model_path
            self.assertTrue(model_file.exists())
            self.assertIn("r2_score", metrics)
            self.assertIn("rmse", metrics)
            self.assertIn("mae", metrics)
            self.assertTrue(metrics["r2_score"] > 0)  # Should be better than random

    def test_classification_pipeline(self):
        """Test complete classification pipeline with different models"""
        for model_name in ["LogisticRegression", "RandomForest"]:
            request_id, model_path, metrics = mpb.process_client_request(
                self.request_id,
                "classification",
                model_name,
                self.class_file_path,
                "target",
            )

            self.assertEqual(request_id, self.request_id)
            model_file = mpb.MODEL_DIR / model_path
            self.assertTrue(model_file.exists())
            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertTrue(metrics["accuracy"] > 0.5)  # Should be better than random

    def test_model_persistence(self):
        """Test if models are correctly saved and loaded"""
        # Train and save a model
        request_id, model_path, _ = mpb.process_client_request(
            self.request_id, "regression", "Linear", self.reg_file_path, "target"
        )

        # Check if model file exists
        model_file = mpb.MODEL_DIR / model_path
        self.assertTrue(model_file.exists())

        # Try to load and use the model
        loaded_model = mpb.load_model(model_file)
        self.assertIsNotNone(loaded_model)

        # Test prediction with loaded model
        X_test = np.random.rand(5, 3)
        predictions = loaded_model.predict(X_test)
        self.assertEqual(len(predictions), 5)


if __name__ == "__main__":
    import xmlrunner

    runner = xmlrunner.XMLTestRunner(output="static/test-reports")
    unittest.main(testRunner=runner)
