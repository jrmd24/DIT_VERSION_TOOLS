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

    def test_Home(self):
        rv = self.app.get("/")
        self.assertEqual(rv.status, "200 OK")
        self.assertIn(b"Accueil", rv.data)

    def test_Classification(self):
        rv = self.app.get("/classification")
        self.assertEqual(rv.status, "200 OK")
        self.assertIn(b"Classification", rv.data)

    def test_Regression(self):
        rv = self.app.get("/regression")
        self.assertEqual(rv.status, "200 OK")
        self.assertIn(b"Regression", rv.data)


if __name__ == "__main__":
    import xmlrunner

    runner = xmlrunner.XMLTestRunner(output="test-reports")
    unittest.main(testRunner=runner)
