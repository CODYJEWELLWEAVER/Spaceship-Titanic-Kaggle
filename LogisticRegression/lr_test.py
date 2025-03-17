import unittest
from lr import LogisticRegressionModel
import numpy as np

class TestLogisticRegressionModel(unittest.TestCase):
    def test_sigmoid(self):
        model = LogisticRegressionModel(dim = 1, init_weights=np.array([1]))
        x = 0
        self.assertEqual(model.sigmoid(np.array([0])), 0.5)
        x = 1e10
        self.assertAlmostEqual(model.sigmoid(np.array([x])), 1)
        x = -1e10
        self.assertAlmostEqual(model.sigmoid(np.array([x])), 0)

    def test_predict(self):
        model = LogisticRegressionModel(dim=2, init_weights=np.array([1, -1]))

        x = np.array([0, 1])
        self.assertFalse(model.predict(x))

        x = np.array([1, 0])
        self.assertTrue(model.predict(x))

    def test_predict_all(self):
        model = LogisticRegressionModel(dim=2, init_weights=np.array([0, -1]))

        x = np.array([[0, 1], [2, 0]])
        self.assertListEqual(list(model.predict_all(x)), [False, True])

if __name__ == "__main__":
    unittest.main()