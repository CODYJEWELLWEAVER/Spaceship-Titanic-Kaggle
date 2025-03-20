"""
Manually implemented logistic regression classifier. I choose not to use a pre-implemented logistic regression
classifier as I wanted to solidify my understanding and gain the experience. As well as see how close I can get
to the classifiers implemented by numpy, sci-kit, etc.
"""

import numpy as np
from tqdm import tqdm
import warnings

class LogisticRegressionModel():
    """
    Binary Logistic Regression Model
    """
    def __init__(
            self,
            learning_rate=1e-3,
            silent=False,
        ):
        self.weights = None
        self.bias = None
        self.pos_label = 1
        self.neg_label = 0
        self.lr = learning_rate
        self.silent = silent


    def bce_loss(self, y_pred, y_true):
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)


    def predict(self, X):
        """ Predict class labels. """
        y_hat = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(y_hat)
        return np.where(y_hat >= 0.5, 1, 0)
    
    
    def forward(self, X):
        # compute the forward pass
        z = np.dot(self.weights, X.T) + self.bias
        A = self.sigmoid(z)
        return A

    
    def fit(self, X, y, max_iter=100, fit_bias=True):
        """
        Trains logistic regression model with given data.
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(max_iter):
            A = self.forward(X)
            dz = A - y # derivative of sigmoid and bce X.T * (A - y)
            # gradients 
            dw = (1 / n_samples) * np.dot(X.T, dz)
            if fit_bias:
                db = (1 / n_samples) * np.sum(dz)
            else:
                db = 0
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def evaluate(self, X, y):
        """ 
        Calculates accuracy and precision scores. 
        Returns:
            scores: (accuracy, precision, recall)
        """

        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)

        return accuracy


    def _positive_sigmoid(self, x):
        # prevents overflow when x >> 0
        return 1 / (1 + np.exp(-x))
    
    
    def _negative_sigmoid(self, x):
        # prevents overflow when x << 0
        return np.exp(x) / (1 + np.exp(x))
    

    def sigmoid(self, x):
        """ Sigmoid function with increased numerical stability. """
        # warning must be suppressed since the warning comes from the
        # branch which is not taken during execution.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.where(
                x >= 0,
                self._positive_sigmoid(x),
                self._negative_sigmoid(x)
            )
        
        
    def logistic_grad(self, x, y_gold):
        """ computes the gradient of logistic loss w.r.t. current weight vector """
        if y_gold == 0:
            yi = -1
        else:
            yi = 1
        z = self.sigmoid(-yi * np.dot(self.w, x))
        return -z * yi * x