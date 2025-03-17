"""
Manually implemented logistic regression classifier. I choose not to use a pre-implemented logistic regression
classifier as I wanted to solidify my understanding and gain the experience. As well as see how close I can get
to the classifiers implemented by numpy, sci-kit, etc.
"""

import numpy as np
from tqdm import tqdm
import warnings

class LogisticRegressionModel():
    def __init__(
            self, 
            dim, 
            init_weights=None, 
            class_labels=[False, True], 
            learning_rate=1e-3
        ):
        self.dim = dim
        self.w = np.zeros(dim) if init_weights is None else init_weights
        self.class_labels = class_labels
        self.lr = learning_rate
        self.pos_index = 1
        self.neg_index = 0


    def predict(self, example):
        """ Predict class label of example. """
        if example.shape != self.w.shape:
            raise ValueError("Incorrect example shape %d" % example.shape)
        return np.where(
            np.dot(self.w, example) >= 0,
            self.class_labels[self.pos_index],
            self.class_labels[self.neg_index]
        )
        
    
    def predict_all(self, examples):
        """ Predicts class labels of multiple examples. """
        return np.array([pred for pred in map(self.predict, examples)])
    
    
    def fit_model(self, examples, labels, max_iter=1000):
        """
        Trains logistic regression model with given data using 
        stochastic gradient descent and squared error.
        Params:
            data: list of tuples of the form (x, y)
            max_iter: maximum number of samples to use during training
        """
        data = zip(examples, labels)
        rng = np.random.default_rng()

        training_loop = tqdm(range(max_iter), "Fitting LR Model")
        for _ in training_loop:
            # sample data
            x, y = rng.choice(data)
            # compute gradient 
            grad = self.logistic_grad(x, y)
            # perform gradient descent w.r.t. (x, y)
            w -= self.lr * grad


    def evaluate(self, examples, labels):
        """ 
        Calculates accuracy and precision scores. 
        Returns:
            scores: (accuracy_score, precision_score)
        """

        predictions = self.predict_all(examples)
        N = predictions.shape[0]
        accuracy = np.mean(predictions == labels)
        FP = ((predictions == True) and (labels != self.class_labels[])).sum()
        TP = ((predictions == True) and (labels == 1)).sum()
        precision = TP / (FP + TP)

        return (accuracy, precision)


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
            warnings.simplefilter('ignore')
            return np.where(
                x >= 0,
                self._positive_sigmoid(x),
                self._negative_sigmoid(x)
            )
        
        
    def logistic_grad(self, x, y_gold):
        # computes the gradient of squared loss w.r.t. weights for example x
        z = self.sigmoid(np.dot(self.w, x))
        return (z - y_gold) * (z * (1 - z)) * x