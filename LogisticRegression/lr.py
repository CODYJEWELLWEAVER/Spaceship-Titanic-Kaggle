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
            dim, 
            init_weights=None, 
            learning_rate=1e-4
        ):
        self.dim = dim
        self.w = np.zeros(dim) if init_weights is None else init_weights
        self.pos_label = 1
        self.neg_label = 0
        self.lr = learning_rate


    def predict(self, example):
        """ Predict class label of example. """
        if example.shape != self.w.shape:
            raise ValueError("Incorrect example shape %d" % example.shape)
        return np.where(
            np.dot(self.w, example) >= 0,
            self.pos_label,
            self.neg_label
        )
        
    
    def predict_all(self, examples):
        """ Predicts class labels of multiple examples. """
        return np.array([pred for pred in map(self.predict, examples)])
    
    
    def fit_model(self, examples, labels, num_epochs=10):
        """
        Trains logistic regression model with given data using 
        stochastic gradient descent and squared error.
        Params:
            data: list of tuples of the form (x, y)
            max_iter: maximum number of samples to use during training
        """
        indices = [idx for idx in range(len(examples))]
        rng = np.random.default_rng()

        for epoch in range(num_epochs):
            rng.shuffle(indices)

            epoch_loss = 0
            for i in tqdm(range(len(indices)), 'Epoch: %d' % epoch):
                idx = indices[i]
                x = examples[idx]
                y = labels[idx]

                # compute gradient 
                grad = self.logistic_grad(x, y)

                # perform gradient descent w.r.t. (x, y)
                self.w -= self.lr * grad

                ex_loss = np.log(1 + np.exp(-y * np.dot(self.w, x)))
                epoch_loss += ex_loss

            avg_loss = epoch_loss / len(indices)
            print('Average Log Loss for epoch %d: %.2f' % (epoch, avg_loss))


    def evaluate(self, examples, labels):
        """ 
        Calculates accuracy and precision scores. 
        Returns:
            scores: (accuracy, precision, recall)
        """

        predictions = self.predict_all(examples)
        accuracy = np.mean(predictions == labels)

        TP = ((predictions == 1) & (labels == 1)).sum()
        FP = ((predictions == 1) & (labels == 0)).sum()
        precision = TP / (TP + FP)

        FN = ((predictions == 0) & (labels == 1)).sum()
        recall = TP / (TP + FN)

        return accuracy, precision, recall


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
        z = self.sigmoid(-y_gold * np.dot(self.w, x))
        return -z * y_gold * x