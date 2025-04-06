"""
Soft-Margin Support Vector Machine Classifier
Uses a Gaussian Radial Basis Function kernel
"""

import numpy as np
import scipy.spatial.distance as distance


class KSVM():
    def __init__(self):
        self.alpha = None
        self.X = None
        self.h = None


    def predict(self, z):
        k = self._rbf_kernel(self.X, z, self.h)
        
        clf_scores = np.dot(self.alpha, k)

        return np.where(
            clf_scores >= 0,
            1,
            -1
        )

        
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    

    def fit(self, X, y, lamb=1e-1, num_iters=10000):
        beta = np.zeros(len(X))
        alphas = np.zeros((num_iters, len(X)))

        self.X = X

        self.h = self._median_distance(X)

        K = self._rbf_kernel(X, X, self.h)

        for t in range(num_iters):
            rng = np.random.default_rng()

            i = rng.integers(0, len(X))

            alphas[t] = (1 / lamb) * beta

            z = np.dot(alphas[t], K[:][i]) * y[i]

            if z < 1:
                beta[i] = beta[i] + y[i]

        self.alpha = np.mean(alphas, axis=0)


    def _median_distance(self, X):
        """ Compute the median distance between points in the training set. """
        n = len(X)

        distances = distance.cdist(X, X)
        
        h = np.median(
            [distances[i, j] for i in range(n) for j in range(n) if i != j]
        )
        
        return h

    
    def _rbf_kernel(self, X, Xp, h):
        """ Kernel function that measures similarity between x and xp. """
        return np.exp(-(distance.cdist(X, Xp) ** 2) / (2 * h ** 2))