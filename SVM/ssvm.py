import numpy as np


class SoftSVM():
    """
    Soft-Margin SVM classifier
    """
    def __init__(self):
        self.weights = None


    def predict(self, X):
        A = np.dot(self.weights, X.T)
        return np.where(
            A >= 0,
            1,
            -1
        )
    

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    
    def fit(self, X, y, num_iters=100, lamb=1e-1, random_state=101):
        epsilon = 1e-9
        rng = np.random.default_rng(random_state)
        n_samples, n_features = X.shape

        weights = np.zeros((num_iters+1, n_features))
        
        theta = np.zeros(n_features)

        for t in range(num_iters):
            i = int(rng.uniform() * n_samples)

            weights[t, :] = (1 / ( lamb * t + epsilon)) * theta

            if (y[i] * np.dot(weights[t][:], X[i])) < 1:
                theta += y[i] * X[i]

        weights[num_iters, :] = (1 / ( lamb * num_iters)) * theta

        w_opt = np.mean(weights, axis=0)

        self.weights = w_opt