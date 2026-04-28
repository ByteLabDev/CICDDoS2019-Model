# custom_model.py

import numpy as np

class ManualScaler:
    def fit_transform(self, X):
        X = np.array(X, dtype=float)
        # Replace any residual NaNs with 0 before math
        X = np.nan_to_num(X) 
        
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        
        range_val = self.max - self.min
        # If range is 0 (constant column), denom is 1 to avoid div by zero
        denom = np.where(range_val == 0, 1.0, range_val)
        return (X - self.min) / denom

    def transform(self, X):
        X = np.array(X, dtype=float)
        range_val = self.max - self.min
        denom = np.where(range_val == 0, 1, range_val)
        return (X - self.min) / denom

class CustomLogisticRegression:
    def __init__(self, lr=0.05, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epochs):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)
            
            # Calculate Log-Loss to track progress
            if i % 100 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
                print(f"Epoch {i}: Loss {loss:.4f}")

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        probs = self._sigmoid(model)
        return np.array([1 if i > 0.5 else 0 for i in probs])