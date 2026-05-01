# custom_model.py

import numpy as np

class ManualScaler:
    def fit_transform(self, X):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X)
        
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # Standart sapması 0 olan kolonlar (sabit değerler) için sıfıra bölme hatasını önle
        self.std = np.where(self.std == 0, 1.0, self.std) 
        return (X - self.mean) / self.std

    def transform(self, X):
        X = np.array(X, dtype=float)
        X = np.nan_to_num(X)
        return (X - self.mean) / self.std

    def save(self, filepath):
        np.savez(filepath, mean=self.mean, std=self.std)

    def load(self, filepath):
        data = np.load(filepath)
        self.mean = data['mean']
        self.std = data['std']

class CustomLogisticRegression:
    def __init__(self, lr=0.05, epochs=1000, batch_size=8192):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for j in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                y_batch = y_shuffled[j:j+self.batch_size]
                
                model = np.dot(X_batch, self.weights) + self.bias
                predictions = self._sigmoid(model)
                
                dw = (1 / len(X_batch)) * np.dot(X_batch.T, (predictions - y_batch))
                db = (1 / len(X_batch)) * np.sum(predictions - y_batch)
                
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
            # Track progress every 10% of total epochs
            print_interval = max(1, self.epochs // 10)
            if i % print_interval == 0 or i == self.epochs - 1:
                # Calculate loss on a random subset to save time
                subset_idx = np.random.choice(n_samples, min(10000, n_samples), replace=False)
                model_subset = np.dot(X[subset_idx], self.weights) + self.bias
                pred_subset = self._sigmoid(model_subset)
                y_subset = y[subset_idx]
                loss = -np.mean(y_subset * np.log(pred_subset + 1e-9) + (1 - y_subset) * np.log(1 - pred_subset + 1e-9))
                print(f"Epoch {i}: Loss {loss:.4f}")

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        probs = self._sigmoid(model)
        return np.array([1 if i > 0.5 else 0 for i in probs])

    def save(self, filepath):
        np.savez(filepath, weights=self.weights, bias=self.bias)

    def load(self, filepath):
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']