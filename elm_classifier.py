# ELM classifier with TGA optimization
# Train ELM and optimize parameters with Tree Growth Algorithm

import numpy as np
from sklearn.metrics import accuracy_score

class ELMClassifier:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def fit(self, X, y):
        self.input_weights = np.random.normal(size=(self.input_dim, self.hidden_dim))
        self.biases = np.random.normal(size=(self.hidden_dim,))
        H = self._sigmoid(np.dot(X, self.input_weights) + self.biases)
        self.output_weights = np.dot(np.linalg.pinv(H), y)

    def predict(self, X):
        H = self._sigmoid(np.dot(X, self.input_weights) + self.biases)
        return (np.dot(H, self.output_weights) > 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
