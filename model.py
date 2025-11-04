import numpy as np

def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)

class Model:
    """
    Multiclass Logistic Regression (Softmax) implemented with pure NumPy.
    """
    def __init__(self, n_features=784, n_classes=10,
                 lr=0.1, epochs=200, batch_size=256, reg=1e-4,
                 verbose=True, seed=42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = reg
        self.verbose = verbose
        self.seed = seed

        rng = np.random.RandomState(self.seed)
        self.W = 0.01 * rng.randn(n_features, n_classes)
        self.b = np.zeros(n_classes)

        self.X_mean = np.zeros(n_features)
        self.X_std = np.ones(n_features)
        self.trained = False

    def _one_hot(self, y):
        oh = np.zeros((y.size, self.n_classes))
        oh[np.arange(y.size), y] = 1.0
        return oh

    def fit(self, X, y):
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        y = np.asarray(y, dtype=np.int64).ravel()
        n_samples = X.shape[0]

        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.X_std[self.X_std < 1e-6] = 1.0
        X = (X - self.X_mean) / self.X_std

        rng = np.random.RandomState(self.seed)

        for epoch in range(1, self.epochs + 1):
            perm = rng.permutation(n_samples)
            X_shuf, y_shuf = X[perm], y[perm]

            for i in range(0, n_samples, self.batch_size):
                xb = X_shuf[i:i + self.batch_size]
                yb = y_shuf[i:i + self.batch_size]

                logits = xb @ self.W + self.b
                probs = softmax(logits)
                y_onehot = self._one_hot(yb)
                gradW = xb.T @ (probs - y_onehot) / xb.shape[0] + self.reg * self.W
                gradb = np.mean(probs - y_onehot, axis=0)

                self.W -= self.lr * gradW
                self.b -= self.lr * gradb

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == 1 or epoch == self.epochs):
                logits = X @ self.W + self.b
                probs = softmax(logits)
                preds = np.argmax(probs, axis=1)
                acc = np.mean(preds == y)
                loss = -np.mean(np.log(np.clip(probs[np.arange(y.size), y], 1e-12, None)))
                print(f"[Epoch {epoch}/{self.epochs}] loss={loss:.4f} acc={acc:.4f}")

        self.trained = True

    def predict(self, X):
        X = X.reshape(X.shape[0], -1)
        X = (X - self.X_mean) / self.X_std
        logits = X @ self.W + self.b
        probs = softmax(logits)
        return np.argmax(probs, axis=1)
