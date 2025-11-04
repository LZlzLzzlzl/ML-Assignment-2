import numpy as np
from model import Model

class Solution:
    """
    Baseline Solution class 
    """
    def __init__(self, lr=0.1, epochs=200, batch_size=256, reg=1e-4, verbose=True):
        self.n_features = 28 * 28
        self.n_classes = 10

        self.model = Model(
            n_features=self.n_features,
            n_classes=self.n_classes,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            reg=reg,
            verbose=verbose,
            seed=42
        )
        try:
            data = np.load("data/train.npz")
            X_train = data["X_train"]
            y_train = data["y_train"]
        except Exception as e:
            raise RuntimeError(f"Failed to load training data: {e}")

        print("Training logistic-regression baseline...")
        self.model.fit(X_train, y_train)
        print("Training done!")

    def forward(self, sample):
        x = sample["image"] if isinstance(sample, dict) else sample
        x = np.asarray(x, dtype=float).reshape(1, -1)  
        preds = self.model.predict(x)
        return {"prediction": int(preds[0])}
