"""Model zoo. Each entry: (family, factory, description of key configuration)."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


class EpochMLP(BaseEstimator, ClassifierMixin):
    """sklearn MLP trained epoch-by-epoch so we can record real train/validation log-loss curves."""

    def __init__(self, hidden=(64, 32), alpha=1e-4, lr=1e-3, batch_size=1024, epochs=15, seed=C.SEED):
        self.hidden, self.alpha, self.lr = hidden, alpha, lr
        self.batch_size, self.epochs, self.seed = batch_size, epochs, seed

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        Xv = self.scaler_.transform(X_val) if X_val is not None else None
        self.net_ = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha,
                                  learning_rate_init=self.lr, batch_size=self.batch_size,
                                  random_state=self.seed)
        rng = np.random.default_rng(self.seed)
        self.history_ = {"train_loss": [], "val_loss": []}
        self.classes_ = np.array([0, 1])
        sub = rng.choice(len(Xs), min(len(Xs), 100_000), replace=False)  # train-loss monitor subsample
        for _ in range(self.epochs):
            perm = rng.permutation(len(Xs))
            self.net_.partial_fit(Xs[perm], np.asarray(y)[perm], classes=self.classes_)
            self.history_["train_loss"].append(log_loss(np.asarray(y)[sub], self.net_.predict_proba(Xs[sub])[:, 1], labels=[0, 1]))
            if Xv is not None:
                self.history_["val_loss"].append(log_loss(y_val, self.net_.predict_proba(Xv)[:, 1], labels=[0, 1]))
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


KNN_MAX_TRAIN = 50_000   # KNN is memory/time heavy: fit on a random subsample (documented)

MODEL_ZOO = {
    "logreg": ("Linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
               "StandardScaler + L2, C=1.0"),
    "gnb": ("Probabilistic", lambda: make_pipeline(StandardScaler(), GaussianNB()),
            "var_smoothing=1e-9"),
    "knn": ("Instance-based", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=25, weights="distance", n_jobs=-1)),
            f"k=25, distance-weighted, scaled, fit on <= {KNN_MAX_TRAIN} rows"),
    "rf": ("Ensemble (bagging)", lambda: RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, max_samples=0.25, n_jobs=-1, random_state=C.SEED),
           "100 trees, min_samples_leaf=5, bootstrap 25% rows/tree"),
    "lgbm": ("Boosting", lambda: LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=C.SEED, verbose=-1),
             "400 trees, lr=0.05, num_leaves=31"),
    "mlp": ("Neural network", lambda: EpochMLP(),
            "hidden=(64,32), relu, adam lr=1e-3, alpha=1e-4, batch 1024, 15 epochs"),
}


def fit_model(name, X, y, rng=None):
    model = MODEL_ZOO[name][1]()
    if name == "knn" and len(X) > KNN_MAX_TRAIN:
        rng = rng or np.random.default_rng(C.SEED)
        idx = rng.choice(len(X), KNN_MAX_TRAIN, replace=False)
        X, y = X[idx], y[idx]
    model.fit(X, y)
    return model
