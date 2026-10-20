"""Data loading. scikit-learn is used ONLY here, ONLY to load a standard dataset
(sklearn.datasets.load_digits: 1797 8x8 handwritten digits). No sklearn model,
optimizer, metric or gradient is used anywhere in this repository.
"""
from __future__ import annotations

import numpy as np


def load_digits_split(seed: int, val_frac: float = 0.2, test_frac: float = 0.2) -> dict:
    from sklearn.datasets import load_digits  # <- the only scikit-learn import in the repo

    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0                 # pixels are 0..16
    rng = np.random.default_rng(seed)               # the seed controls the split too
    perm = rng.permutation(len(X))
    n_test, n_val = int(len(X) * test_frac), int(len(X) * val_frac)
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)  # standardise with TRAIN statistics only
    sd = np.where(sd < 1e-8, 1.0, sd)               # constant (always-zero) pixels
    norm = lambda a: (a - mu) / sd
    return {"x_train": norm(X[tr]), "y_train": y[tr],
            "x_val": norm(X[va]), "y_val": y[va],
            "x_test": norm(X[te]), "y_test": y[te]}
