import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def metrics_dict(y_true, y_pred) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred)),
        "rec": float(recall_score(y_true, y_pred)),
    }

def psi(a: pd.Series, b: pd.Series, bins=10) -> float:
    """Population Stability Index simple (para drift)."""
    a = a.dropna(); b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    quantiles = np.quantile(a, np.linspace(0, 1, bins + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        return float("nan")
    a_hist, _ = np.histogram(a, bins=quantiles)
    b_hist, _ = np.histogram(b, bins=quantiles)
    a_prop = a_hist / max(a_hist.sum(), 1)
    b_prop = b_hist / max(b_hist.sum(), 1)
    mask = (a_prop > 0) & (b_prop > 0)
    return float(np.sum((a_prop[mask] - b_prop[mask]) * np.log((a_prop[mask] + 1e-6) / (b_prop[mask] + 1e-6))))
