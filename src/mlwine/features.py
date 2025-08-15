import numpy as np
import pandas as pd

def build_qa(X: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Crea un set QA con extremos y algunos NaNs para testear robustez."""
    parts = []
    for c in num_cols:
        if c not in X.columns: 
            continue
        lo, hi = np.nanpercentile(X[c], [1, 99])
        parts.append(pd.DataFrame({c: [lo - abs(lo)*0.5, hi + abs(hi)*0.5]}))
    qa = pd.concat(parts, axis=1)
    qa = qa.reindex(columns=X.columns)
    # mete NaNs en primeras columnas
    if len(qa) > 0:
        qa.iloc[0, :min(3, qa.shape[1])] = np.nan
    return qa
