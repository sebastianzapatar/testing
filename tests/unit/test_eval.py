import numpy as np
import pandas as pd
from mlwine.eval import psi, metrics_dict

def test_metrics_dict():
    y_true = [0,1,1,0]
    y_pred = [0,1,0,0]
    m = metrics_dict(y_true, y_pred)
    for k in ("acc","f1","prec","rec"):
        assert k in m

def test_psi_returns_number():
    a = pd.Series(np.random.randn(300))
    b = a * 1.1 + 0.05
    val = psi(a,b)
    assert np.isfinite(val) or np.isnan(val)
