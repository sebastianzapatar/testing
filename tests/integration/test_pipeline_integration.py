import numpy as np
import pandas as pd
from mlwine.model import build_pipeline

def test_pipeline_fit_predict_smoke():
    X = pd.DataFrame({"x1": np.random.randn(100), "x2": np.random.rand(100)})
    y = (X["x1"] + X["x2"] > 0.5).astype(int)
    pipe = build_pipeline(["x1","x2"])
    pipe.fit(X,y)
    preds = pipe.predict(X.iloc[:5])
    assert len(preds) == 5
