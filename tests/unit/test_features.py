import pandas as pd
from mlwine.features import build_qa

def test_build_qa_extremes_and_nans():
    X = pd.DataFrame({"a": [1,2,3,4,5], "b": [10,11,12,13,14]})
    qa = build_qa(X, ["a","b"])
    assert qa.shape[0] == 2
    assert qa.isna().sum().sum() >= 1  # inyectamos NaNs
