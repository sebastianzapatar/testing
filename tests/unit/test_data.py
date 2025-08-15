import pandas as pd
from mlwine.data import make_binary_target, TARGET

def test_make_binary_target():
    df = pd.DataFrame({"quality": [5, 6, 7, 8]})
    df2 = make_binary_target(df)
    assert TARGET in df2.columns
    assert df2[TARGET].tolist() == [0, 0, 1, 1]
