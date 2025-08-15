from typing import Iterable
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

def build_pipeline(num_cols: Iterable[str], class_weight: str | None = None) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]), list(num_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=400,
        class_weight=class_weight  # prueba: None o 'balanced'
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe
