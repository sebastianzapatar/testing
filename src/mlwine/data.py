from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "quality_good"

def load_wine(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)  # fallback
    return df

def make_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = (df["quality"] >= 7).astype(int)
    return df

def get_splits(df: pd.DataFrame, random_state: int = 42):
    feature_cols = [c for c in df.columns if c not in ("quality", TARGET)]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state, stratify=y_train
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols
