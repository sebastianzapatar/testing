import sys
from pathlib import Path
import pandas as pd
import joblib

def main(model_path: str, csv_path: str, sep=";"):
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path, sep=sep)
    # si el csv trae "quality", lo ignoramos
    X = df[[c for c in df.columns if c.lower() != "quality"]]
    preds = model.predict(X)
    print(preds.tolist())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python scripts/predict.py artifacts/model.joblib data/raw/winequality-white.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
