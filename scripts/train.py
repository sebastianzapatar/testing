import json, time
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from mlwine.data import load_wine, make_binary_target, get_splits
from mlwine.features import build_qa
from mlwine.model import build_pipeline
from mlwine.eval import metrics_dict, psi

import pandas as pd

def main(data_path: str = "data/raw/winequality-white.csv",
         out_dir: str = "artifacts",
         class_weight: str | None = None,
         random_state: int = 42):

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    df = load_wine(data_path)
    df = make_binary_target(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), num_cols = get_splits(df, random_state)

    pipe = build_pipeline(num_cols, class_weight=class_weight)

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    # pred latencia aprox sobre 1000 filas
    n_batch = min(1000, len(X_test))
    t0 = time.perf_counter()
    _ = pipe.predict(X_test.iloc[:n_batch])
    predict_s = time.perf_counter() - t0

    y_pred = pipe.predict(X_test)
    m = metrics_dict(y_test, y_pred)
    m.update({"fit_s": float(fit_s), "predict_s": float(predict_s),
              "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test))})

    # QA (extremos+NaNs)
    qa = build_qa(X_train, num_cols)
    qa_status = "NA"
    qa_preds = None
    if len(qa) > 0:
        try:
            qa_preds = pipe.predict(qa).tolist()
            qa_status = f"OK (rows={len(qa)})"
        except Exception as e:
            qa_status = f"FAILED: {e.__class__.__name__}: {e}"

    # Drift simple (PSI) simulando pequeño shift
    X_new = X_test.copy()
    for c in num_cols:
        X_new[c] = X_new[c] * 1.05 + 0.1
    psis = []
    for c in num_cols:
        val = psi(X_test[c], X_new[c])
        if not np.isnan(val): psis.append(val)
    avg_psi = float(np.mean(psis)) if psis else float("nan")

    # Guardar artefactos
    model_path = out / "model.joblib"
    manifest_path = out / "manifest.json"
    joblib.dump(pipe, model_path)

    manifest = {
        "metrics": m,
        "qa_status": qa_status,
        "qa_pred_sample": qa_preds[:10] if qa_preds else None,
        "avg_psi": avg_psi,
        "model_path": str(model_path),
        "created": pd.Timestamp.utcnow().isoformat()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/winequality-white.csv")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--class_weight", default=None)
    args = p.parse_args()
    cw = None if args.class_weight in (None, "None", "") else args.class_weight
    main(data_path=args.data, out_dir=args.out, class_weight=cw)
