import json, os, sys, subprocess, tempfile, pathlib

def test_e2e_train_script_smoke():
    data_path = "data/raw/winequality-white.csv"
    if not os.path.exists(data_path):
        import pytest; pytest.skip("dataset real no encontrado")

    out = tempfile.mkdtemp()
    cmd = [sys.executable, "scripts/train.py", "--data", data_path, "--out", out]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    manifest = json.loads(res.stdout)
    # sanity checks
    assert "metrics" in manifest
    assert "model_path" in manifest
    # un umbral razonable (ajústalo si cambias modelo)
    assert manifest["metrics"]["acc"] > 0.70
