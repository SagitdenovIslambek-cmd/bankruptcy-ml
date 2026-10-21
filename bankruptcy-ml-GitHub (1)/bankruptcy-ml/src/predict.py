"""predict(df) — вероятность банкротства (PD) для новых данных в формате исходного data.csv."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = None


def load_bundle(path=ROOT / "model.pkl"):
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def predict(df: pd.DataFrame, model_path=ROOT / "model.pkl", return_decision: bool = False):
    """df — строки с теми же колонками, что в исходном data.csv (колонка целевой переменной не нужна).
    Возвращает pd.Series с калиброванной PD; при return_decision=True — DataFrame c колонками pd, decision
    (decision='reject', если PD >= порога, максимизирующего ожидаемую прибыль)."""
    b = load_bundle(model_path)
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    if "obs_order" not in d.columns:
        d.insert(0, "obs_order", np.arange(len(d)))
    X = b["feature_builder"].transform(d)
    pd_hat = pd.Series(b["calibrator"].predict_proba(X)[:, 1], index=df.index, name="pd")
    if not return_decision:
        return pd_hat
    out = pd_hat.to_frame()
    out["decision"] = np.where(pd_hat >= b["threshold"], "reject", "approve")
    return out


if __name__ == "__main__":
    import sys
    data = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "data.csv")
    print(predict(data.head(5), return_decision=True))
