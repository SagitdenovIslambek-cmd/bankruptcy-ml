"""Загрузка данных, разбиение по времени, аудит признаков (константы, дубли, подозрение на утечку)."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["data"]["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла {path}. Скачайте data.csv с Kaggle «Company Bankruptcy Prediction» "
            "(fedesoriano) и положите в data/.")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Порядок строк = прокси времени. Явный индекс «периода» делаем только для графиков/описания.
    df.insert(0, "obs_order", np.arange(len(df)))
    return df


def time_split(df: pd.DataFrame, train: float = 0.70, valid: float = 0.15):
    """train — первые 70%, valid — следующие 15%, test (out-of-time) — последние 15%."""
    df = df.sort_values("obs_order", kind="stable")
    n = len(df)
    a, b = int(n * train), int(n * (train + valid))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


# ----- описание признаков -----
def describe_features(train: pd.DataFrame, target: str) -> pd.DataFrame:
    X = train.drop(columns=[target, "obs_order"])
    rows = []
    for c in X.columns:
        s = X[c]
        rows.append({"feature": c, "dtype": str(s.dtype), "missing_share": float(s.isna().mean()),
                     "n_unique": int(s.nunique()), "min": float(s.min()), "median": float(s.median()),
                     "max": float(s.max()),
                     "gini_univariate": abs(2 * roc_auc_score(train[target], s.fillna(s.median())) - 1)})
    return pd.DataFrame(rows)


def audit_features(train: pd.DataFrame, target: str, dup_thr: float = 0.98, leak_gini: float = 0.90):
    """Аудит ТОЛЬКО по train. Возвращает (список удаляемых признаков, таблица с причинами).

    Классы проблем:
      constant   — признак не несёт информации (<=1 уникального значения);
      duplicate  — |rho| >= dup_thr с другим признаком (оставляем более сильный по одномерному Gini на train);
      leak_suspect — одномерный Gini > leak_gini (подозрительно «слишком хорошо», кандидат на утечку).
    Признаки «после события» (сумма просрочки и т.п.) в этом датасете отсутствуют: все 95 показателей —
    коэффициенты годовой отчётности за период ДО банкротства (проверено по названиям, см. отчёт).
    """
    X = train.drop(columns=[target, "obs_order"])
    y = train[target]
    ug = {c: abs(2 * roc_auc_score(y, X[c]) - 1) for c in X.columns}
    removed = []
    for c in X.columns:
        if X[c].nunique() <= 1:
            removed.append({"feature": c, "reason": "constant", "detail": "единственное значение на train", "partner": ""})
    gone = {r["feature"] for r in removed}
    # однозначные «двойники» (|rho| ~ 1) по Спирмену: монотонные преобразования не ловим пирсоном
    corr = X.drop(columns=list(gone)).corr(method="spearman").abs()
    cols = list(corr.columns)
    for i, j in itertools.combinations(cols, 2):
        if i in gone or j in gone:
            continue
        if corr.loc[i, j] >= dup_thr:
            drop, keep = (i, j) if ug[i] < ug[j] else (j, i)
            gone.add(drop)
            removed.append({"feature": drop, "reason": "duplicate",
                            "detail": f"|rho_Spearman|={corr.loc[i, j]:.3f}, Gini {ug[drop]:.3f} < {ug[keep]:.3f}",
                            "partner": keep})
    for c, g in ug.items():
        if g > leak_gini and c not in gone:
            removed.append({"feature": c, "reason": "leak_suspect", "detail": f"одномерный Gini={g:.3f}", "partner": ""})
            gone.add(c)
    return sorted(gone), pd.DataFrame(removed)
