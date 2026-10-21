"""Обучение: бустинг (LightGBM; запасной движок sklearn HistGradientBoosting), подбор параметров по времени,
ранняя остановка, калибровка. Никакого обычного KFold — только TimeSeriesSplit / validation."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

try:                                    # основной движок — как требует задание
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:                     # только для проверки пайплайна в среде без lightgbm
    lgb, HAS_LGB = None, False

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:                     # sklearn < 1.6
    FrozenEstimator = None


def resolve_engine(requested: str) -> str:
    if requested == "lightgbm" and not HAS_LGB:
        warnings.warn("lightgbm не установлен — используется запасной движок hist_gb (результаты НЕ финальные). "
                      "Установите зависимости: pip install -r requirements.txt", RuntimeWarning)
        return "hist_gb"
    return requested


SEARCH_SPACE = {
    "num_leaves": [4, 8, 16, 31],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "min_child_samples": [20, 50, 100],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 20.0],
    "scale_pos_weight": [1.0, 3.0, 10.0],
}


def search_space(engine: str) -> dict:
    sp = dict(SEARCH_SPACE)
    if engine != "lightgbm":                # у запасного движка веса классов сильно замедляют обучение
        sp["scale_pos_weight"] = [1.0]
    return sp


def make_model(params: dict, engine: str, seed: int, n_estimators: int | None = None):
    p = dict(params)
    for k in ("num_leaves", "min_child_samples", "n_estimators", "subsample_freq", "max_depth"):
        if k in p:                          # после pandas целые приходят как float
            p[k] = int(p[k])
    if n_estimators is not None:
        p["n_estimators"] = n_estimators
    if engine == "lightgbm":
        p.setdefault("subsample_freq", 1)           # без subsample_freq>0 параметр subsample игнорируется
        return lgb.LGBMClassifier(**p, metric="auc", random_state=seed, deterministic=True,
                                  force_row_wise=True, n_jobs=1, verbose=-1)
    return HistGradientBoostingClassifier(
        max_iter=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", 0.05),
        max_leaf_nodes=p.get("num_leaves", 31), min_samples_leaf=p.get("min_child_samples", 20),
        l2_regularization=p.get("reg_lambda", 0.0), max_features=p.get("colsample_bytree", 1.0),
        class_weight=(None if float(p.get("scale_pos_weight", 1.0)) == 1.0
                      else {0: 1.0, 1: float(p["scale_pos_weight"])}), early_stopping=False,
        random_state=seed)


def fit_early_stopping(params, engine, seed, Xtr, ytr, Xva, yva, rounds=100, max_estimators=3000):
    """Обучает на train, останавливается по AUC на validation. Возвращает (модель, лучшая итерация)."""
    if engine == "lightgbm":
        m = make_model(params, engine, seed, n_estimators=max_estimators)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(rounds, first_metric_only=True, verbose=False)])
        best = int(m.best_iteration_ or max_estimators)
        return m, best
    m = make_model(params, engine, seed, n_estimators=min(max_estimators, 600))
    m.fit(Xtr, ytr)
    best, best_auc, since = 1, -1, 0
    for i, pr in enumerate(m.staged_predict_proba(Xva), start=1):
        a = roc_auc_score(yva, pr[:, 1])
        if a > best_auc:
            best, best_auc, since = i, a, 0
        else:
            since += 1
            if since >= rounds:
                break
    m = make_model(params, engine, seed, n_estimators=best).fit(Xtr, ytr)
    return m, best


def timeseries_search(X, y, engine, seed, n_iter=30, n_splits=5, rounds=100, max_estimators=3000):
    """Случайный поиск; оценка — средний AUC по расширяющимся окнам TimeSeriesSplit внутри train
    (валидация всегда «после» обучения). Возвращает таблицу результатов, отсортированную по качеству."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for k, params in enumerate(ParameterSampler(search_space(engine), n_iter=n_iter, random_state=seed)):
        aucs, iters = [], []
        for tr_i, va_i in tscv.split(X):
            ytr, yva = y.iloc[tr_i], y.iloc[va_i]
            if yva.sum() == 0:
                continue
            m, best = fit_early_stopping(params, engine, seed, X.iloc[tr_i], ytr, X.iloc[va_i], yva,
                                         rounds, max_estimators)
            aucs.append(roc_auc_score(yva, m.predict_proba(X.iloc[va_i])[:, 1]))
            iters.append(best)
        rows.append({**params, "cv_gini": 2 * np.mean(aucs) - 1, "cv_gini_std": 2 * np.std(aucs),
                     "mean_best_iter": float(np.mean(iters))})
    return pd.DataFrame(rows).sort_values("cv_gini", ascending=False).reset_index(drop=True)


class PDCalibrator:
    """Обёртка над CalibratedClassifierCV: ограничивает PD снизу (isotonic на малой выборке даёт ровно 0,
    а PD = 0 для компании не бывает)."""

    def __init__(self, cal, floor=0.0005):
        self.cal, self.floor = cal, floor
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = np.clip(self.cal.predict_proba(X)[:, 1], self.floor, 1 - self.floor)
        return np.c_[1 - p, p]


def calibrate(model, Xva, yva, method="isotonic", floor=0.0005):
    """Калибровка вероятностей на validation (модель «заморожена»)."""
    if FrozenEstimator is not None:
        cal = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    else:
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
    return PDCalibrator(cal.fit(Xva, yva), floor)


def cv_gini_baseline(builder_fn, X, y, n_splits=5):
    """Те же окна TimeSeriesSplit для базовой модели (WoE-карта переобучается в каждом окне)."""
    out = []
    for tr_i, va_i in TimeSeriesSplit(n_splits=n_splits).split(X):
        if y.iloc[va_i].sum() == 0:
            continue
        m = builder_fn().fit(X.iloc[tr_i], y.iloc[tr_i])
        out.append(2 * roc_auc_score(y.iloc[va_i], m.predict_proba(X.iloc[va_i])) - 1)
    return float(np.mean(out)), float(np.std(out))
