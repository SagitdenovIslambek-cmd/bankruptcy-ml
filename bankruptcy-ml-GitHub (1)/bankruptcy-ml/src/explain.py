"""SHAP-интерпретация. Основной путь — shap.TreeExplainer (LightGBM). Если пакет shap не установлен
(только в среде проверки), используется запасной метод «замещение признака медианой train» —
результаты такого запуска помечаются method='occlusion' и не являются финальными."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap, HAS_SHAP = None, False


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def explain(model, X: pd.DataFrame, background_median: pd.Series):
    """Возвращает (матрица вкладов в log-odds, базовое значение, метод)."""
    if HAS_SHAP:
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(X)
        if isinstance(sv, list):                 # старые версии shap: [класс 0, класс 1]
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                         # (n, features, classes)
            sv = sv[:, :, 1]
        base = ex.expected_value
        base = float(np.ravel(base)[-1])
        return sv, base, "shap"
    full = _logit(model.predict_proba(X)[:, 1])
    sv = np.zeros(X.shape)
    for j, c in enumerate(X.columns):
        Xo = X.copy()
        Xo[c] = background_median[c]
        sv[:, j] = full - _logit(model.predict_proba(Xo)[:, 1])
    return sv, float(np.mean(full - sv.sum(1))), "occlusion"


def mean_abs(sv: np.ndarray, cols) -> pd.Series:
    return pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)


def plot_summary(sv, X: pd.DataFrame, path, max_display=15, method="shap"):
    if HAS_SHAP and method == "shap":
        plt.figure(figsize=(8, 5.2))
        shap.summary_plot(sv, X, max_display=max_display, show=False, plot_size=None)
        plt.tight_layout()
        plt.savefig(path, dpi=170, bbox_inches="tight")
        plt.close()
        return
    imp = mean_abs(sv, X.columns).head(max_display)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    rng = np.random.default_rng(0)
    for i, c in enumerate(imp.index):
        j = list(X.columns).index(c)
        v = X[c].rank(pct=True).to_numpy()
        ax.scatter(sv[:, j], i + rng.uniform(-0.3, 0.3, len(v)), c=v, cmap="coolwarm", s=6)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=8)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("вклад признака в log-odds банкротства (запасной метод: замещение медианой)")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def top_contrib(sv_row, x_row: pd.Series, k=5) -> pd.DataFrame:
    idx = np.argsort(-np.abs(sv_row))[:k]
    return pd.DataFrame({"feature": x_row.index[idx], "value": x_row.values[idx], "shap": sv_row[idx]})


def plot_cases(cases: list, path):
    """cases: список dict(title, contrib DataFrame, pd). Три горизонтальные панели вкладов."""
    fig, axes = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 3.6))
    for ax, c in zip(np.atleast_1d(axes), cases):
        t = c["contrib"].iloc[::-1]
        ax.barh([f[:34] for f in t["feature"]], t["shap"], color=["#C0392B" if v > 0 else "#2E86C1" for v in t["shap"]])
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_title(c["title"], fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("вклад в log-odds (красный — к банкротству)", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()
