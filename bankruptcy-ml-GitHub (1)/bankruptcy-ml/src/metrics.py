"""Метрики: Gini, KS, PSI, полнота при доле проверок, прибыль портфеля, калибровка."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks_stat(y, p) -> float:
    order = np.argsort(p)
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index; границы — квантили expected (train)."""
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:                       # почти константа — нечего сравнивать
        return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected) + 1e-6
    a = np.histogram(actual, cuts)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def recall_at_share(y, p, share: float) -> float:
    """Полнота (доля пойманных банкротов) среди доли `share` компаний с наибольшим скором."""
    y = np.asarray(y)
    k = max(1, int(np.ceil(share * len(y))))
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def lift_at_share(y, p, share: float) -> float:
    y = np.asarray(y)
    k = max(1, int(np.ceil(share * len(y))))
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(y[top].mean() / y.mean())


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error по квантильным корзинам."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def summary(y, p, shares=(0.05, 0.10, 0.20)) -> dict:
    out = {"gini": gini(y, p), "auc": roc_auc_score(y, p), "ks": ks_stat(y, p),
           "pr_auc": pr_auc(y, p), "brier": brier(y, p), "ece": ece(y, p)}
    for s in shares:
        out[f"recall@{int(s * 100)}%"] = recall_at_share(y, p, s)
    return out


def bootstrap_ci(y, p, fn, n: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Стратифицированный бутстреп (банкротов мало — сохраняем их долю)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(fn(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---------- финансовая оценка (вариант 5: решение «выдать / отказать») ----------

def portfolio_profit(y, p, threshold: float, margin: float, lgd: float, ead: float = 1.0) -> dict:
    """Одобряем компании с PD < threshold.
    Доход: margin*EAD с каждой одобренной «хорошей»; потери: LGD*EAD с каждой одобренной обанкротившейся."""
    y, p = np.asarray(y), np.asarray(p)
    ok = p < threshold
    good, bad = int(((y == 0) & ok).sum()), int(((y == 1) & ok).sum())
    income, loss = margin * ead * good, lgd * ead * bad
    return {"threshold": float(threshold), "approved": int(ok.sum()), "approve_rate": float(ok.mean()),
            "good_approved": good, "bad_approved": bad, "income": income, "loss": loss,
            "profit": income - loss, "profit_per_exposure": (income - loss) / (ead * len(y))}


def expected_profit(p, threshold: float, margin: float, lgd: float, ead: float = 1.0) -> float:
    """Ожидаемая прибыль по прогнозным PD (без знания факта): sum[(1-PD)*m - PD*LGD] по одобренным."""
    p = np.asarray(p)
    ok = p < threshold
    return float(np.sum((1 - p[ok]) * margin * ead - p[ok] * lgd * ead))


def best_threshold(p, margin: float, lgd: float, ead: float = 1.0):
    """Порог, максимизирующий ожидаемую прибыль по прогнозным PD (перебор по уникальным значениям PD)."""
    p = np.asarray(p)
    cands = np.unique(np.concatenate([[p.min() - 1e-9], p, [p.max() + 1e-9]]))
    vals = np.array([expected_profit(p, t, margin, lgd, ead) for t in cands])
    i = int(np.argmax(vals))
    return float(cands[i]), float(vals[i])
