"""Классическая скоринговая карта: WoE-биннинг + логистическая регрессия + шкала баллов."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _bin_stats(y, bins, k):
    n_bad = np.bincount(bins, weights=y, minlength=k)
    n_tot = np.bincount(bins, minlength=k).astype(float)
    return n_tot, n_bad


class WoEBinner:
    """Квантильный биннинг с объединением соседних корзин, пока в каждой не меньше min_share наблюдений
    и min_bads банкротов. WoE = ln(доля хороших / доля плохих), со сглаживанием +0.5."""

    def __init__(self, max_bins=10, min_share=0.05, min_bads=5):
        self.max_bins, self.min_share, self.min_bads = max_bins, min_share, min_bads

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.edges_, self.woe_, self.iv_, self.table_ = {}, {}, {}, {}
        G, B = (y == 0).sum(), (y == 1).sum()
        for c in X.columns:
            x = X[c].to_numpy(float)
            q = np.unique(np.quantile(x, np.linspace(0, 1, self.max_bins + 1)[1:-1]))
            while True:
                b = np.searchsorted(q, x, side="right")
                k = len(q) + 1
                tot, bad = _bin_stats(y, b, k)
                weak = (tot < self.min_share * len(y)) | (bad < self.min_bads)
                if not weak.any() or k <= 2:
                    break
                i = int(np.argmin(np.where(weak, tot, np.inf)))     # самая слабая корзина
                # сливаем с соседом, у которого меньше наблюдений
                nb = i - 1 if i == k - 1 else (i + 1 if i == 0 else (i - 1 if tot[i - 1] <= tot[i + 1] else i + 1))
                cut = min(i, nb)                                     # убираем границу между i и nb
                q = np.delete(q, cut)
            b = np.searchsorted(q, x, side="right")
            k = len(q) + 1
            tot, bad = _bin_stats(y, b, k)
            good = tot - bad
            woe = np.log(((good + 0.5) / (G + 0.5 * k)) / ((bad + 0.5) / (B + 0.5 * k)))
            iv = float(np.sum(((good + 0.5) / (G + 0.5 * k) - (bad + 0.5) / (B + 0.5 * k)) * woe))
            self.edges_[c], self.woe_[c], self.iv_[c] = q, woe, iv
            lo = np.concatenate([[-np.inf], q]); hi = np.concatenate([q, [np.inf]])
            self.table_[c] = pd.DataFrame({"feature": c, "bin": range(k), "lo": lo, "hi": hi, "n": tot.astype(int),
                                           "bads": bad.astype(int), "bad_rate": np.divide(bad, tot, out=np.zeros_like(bad), where=tot > 0), "woe": woe})
        return self

    def transform(self, X: pd.DataFrame, cols=None) -> pd.DataFrame:
        cols = cols or list(self.edges_)
        out = {c: self.woe_[c][np.searchsorted(self.edges_[c], X[c].to_numpy(float), side="right")] for c in cols}
        return pd.DataFrame(out, index=X.index)


class Scorecard:
    """WoE → отбор по IV и корреляции → логистическая регрессия → баллы."""

    def __init__(self, max_bins=10, min_share=0.05, min_bads=5, iv_min=0.05, corr_max=0.7, max_features=15, C=0.1,
                 base_score=600, base_odds=50, pdo=20, seed=42):
        self.p = dict(max_bins=max_bins, min_share=min_share, min_bads=min_bads)
        self.iv_min, self.corr_max, self.max_features, self.C = iv_min, corr_max, max_features, C
        self.base_score, self.base_odds, self.pdo, self.seed = base_score, base_odds, pdo, seed

    def fit(self, X: pd.DataFrame, y):
        self.binner_ = WoEBinner(**self.p).fit(X, y)
        iv = pd.Series(self.binner_.iv_).sort_values(ascending=False)
        cand = [c for c in iv.index if iv[c] >= self.iv_min]
        W = self.binner_.transform(X, cand)
        chosen = []
        for c in cand:                                    # жадно: сильные признаки первыми, дубли по корреляции отбрасываем
            if len(chosen) >= self.max_features:
                break
            if all(abs(np.corrcoef(W[c], W[s])[0, 1]) < self.corr_max for s in chosen):
                chosen.append(c)
        self.iv_series_ = iv
        y = np.asarray(y)
        while True:                                       # WoE = ln(good/bad): коэффициент обязан быть < 0 (знак логичен)
            self.lr_ = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed).fit(W[chosen], y)
            bad_sign = [c for c, b in zip(chosen, self.lr_.coef_[0]) if b > 0]
            if not bad_sign:
                break
            chosen.remove(bad_sign[-1])                   # убираем самый слабый по IV среди «неправильных»
        self.features_ = chosen
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.binner_.transform(X, self.features_))[:, 1]

    def get_params(self, deep=False):
        return {}

    def points_table(self) -> pd.DataFrame:
        """Баллы за каждую корзину: score = offset − factor·logit(PD)."""
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        n = len(self.features_)
        rows = []
        for j, c in enumerate(self.features_):
            t = self.binner_.table_[c].copy()
            coef = self.lr_.coef_[0][j]
            t["coef"] = coef
            t["points"] = -(coef * t["woe"] + self.lr_.intercept_[0] / n) * factor + offset / n
            rows.append(t)
        return pd.concat(rows, ignore_index=True)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.predict_proba(X), 1e-9, 1 - 1e-9)
        factor = self.pdo / np.log(2)
        return self.base_score - factor * np.log(self.base_odds) - factor * np.log(p / (1 - p))
