"""Генерация признаков с экономическим смыслом. Все статистики (квантили) считаются ТОЛЬКО на train."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 0.05


def _u(s: pd.Series) -> pd.Series:
    """Коэффициент в [0, 1]: в датасете часть значений закодирована как 1e9–1e10 (выбросы/«заглушки»)."""
    return s.clip(0, 1)


# (имя, функция, экономическое обоснование)
FEATURE_SPECS = [
    ("cash_cover_gap",
     lambda d: _u(d["Cash/Total Assets"]) - _u(d["Current Liability to Assets"]),
     "Денежные средства минус краткосрочные обязательства (в долях активов): способность погасить "
     "ближайшие долги без продажи активов. Отрицательное значение — кассовый разрыв."),
    ("quick_cover_gap",
     lambda d: _u(d["Quick Assets/Total Assets"]) - _u(d["Current Liability to Assets"]),
     "Быстрые активы минус краткосрочные обязательства: расширенная ликвидность "
     "(аналог quick ratio в разностной форме, устойчив к делению на малые числа)."),
    ("roa_per_leverage",
     lambda d: _u(d["Net Income to Total Assets"]) / (_u(d["Debt ratio %"]) + EPS),
     "Рентабельность активов на единицу долговой нагрузки: прибыльность компании с поправкой на то, "
     "каким долгом она достигнута."),
    ("cfo_per_leverage",
     lambda d: _u(d["CFO to Assets"]) / (_u(d["Debt ratio %"]) + EPS),
     "Операционный денежный поток на единицу долга: банкротство определяется нехваткой денег, "
     "а не бухгалтерской прибылью."),
    ("interest_drag",
     lambda d: _u(d["Operating Profit Rate"]) - _u(d["Pre-tax net Interest Rate"]),
     "Разрыв между операционной рентабельностью и рентабельностью после процентов: "
     "доля маржи, которую съедает обслуживание долга."),
    ("tax_drag",
     lambda d: _u(d["After-tax net Interest Rate"]) - _u(d["Pre-tax net Interest Rate"]),
     "Налоговая нагрузка на прибыль после процентов (разрыв до/после налога): "
     "аномалии могут говорить о качестве отчётности."),
    ("retained_self_financing",
     lambda d: _u(d["Retained Earnings to Total Assets"]) * _u(d["Net worth/Assets"]),
     "Накопленная нераспределённая прибыль × доля собственного капитала: "
     "способность компании финансировать себя изнутри (буфер на убытки)."),
    ("short_debt_leverage",
     lambda d: _u(d["Current Liability to Liability"]) * _u(d["Debt ratio %"]),
     "Доля краткосрочных обязательств в долге × общая долговая нагрузка: риск рефинансирования — "
     "короткий долг в большом объёме нужно постоянно перекредитовывать."),
    ("fixed_asset_debt_burden",
     lambda d: _u(d["Fixed Assets to Assets"]) * _u(d["Debt ratio %"]),
     "Доля основных средств × долг: неликвидные активы, профинансированные долгом, "
     "трудно быстро превратить в деньги при стрессе."),
    ("leverage_effect",
     lambda d: _u(d["Net Income to Stockholder's Equity"]) - _u(d["Net Income to Total Assets"]),
     "ROE − ROA: эффект финансового рычага (насколько доходность капитала держится на заёмных деньгах)."),
    ("eps_to_book",
     lambda d: _u(d["Persistent EPS in the Last Four Seasons"]) / (_u(d["Net Value Per Share (B)"]) + 0.01),
     "Устойчивая прибыль на акцию к балансовой стоимости акции: доходность на собственный капитал, "
     "очищенная от сезонных всплесков (4 квартала)."),
    ("cashflow_vs_borrowing",
     lambda d: _u(d["Cash Flow to Liability"]) - _u(d["Borrowing dependency"]),
     "Покрытие обязательств денежным потоком минус зависимость от заёмных средств: "
     "хватает ли потока на обслуживание той доли долга, на которой держится бизнес."),
    ("margin_x_turnover",
     lambda d: _u(d["Gross Profit to Sales"]) * _u(d["Total Asset Turnover"]),
     "Валовая маржа × оборачиваемость активов (компонента модели Дюпона): "
     "бизнес-модель «высокая маржа или быстрый оборот» защищает от убытков."),
    ("working_capital_vs_debt",
     lambda d: _u(d["Working Capital to Total Assets"]) - _u(d["Debt ratio %"]),
     "Чистый оборотный капитал минус общий долг (в долях активов): запас «прочности» после обслуживания долга."),
]
NEW_FEATURES = [n for n, _, _ in FEATURE_SPECS] + ["stress_flag"]
STRESS_RATIONALE = ("Флаг двойного стресса: рентабельность активов в нижнем квартиле И долговая нагрузка в верхнем "
                    "квартиле (пороги по train). Одновременно слабая прибыль и высокий долг — классический "
                    "предвестник дефолта, нелинейная комбинация, которую линейная модель сама не соберёт.")


def feature_rationales() -> dict:
    d = {n: r for n, _, r in FEATURE_SPECS}
    d["stress_flag"] = STRESS_RATIONALE
    return d


class FeatureBuilder:
    """fit() — только на train; transform() — на любой выборке. Никаких статистик от validation/test."""

    def __init__(self, drop_cols=(), clip_q=(0.01, 0.99), target="Bankrupt?"):
        self.drop_cols = list(drop_cols)
        self.clip_q = clip_q
        self.target = target

    def _new(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({n: f(df) for n, f, _ in FEATURE_SPECS}, index=df.index)
        out["stress_flag"] = ((_u(df["Net Income to Total Assets"]) <= self.roa_q25_)
                              & (_u(df["Debt ratio %"]) >= self.debt_q75_)).astype(float)
        return out

    def fit(self, train: pd.DataFrame):
        self.roa_q25_ = _u(train["Net Income to Total Assets"]).quantile(0.25)
        self.debt_q75_ = _u(train["Debt ratio %"]).quantile(0.75)
        new = self._new(train)
        self.bounds_ = {c: (new[c].quantile(self.clip_q[0]), new[c].quantile(self.clip_q[1]))
                        for c in new.columns if c != "stress_flag"}
        raw = [c for c in train.columns if c not in (self.target, "obs_order") and c not in self.drop_cols]
        self.raw_cols_ = raw
        self.feature_names_ = raw + NEW_FEATURES
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new = self._new(df)
        for c, (lo, hi) in self.bounds_.items():
            new[c] = new[c].clip(lo, hi)
        X = pd.concat([df[self.raw_cols_], new], axis=1)
        return X[self.feature_names_].astype(float)

    def fit_transform(self, train):
        return self.fit(train).transform(train)
