"""Четыре этапа задания как функции. Ноутбуки и run_all.py только вызывают их.
Промежуточные результаты — в artifacts/, итог — artifacts/results.json, графики — reports/figures/."""
from __future__ import annotations

import json
import warnings
from functools import lru_cache

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from . import explain as ex
from . import metrics as M
from .data import ROOT, audit_features, describe_features, load_config, load_data, time_split
from .features import NEW_FEATURES, FeatureBuilder, feature_rationales
from .train import (calibrate, cv_gini_baseline, fit_early_stopping, make_model, resolve_engine, search_space,
                    timeseries_search)
from .woe import Scorecard

ART = ROOT / "artifacts"
FIG = ROOT / "reports" / "figures"
ART.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
C_BASE, C_BOOST, C_NEUT = "#7F8C8D", "#1F618D", "#C0392B"


def _save_json(name, obj):
    with open(ART / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def _load_json(name):
    with open(ART / name, encoding="utf-8") as f:
        return json.load(f)


def update_results(section: str, obj: dict):
    path = ART / "results.json"
    res = json.load(open(path, encoding="utf-8")) if path.exists() else {}
    res[section] = obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)


@lru_cache(maxsize=1)
def context():
    """Конфиг, данные, разбиение по времени, удаляемые признаки, FeatureBuilder (fit на train)."""
    cfg = load_config()
    t = cfg["data"]["target"]
    df = load_data(cfg)
    tr, va, te = time_split(df, cfg["split"]["train"], cfg["split"]["valid"])
    drop, audit = audit_features(tr, t, cfg["features"]["corr_duplicate_threshold"])
    fb = FeatureBuilder(drop, tuple(cfg["features"]["clip_quantiles"]), t).fit(tr)
    c = dict(cfg=cfg, target=t, df=df, tr=tr, va=va, te=te, drop=drop, audit=audit, fb=fb,
             engine=resolve_engine(cfg["boosting"]["engine"]))
    c["Xtr"], c["Xva"] = fb.transform(tr), fb.transform(va)
    c["ytr"], c["yva"] = tr[t].reset_index(drop=True), va[t].reset_index(drop=True)
    c["Xtr"], c["Xva"] = c["Xtr"].reset_index(drop=True), c["Xva"].reset_index(drop=True)
    return c


def _test(c):
    """Test создаётся ТОЛЬКО здесь — и вызывается только из этапа 4."""
    Xte = c["fb"].transform(c["te"]).reset_index(drop=True)
    return Xte, c["te"][c["target"]].reset_index(drop=True)


# =========================== ЭТАП 1 ===========================
def step1_data():
    c = context()
    cfg, t, df, tr, va, te = c["cfg"], c["target"], c["df"], c["tr"], c["va"], c["te"]
    seed = cfg["seed"]
    desc = describe_features(tr, t)
    desc["status"] = np.where(desc["feature"].isin(c["drop"]), "удалён", "оставлен")
    desc.to_csv(ART / "features_description.csv", index=False)
    c["audit"].to_csv(ART / "removed_features.csv", index=False)
    split = pd.DataFrame({"sample": ["train", "valid", "test (OOT)"], "n": [len(tr), len(va), len(te)],
                          "bankrupt": [int(x[t].sum()) for x in (tr, va, te)],
                          "bankrupt_rate": [float(x[t].mean()) for x in (tr, va, te)],
                          "row_from": [int(x.obs_order.min()) for x in (tr, va, te)],
                          "row_to": [int(x.obs_order.max()) for x in (tr, va, te)]})
    split.to_csv(ART / "split_summary.csv", index=False)

    # демонстрация: что дало бы случайное разбиение (одна и та же простая модель)
    p = dict(learning_rate=0.05, num_leaves=8, min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=1.0, scale_pos_weight=1.0)
    m_t = make_model(p, c["engine"], seed, 150).fit(c["Xtr"], c["ytr"])
    g_time = M.gini(c["yva"], m_t.predict_proba(c["Xva"])[:, 1])
    # случайное разбиение — только на строках train+valid (test до этапа 4 не трогаем)
    pool = pd.concat([tr, va])
    Xp = c["fb"].transform(pool).reset_index(drop=True)
    yp = pool[t].reset_index(drop=True)
    idx = np.random.default_rng(seed).permutation(len(pool))
    a_, b_ = len(tr), len(pool)
    m_r = make_model(p, c["engine"], seed, 150).fit(Xp.iloc[idx[:a_]], yp.iloc[idx[:a_]])
    g_rand = M.gini(yp.iloc[idx[a_:b_]], m_r.predict_proba(Xp.iloc[idx[a_:b_]])[:, 1])
    demo = {"gini_valid_time_split": g_time, "gini_valid_random_split": g_rand}

    # динамика доли банкротств по порядку строк
    blocks = np.array_split(np.arange(len(df)), 10)
    rate = [float(df[t].iloc[b_].mean()) for b_ in blocks]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.bar(range(10), rate, color=C_BOOST)
    ax.axvline(6.5, color=C_NEUT, ls="--", lw=1); ax.axvline(8.0, color=C_NEUT, ls="--", lw=1)
    ax.set_xticks(range(10)); ax.set_xticklabels([f"{i + 1}" for i in range(10)])
    ax.set_xlabel("десятая часть выборки (по порядку строк)"); ax.set_ylabel("доля банкротств")
    ax.set_title("Доля банкротств по порядку строк", fontsize=8.5)
    for x0, x1, lab in ((-.5, 6.5, "train"), (6.5, 8.0, "valid"), (8.0, 9.5, "test")):
        ax.text((x0 + x1) / 2, max(rate) * 1.03, lab, ha="center", fontsize=8, color=C_NEUT)
    plt.tight_layout(); plt.savefig(FIG / "fig1_drift.png", dpi=170); plt.close()

    out = {"n_rows": len(df), "n_raw_features": int(len(df.columns) - 2), "bankrupt_total": int(df[t].sum()),
           "split": split.to_dict("records"), "n_removed": int(len(c["drop"])),
           "removed_by_reason": c["audit"]["reason"].value_counts().to_dict(),
           "n_features_after_audit": int(len(c["fb"].raw_cols_)), "n_new_features": len(NEW_FEATURES),
           "block_rates": rate, "split_demo": demo, "missing_total": int(df.isna().sum().sum()),
           "max_univariate_gini": float(desc.gini_univariate.max()), "engine": c["engine"]}
    update_results("step1", out)
    return out


# =========================== ЭТАП 2 ===========================
def _scorecard_factory(cfg, C):
    b = cfg["baseline"]
    return lambda: Scorecard(b["max_bins"], b["min_bin_share"], b["min_bin_bads"], b["iv_min"], b["corr_max"],
                             b.get("max_features", 15), C, b["scorecard"]["base_score"], b["scorecard"]["base_odds"],
                             b["scorecard"]["pdo"], cfg["seed"])


def step2_baseline():
    c = context()
    cfg, Xtr, Xva, ytr, yva = c["cfg"], c["Xtr"], c["Xva"], c["ytr"], c["yva"]
    grid = []
    for C in cfg["baseline"]["c_grid"]:
        mu, sd = cv_gini_baseline(_scorecard_factory(cfg, C), Xtr, ytr)
        grid.append({"C": C, "cv_gini": mu, "cv_gini_std": sd})
    grid = pd.DataFrame(grid)
    bestC = float(grid.sort_values("cv_gini", ascending=False).iloc[0]["C"])
    sc = _scorecard_factory(cfg, bestC)().fit(Xtr, ytr)
    p_tr, p_va = sc.predict_proba(Xtr), sc.predict_proba(Xva)
    cal = calibrate_prefit_scorecard(sc, Xva, yva)
    sh = cfg["finance"]["check_shares"]
    res = {"best_C": bestC, "n_features": len(sc.features_), "features": sc.features_,
           "cv_gini": float(grid.cv_gini.max()), "cv_gini_std": float(grid.sort_values("cv_gini").iloc[-1].cv_gini_std),
           "train": M.summary(ytr, p_tr, sh), "valid": M.summary(yva, p_va, sh),
           "valid_calibrated": M.summary(yva, cal.predict_proba(Xva)[:, 1], sh),
           "valid_gini_ci": M.bootstrap_ci(yva, p_va, M.gini, cfg["bootstrap"]["n"], cfg["seed"]),
           "sign_ok": bool((sc.lr_.coef_ < 0).all()), "grid": grid.to_dict("records")}
    pts = sc.points_table()
    pts.to_csv(ART / "scorecard_points.csv", index=False)
    pd.DataFrame({"feature": sc.iv_series_.index, "iv": sc.iv_series_.values,
                  "selected": sc.iv_series_.index.isin(sc.features_)}).to_csv(ART / "iv_table.csv", index=False)
    joblib.dump({"scorecard": sc, "calibrator": cal}, ART / "baseline.pkl")
    update_results("step2", res)
    return res


class ScorecardCalibrator:
    """Isotonic-калибровка выхода скоринговой карты (обучается на validation)."""
    def __init__(self, sc, Xva, yva):
        from sklearn.isotonic import IsotonicRegression
        self.sc = sc
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sc.predict_proba(Xva), yva)

    def predict_proba(self, X):
        p = self.iso.predict(self.sc.predict_proba(X))
        return np.c_[1 - p, p]


def calibrate_prefit_scorecard(sc, Xva, yva):
    return ScorecardCalibrator(sc, Xva, yva)


# =========================== ЭТАП 3 ===========================
def _write_final_params(final: dict):
    """Обновляет только блок boosting.final_params в config.yaml, сохраняя комментарии."""
    import re
    path = ROOT / "config.yaml"
    txt = path.read_text(encoding="utf-8")
    lines = ["  final_params:"] + [f"    {k}: {v}" for k, v in
                                   {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in final.items()}.items()]
    new = re.sub(r"  final_params:\n(?:    .*\n)+", "\n".join(lines) + "\n", txt)
    path.write_text(new, encoding="utf-8")


def _reliability(ax, y, p, label, color, bins=6):
    fy, fx = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    ax.plot(fx, fy, "o-", color=color, label=label, ms=4)


def step3_boosting():
    c = context()
    cfg, Xtr, Xva, ytr, yva, eng = c["cfg"], c["Xtr"], c["Xva"], c["ytr"], c["yva"], c["engine"]
    seed, bc = cfg["seed"], cfg["boosting"]
    n_iter = bc["search"]["n_iter"] if eng == "lightgbm" else min(bc["search"]["n_iter"], 8)
    max_est = bc["max_estimators"] if eng == "lightgbm" else 400
    rounds = bc["early_stopping_rounds"]
    search = timeseries_search(Xtr, ytr, eng, seed, n_iter, bc["search"]["n_splits"], rounds, max_est)
    search.to_csv(ART / "search_results.csv", index=False)
    keys = list(search_space(eng))
    best = {k: (int(v) if k in ("num_leaves", "min_child_samples") else float(v)) for k, v in search.iloc[0][keys].items()}
    model, n_best = fit_early_stopping(best, eng, seed, Xtr, ytr, Xva, yva, rounds, max_est)
    final = {**best, "n_estimators": int(n_best), "subsample_freq": 1, "reg_alpha": 0.0}
    if eng != "lightgbm":
        final = {k: v for k, v in final.items() if k not in ("subsample_freq", "reg_alpha")}

    p_raw = model.predict_proba(Xva)[:, 1]
    cal = calibrate(model, Xva, yva, cfg["calibration"]["method"], cfg["calibration"]["pd_floor"])
    p_cal = cal.predict_proba(Xva)[:, 1]
    sh = cfg["finance"]["check_shares"]

    # вклад новых признаков: та же конфигурация на «сырых» признаках
    raw_cols = c["fb"].raw_cols_
    m_raw, n_raw = fit_early_stopping(best, eng, seed, Xtr[raw_cols], ytr, Xva[raw_cols], yva, rounds, max_est)
    g_rawfeat = M.gini(yva, m_raw.predict_proba(Xva[raw_cols])[:, 1])

    base = _load_json("results.json")["step2"]
    table = pd.DataFrame([
        {"model": "Скоринговая карта (WoE + логрег)", "cv_gini_train": base["cv_gini"], **{k: base["valid"][k] for k in ("gini", "ks", "pr_auc", "brier", "recall@10%")}},
        {"model": "Бустинг (сырые вероятности)", "cv_gini_train": float(search.iloc[0].cv_gini), **{k: M.summary(yva, p_raw, sh)[k] for k in ("gini", "ks", "pr_auc", "brier", "recall@10%")}},
        {"model": "Бустинг + isotonic-калибровка", "cv_gini_train": float(search.iloc[0].cv_gini), **{k: M.summary(yva, p_cal, sh)[k] for k in ("gini", "ks", "pr_auc", "brier", "recall@10%")}},
    ])
    table.to_csv(ART / "comparison_valid.csv", index=False)
    ci_b = M.bootstrap_ci(yva, p_raw, M.gini, cfg["bootstrap"]["n"], seed)

    # калибровочная кривая на validation (по сути in-sample для isotonic — честная кривая на test в этапе 4)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, max(p_raw.max(), .1)], [0, max(p_raw.max(), .1)], ":", color="gray")
    _reliability(ax, yva, p_raw, "до калибровки", C_BASE)
    _reliability(ax, yva, p_cal, "isotonic", C_BOOST)
    ax.set_xlabel("прогнозная PD"); ax.set_ylabel("доля банкротств"); ax.legend(fontsize=7)
    ax.set_title("Калибровка, validation", fontsize=8.5)
    plt.tight_layout(); plt.savefig(FIG / "fig_calibration_valid.png", dpi=170); plt.close()

    joblib.dump({"model": model, "calibrator": cal, "params": final, "engine": eng}, ART / "boosting.pkl")
    _write_final_params(final)

    res = {"engine": eng, "final_params": final, "best_iteration": int(n_best), "n_search": int(len(search)),
           "search_best_cv_gini": float(search.iloc[0].cv_gini), "search_best_cv_std": float(search.iloc[0].cv_gini_std),
           "search_median_cv_gini": float(search.cv_gini.median()),
           "valid_raw": M.summary(yva, p_raw, sh), "valid_calibrated": M.summary(yva, p_cal, sh),
           "valid_gini_ci": ci_b, "gini_valid_raw_features_only": g_rawfeat, "n_features_model": int(Xtr.shape[1]),
           "comparison": table.to_dict("records")}
    update_results("step3", res)
    return res


# =========================== ЭТАП 4 ===========================
def _psi_table(ctx, model, sv_train, X_tr, X_va, X_te, p_tr, p_va, p_te, base_scores):
    bc = ctx["cfg"]["stability"]
    top = ex.mean_abs(sv_train, X_tr.columns).head(bc["top_features"]).index.tolist()
    rows = []
    for name, a, b, d in [("score (бустинг)", p_tr, p_va, p_te)] + [(f, X_tr[f].values, X_va[f].values, X_te[f].values) for f in top] + \
                         [("score (скоринговая карта)",) + base_scores]:
        rows.append({"item": name, "psi_train_valid": M.psi(a, b, bc["bins"]), "psi_train_test": M.psi(a, d, bc["bins"]),
                     "train_median": float(np.median(a)), "test_median": float(np.median(d)),
                     "median_shift_pct": float((np.median(d) - np.median(a)) / (abs(np.median(a)) + 1e-12) * 100)})
    t = pd.DataFrame(rows)
    t["flag"] = np.where(t.psi_train_test > bc["psi_threshold"], "НЕСТАБИЛЕН (>0,25)",
                         np.where(t.psi_train_test > bc["psi_warn"], "умеренный сдвиг", "стабилен"))
    return t, top


def step4_evaluation():
    c = context()
    cfg, t, eng = c["cfg"], c["target"], c["engine"]
    Xtr, Xva, ytr, yva = c["Xtr"], c["Xva"], c["ytr"], c["yva"]
    Xte, yte = _test(c)                                 # ← единственное обращение к test
    bst = joblib.load(ART / "boosting.pkl")
    model, cal = bst["model"], bst["calibrator"]
    base = joblib.load(ART / "baseline.pkl")
    sc, cal_b = base["scorecard"], base["calibrator"]
    fin, sh, nb, seed = cfg["finance"], cfg["finance"]["check_shares"], cfg["bootstrap"]["n"], cfg["seed"]

    p_raw_te, p_cal_te = model.predict_proba(Xte)[:, 1], cal.predict_proba(Xte)[:, 1]
    p_sc_te, p_scc_te = sc.predict_proba(Xte), cal_b.predict_proba(Xte)[:, 1]
    p_cal_va, p_scc_va = cal.predict_proba(Xva)[:, 1], cal_b.predict_proba(Xva)[:, 1]
    p_raw_tr, p_raw_va = model.predict_proba(Xtr)[:, 1], model.predict_proba(Xva)[:, 1]

    # ---- 1. метрики варианта на out-of-time ----
    def block(p, pc):
        s = M.summary(yte, p, sh)
        s["brier_calibrated"], s["ece_calibrated"] = M.brier(yte, pc), M.ece(yte, pc)
        s["gini_ci"] = M.bootstrap_ci(yte, p, M.gini, nb, seed)
        s["ks_ci"] = M.bootstrap_ci(yte, p, M.ks_stat, nb, seed)
        for sv in sh:
            s[f"recall@{int(sv * 100)}%_ci"] = M.bootstrap_ci(yte, p, lambda a, b, sv=sv: M.recall_at_share(a, b, sv), nb, seed)
            s[f"lift@{int(sv * 100)}%"] = M.lift_at_share(yte, p, sv)
        return s
    test_boost, test_base = block(p_raw_te, p_cal_te), block(p_sc_te, p_scc_te)
    # разность Gini бустинг − база с бутстреп-ИнтервалоМ (пары на одних и тех же наблюдениях)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(yte == 1)[0], np.where(yte == 0)[0]
    diffs = []
    for _ in range(nb):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(M.gini(yte.iloc[i], p_raw_te[i]) - M.gini(yte.iloc[i], p_sc_te[i]))
    gini_diff = {"mean": float(np.mean(diffs)), "ci": [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))],
                 "share_boost_better": float(np.mean(np.array(diffs) > 0))}

    # ---- 2. деньги ----
    lgd, m0, ead = fin["lgd"], fin["margin"], fin["ead"]
    thr, exp_val = M.best_threshold(p_cal_va, m0, lgd, ead)           # порог выбран на validation по ожидаемой прибыли
    thr_b, _ = M.best_threshold(p_scc_va, m0, lgd, ead)
    all_ok = M.portfolio_profit(yte, p_cal_te, 2.0, m0, lgd, ead)
    boost_pf = M.portfolio_profit(yte, p_cal_te, thr, m0, lgd, ead)
    base_pf = M.portfolio_profit(yte, p_scc_te, thr_b, m0, lgd, ead)
    oracle = M.portfolio_profit(yte, yte.astype(float).to_numpy(), 0.5, m0, lgd, ead)
    # реализованная прибыль на сетке порогов (для графика): отказ топ-k% по скору
    shares = np.linspace(0, 0.30, 61)
    def profit_by_reject_share(p):
        order = np.argsort(-p, kind="stable"); out = []
        for s in shares:
            k = int(round(s * len(p))); keep = np.ones(len(p), bool); keep[order[:k]] = False
            yk = yte.to_numpy()[keep]
            out.append((m0 * (yk == 0).sum() - lgd * (yk == 1).sum()) * ead / (ead * len(p)))
        return np.array(out)
    pr_boost, pr_base = profit_by_reject_share(p_raw_te), profit_by_reject_share(p_sc_te)
    sens = []
    for m in fin["margin_sensitivity"]:
        tm, _ = M.best_threshold(p_cal_va, m, lgd, ead)
        a, b_, o = M.portfolio_profit(yte, p_cal_te, 2.0, m, lgd, ead), M.portfolio_profit(yte, p_cal_te, tm, m, lgd, ead), None
        tb, _ = M.best_threshold(p_scc_va, m, lgd, ead)
        bb = M.portfolio_profit(yte, p_scc_te, tb, m, lgd, ead)
        sens.append({"margin": m, "threshold_pd": tm, "reject_rate": 1 - b_["approve_rate"],
                     "profit_all_approved": a["profit_per_exposure"], "profit_boosting": b_["profit_per_exposure"],
                     "profit_baseline": bb["profit_per_exposure"], "gain_vs_all": b_["profit_per_exposure"] - a["profit_per_exposure"]})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.plot(shares * 100, pr_boost * 100, color=C_BOOST, label="бустинг"); ax.plot(shares * 100, pr_base * 100, color=C_BASE, label="скоринговая карта")
    ax.axhline(all_ok["profit_per_exposure"] * 100, color="gray", ls=":", label="одобрить всех")
    ax.set_xlabel("% компаний с отказом (наибольший скор)"); ax.set_ylabel("прибыль, % от экспозиции"); ax.legend(fontsize=7)
    ax.set_title("Прибыль на test от доли отказов", fontsize=8.5)
    ax = axes[1]
    ks = np.linspace(0, 1, 101); gb = [M.recall_at_share(yte, p_raw_te, max(s, 1 / len(yte))) for s in ks]
    gs = [M.recall_at_share(yte, p_sc_te, max(s, 1 / len(yte))) for s in ks]
    ax.plot(ks * 100, np.array(gb) * 100, color=C_BOOST, label="бустинг"); ax.plot(ks * 100, np.array(gs) * 100, color=C_BASE, label="скоринговая карта")
    ax.plot([0, 100], [0, 100], ":", color="gray"); ax.set_xlim(0, 40); ax.set_ylim(0, 105)
    ax.set_xlabel("% компаний, отобранных на проверку"); ax.set_ylabel("полнота, % банкротов"); ax.legend(fontsize=7)
    ax.set_title("Кривая полноты (cumulative gains), test", fontsize=8.5)
    plt.tight_layout(); plt.savefig(FIG / "fig_finance.png", dpi=170); plt.close()

    # сравнение способов калибровки на test (калибраторы обучены на validation)
    from sklearn.linear_model import LogisticRegression
    lg = lambda p_: np.log(np.clip(p_, 1e-6, 1 - 1e-6) / (1 - np.clip(p_, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(lg(p_raw_va), yva)
    p_sig_te = platt.predict_proba(lg(p_raw_te))[:, 1]
    calib_cmp = {"raw": {"brier": M.brier(yte, p_raw_te), "ece": M.ece(yte, p_raw_te), "mean_pd": float(p_raw_te.mean())},
                 "isotonic": {"brier": M.brier(yte, p_cal_te), "ece": M.ece(yte, p_cal_te), "mean_pd": float(p_cal_te.mean())},
                 "sigmoid": {"brier": M.brier(yte, p_sig_te), "ece": M.ece(yte, p_sig_te), "mean_pd": float(p_sig_te.mean())},
                 "actual_rate": float(yte.mean())}

    # калибровка на test (честная)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    mx = max(p_raw_te.max(), p_cal_te.max(), .1)
    ax.plot([0, mx], [0, mx], ":", color="gray")
    _reliability(ax, yte, p_raw_te, "бустинг, до калибровки", C_BASE, 5); _reliability(ax, yte, p_cal_te, "бустинг, isotonic", C_BOOST, 5)
    ax.set_xlabel("прогнозная PD"); ax.set_ylabel("доля банкротств"); ax.legend(fontsize=7); ax.set_title("Калибровка, test (OOT)", fontsize=8.5)
    plt.tight_layout(); plt.savefig(FIG / "fig_calibration_test.png", dpi=170); plt.close()

    # ---- 3. SHAP ----
    med = Xtr.median()
    samp = Xtr.sample(min(1500, len(Xtr)), random_state=seed)
    sv_tr, _, method_tr = ex.explain(model, samp, med)
    sv_te, base_val, method = ex.explain(model, Xte, med)
    ex.plot_summary(sv_te, Xte, FIG / "fig_shap_summary.png", 15, method)
    imp = ex.mean_abs(sv_te, Xte.columns)
    from scipy.stats import spearmanr
    shap_dir = {}
    for f_ in imp.head(10).index:
        j = list(Xte.columns).index(f_)
        r_ = spearmanr(Xte[f_], sv_te[:, j])[0]
        shap_dir[f_] = float(0.0 if np.isnan(r_) else r_)           # >0: рост значения признака повышает риск
    pos_i, neg_i = np.where(yte == 1)[0], np.where(yte == 0)[0]
    pick = {"A": pos_i[np.argmax(p_raw_te[pos_i])], "B": neg_i[np.argmax(p_raw_te[neg_i])], "C": pos_i[np.argmin(p_raw_te[pos_i])]}
    titles = {"A": "A: отказ, банкрот (верно)", "B": "B: отказ, не банкрот (ложная тревога)", "C": "C: банкрот, пропущен"}
    cases, cases_out = [], []
    for k, i in pick.items():
        tc = ex.top_contrib(sv_te[i], Xte.iloc[i], 5)
        cases.append({"title": f"{titles[k]}\nPD={p_cal_te[i]:.1%}", "contrib": tc})
        cases_out.append({"case": k, "title": titles[k], "row_in_test": int(i), "target": int(yte.iloc[i]), "pd_calibrated": float(p_cal_te[i]),
                          "score_pct_rank": float((p_raw_te < p_raw_te[i]).mean()), "decision": "отказ" if p_cal_te[i] >= thr else "одобрение",
                          "top": tc.to_dict("records")})
    ex.plot_cases(cases, FIG / "fig_shap_cases.png")

    # ---- 4. PSI ----
    psi_tab, top = _psi_table(c, model, sv_tr, Xtr, Xva, Xte, p_raw_tr, p_raw_va, p_raw_te,
                              (sc.predict_proba(Xtr), sc.predict_proba(Xva), sc.predict_proba(Xte)))
    psi_tab.to_csv(ART / "psi.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    yy = np.arange(len(psi_tab))[::-1]
    ax.barh(yy + .18, psi_tab.psi_train_valid, .34, color=C_BASE, label="train → valid")
    ax.barh(yy - .18, psi_tab.psi_train_test, .34, color=C_BOOST, label="train → test")
    ax.axvline(cfg["stability"]["psi_threshold"], color=C_NEUT, ls="--", lw=1); ax.axvline(cfg["stability"]["psi_warn"], color="orange", ls=":", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels([s[:38] for s in psi_tab["item"]], fontsize=7.5); ax.set_xlabel("PSI"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "fig_psi.png", dpi=170); plt.close()

    # ---- 5. model.pkl + проверка predict(df) ----
    bundle = {"feature_builder": c["fb"], "model": model, "calibrator": cal, "threshold": thr, "engine": eng,
              "params": bst["params"], "features": list(Xtr.columns)}
    joblib.dump(bundle, ROOT / "model.pkl")
    from . import predict as P
    P._BUNDLE = None
    chk = P.predict(c["te"].drop(columns=[t, "obs_order"]).iloc[:50])
    pred_ok = bool(np.allclose(chk.values, p_cal_te[:50], atol=1e-9))

    res = {"engine": eng, "shap_method": method, "boost": test_boost, "base": test_base, "gini_diff": gini_diff,
           "finance": {"margin": m0, "lgd": lgd, "threshold_pd": thr, "threshold_pd_baseline": thr_b,
                       "expected_profit_valid": exp_val, "calibration_compare": calib_cmp, "all_approved": all_ok, "boosting": boost_pf,
                       "baseline": base_pf, "oracle": oracle, "sensitivity": sens,
                       "money_example_ead": fin["ead_money_example"]},
           "shap_top": imp.head(10).to_dict(), "shap_direction": shap_dir, "shap_top_new": {k: float(imp[k]) for k in NEW_FEATURES if k in imp.index}, "cases": cases_out, "psi": psi_tab.to_dict("records"),
           "top_features_stability": top, "predict_check_ok": pred_ok,
           "test_n": int(len(yte)), "test_bankrupt": int(yte.sum())}
    update_results("step4", res)
    return res


# =========================== вспомогательное для ноутбуков ===========================
_STEPS = {"step1": step1_data, "step2": step2_baseline, "step3": step3_boosting, "step4": step4_evaluation}


def ensure(step: str) -> dict:
    """Результат этапа из artifacts/results.json; если этапа ещё не было — запускает его."""
    path = ART / "results.json"
    if path.exists():
        res = json.load(open(path, encoding="utf-8"))
        if step in res:
            return res[step]
    for s in ("step1", "step2", "step3", "step4"):      # этапы зависят друг от друга — идём по порядку
        have = path.exists() and s in json.load(open(path, encoding="utf-8"))
        if not have:
            _STEPS[s]()
        if s == step:
            break
    return json.load(open(path, encoding="utf-8"))[step]


def show_fig(name: str):
    from IPython.display import Image, display
    display(Image(filename=str(FIG / name)))
