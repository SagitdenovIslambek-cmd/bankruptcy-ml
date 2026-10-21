"""Быстрые проверки корректности: python -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import metrics as M  # noqa: E402
from src.data import load_config, load_data, time_split  # noqa: E402
from src.features import FeatureBuilder  # noqa: E402


class TestTimeAndLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.df = load_data(cls.cfg)
        cls.tr, cls.va, cls.te = time_split(cls.df)

    def test_split_is_chronological_and_disjoint(self):
        self.assertLess(self.tr.obs_order.max(), self.va.obs_order.min())
        self.assertLess(self.va.obs_order.max(), self.te.obs_order.min())
        self.assertEqual(len(self.tr) + len(self.va) + len(self.te), len(self.df))
        self.assertAlmostEqual(len(self.tr) / len(self.df), 0.70, places=2)

    def test_feature_builder_uses_only_train_statistics(self):
        t = self.cfg["data"]["target"]
        fb = FeatureBuilder([], (0.01, 0.99), t).fit(self.tr)
        before = dict(fb.bounds_)
        fb.transform(self.te)                       # transform на test не должен менять статистики train
        self.assertEqual(before, fb.bounds_)
        fb2 = FeatureBuilder([], (0.01, 0.99), t).fit(self.tr.iloc[:2000])
        self.assertNotEqual(before, fb2.bounds_)    # границы действительно зависят от данных fit

    def test_no_nan_in_features(self):
        t = self.cfg["data"]["target"]
        X = FeatureBuilder([], (0.01, 0.99), t).fit(self.tr).transform(self.te)
        self.assertEqual(int(X.isna().sum().sum()), 0)


class TestMetrics(unittest.TestCase):
    def test_gini_ks_perfect_and_random(self):
        y = np.array([0] * 50 + [1] * 50)
        self.assertAlmostEqual(M.gini(y, y.astype(float)), 1.0)
        self.assertAlmostEqual(M.ks_stat(y, y.astype(float)), 1.0)
        rng = np.random.default_rng(0)
        self.assertLess(abs(M.gini(y, rng.random(100))), 0.4)

    def test_psi_identity_and_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=5000)
        self.assertLess(M.psi(a, a), 1e-3)
        self.assertGreater(M.psi(a, a + 1.5), 0.25)

    def test_recall_at_share(self):
        y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        p = np.array([.9, .1, .2, .3, .8, .1, .1, .1, .1, .1])
        self.assertAlmostEqual(M.recall_at_share(y, p, 0.2), 1.0)

    def test_profit_threshold(self):
        y = np.array([0, 0, 0, 1])
        p = np.array([.01, .02, .03, .9])
        pf = M.portfolio_profit(y, p, 0.5, margin=0.1, lgd=0.45)
        self.assertAlmostEqual(pf["profit"], 0.3)
        thr, _ = M.best_threshold(p, 0.1, 0.45)
        self.assertLess(thr, 0.9 + 1e-9)


if __name__ == "__main__":
    unittest.main()
