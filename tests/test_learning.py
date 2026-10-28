import copy
import math
import unittest
from datetime import date, timedelta

from learning import MODEL_FEATURES, train_models
from lyupang import CollectionError


def dataset(count=500):
    features, outcomes = [], []
    for i in range(count):
        day = date(2024, 1, 1) + timedelta(days=i)
        x = math.sin(i / 11)
        feature = dict.fromkeys(MODEL_FEATURES)
        feature.update(code="005930", as_of_date=day.isoformat(), available_from=(day + timedelta(days=1)).isoformat() + "T00:00:00+09:00",
                       financial_available=True, financial_receipt_date="2023-03-01", financial_available_from="2023-03-02T00:00:00+09:00",
                       quality_issue=False, return_5_pct=x, return_20_pct=x * 2, return_60_pct=x * 3,
                       volume_ratio_previous_20=1 + i / 10000, debt_ratio_pct=40, disclosures_previous_7_days=i % 4)
        features.append(feature)
        for horizon in (5, 20, 60):
            outcomes.append({"code": "005930", "as_of_date": day.isoformat(), "observed_sessions": horizon,
                             "status": "observed" if i + horizon < count else "pending",
                             "entry_date": (day + timedelta(days=1)).isoformat(),
                             "exit_date": (day + timedelta(days=horizon)).isoformat(), "price_return_pct": x * 3 + 1})
    return features, outcomes


class LearningTests(unittest.TestCase):
    def test_time_split_purges_labels_and_trains_against_a_baseline(self):
        features, outcomes = dataset()
        result = train_models(features, outcomes)
        self.assertEqual(result["status"], "trained")
        self.assertTrue(result["research_only"])
        self.assertFalse(result["live_trading_allowed"])
        for horizon, model in result["models"].items():
            self.assertLess(model["training_label_end"], result["split"]["test_start"])
            self.assertLess(model["test_metrics"]["mae_percentage_points"], model["test_metrics"]["baseline_mae_percentage_points"])
            self.assertGreater(result["excluded"][horizon]["cross_boundary"], 0)
            self.assertEqual(model["input_observed_counts"]["net_margin_pct"], 0)
            self.assertEqual(model["coefficients"][MODEL_FEATURES.index("net_margin_pct")], 0)
        self.assertNotIn("close", result["feature_whitelist"])

    def test_test_data_never_changes_imputation_scaling_tuning_or_coefficients(self):
        features, outcomes = dataset()
        first = train_models(features, outcomes)
        cutoff = first["split"]["test_start"]
        changed_features, changed_outcomes = copy.deepcopy(features), copy.deepcopy(outcomes)
        for row in changed_features:
            if row["as_of_date"] >= cutoff:
                row["return_5_pct"] = 99999999
                row["net_margin_pct"] = 99999999
        for row in changed_outcomes:
            if row["as_of_date"] >= cutoff:
                row["price_return_pct"] = -10000
        second = train_models(changed_features, changed_outcomes)
        for horizon in first["models"]:
            for key in ("median", "mean", "scale", "alpha", "coefficients", "intercept"):
                self.assertEqual(first["models"][horizon][key], second["models"][horizon][key])
            self.assertNotEqual(first["models"][horizon]["test_metrics"], second["models"][horizon]["test_metrics"])

    def test_unverified_financials_block_training_instead_of_price_only_fallback(self):
        features, outcomes = dataset()
        for row in features:
            row["financial_available"] = False
        result = train_models(features, outcomes)
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(all(model["status"] == "blocked" for model in result["models"].values()))

    def test_future_financial_receipt_is_rejected(self):
        features, outcomes = dataset()
        features[0]["financial_receipt_date"] = "2025-01-01"
        with self.assertRaises(CollectionError):
            train_models(features, outcomes)

    def test_delayed_information_availability_is_rejected_even_with_old_receipt(self):
        features, outcomes = dataset()
        features[0]["financial_available_from"] = "2025-01-01T00:00:00+09:00"
        with self.assertRaises(CollectionError):
            train_models(features, outcomes)


if __name__ == "__main__":
    unittest.main()
