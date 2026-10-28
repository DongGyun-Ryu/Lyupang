import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import review
from financial_history import FINANCIAL_COLUMNS
from learning import train_models
from lyupang import CollectionError
from research_data import FEATURE_COLUMNS, OUTCOME_COLUMNS, write_table
from test_learning import dataset


def save_source(root, models, features, outcomes):
    root.mkdir(parents=True)
    # Use the actual runtime CSV schema, including nullable and boolean columns.
    source_features = [{key: row.get(key) for key in FEATURE_COLUMNS + FINANCIAL_COLUMNS} for row in features]
    for row in source_features:
        row["dart_available"] = True
    source_outcomes = [{key: row.get(key) for key in OUTCOME_COLUMNS} for row in outcomes]
    write_table(root / "features_with_financials.csv", FEATURE_COLUMNS + FINANCIAL_COLUMNS, source_features)
    write_table(root / "future_outcomes.csv", OUTCOME_COLUMNS, source_outcomes)
    metadata = {"created_at": "2026-10-28T11:43:27+09:00",
                "feature_sha256": hashlib.sha256((root / "features_with_financials.csv").read_bytes()).hexdigest(),
                "outcome_sha256": hashlib.sha256((root / "future_outcomes.csv").read_bytes()).hexdigest()}
    (root / "summary.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "models.json").write_text(json.dumps(models), encoding="utf-8")


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.features, self.outcomes = dataset()
        self.models = train_models(self.features, self.outcomes)
        self.source = self.root / "learning" / "sample"
        save_source(self.source, self.models, self.features, self.outcomes)

    def test_roundtrip_reproduces_metrics_without_refitting_or_changing_source(self):
        original = {p.name: p.read_bytes() for p in self.source.iterdir()}
        with patch("learning.fit", side_effect=AssertionError("no retraining")):
            target, result = review.run(self.source, self.root / "diagnostics", self.root / "registry")
        for horizon, item in result["models"].items():
            self.assertEqual(item["disposition"], "hold")
            self.assertAlmostEqual(item["overall"]["mae_pp"], self.models["models"][horizon]["test_metrics"]["mae_percentage_points"])
            self.assertEqual(sum(group["rows"] for group in item["quarters"].values()), item["overall"]["rows"])
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, original)
        self.assertFalse(result["live_trading_allowed"])
        self.assertFalse(result["automatic_promotion"])
        self.assertFalse(result["fresh_unseen_evidence"])
        with (target / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertGreater(len(list(csv.DictReader(stream))), 0)

    def test_worse_model_is_rejected_and_not_rescued_by_selecting_a_subperiod(self):
        models = copy.deepcopy(self.models)
        bad = models["models"]["60"]
        bad["coefficients"] = [-100 * v for v in bad["coefficients"]]
        from learning import index_outcomes, metrics, partition_examples
        parts, _ = partition_examples(self.features, index_outcomes(self.outcomes), 60,
                                      models["split"]["validation_start"], models["split"]["test_start"])
        bad["test_metrics"] = metrics(bad, parts["test"])
        result, _ = review.diagnose(models, self.features, self.outcomes)
        self.assertEqual(result["models"]["60"]["disposition"], "rejected")
        self.assertLess(result["models"]["60"]["overall"]["relative_error_reduction_pct"], 0)
        self.assertFalse(result["live_trading_allowed"])

    def test_modified_csv_or_inconsistent_saved_metrics_stop_review(self):
        with (self.source / "features_with_financials.csv").open("a", encoding="utf-8") as stream:
            stream.write("changed")
        with self.assertRaisesRegex(CollectionError, "해시"):
            review.run(self.source, self.root / "diagnostics", self.root / "registry")
        models = copy.deepcopy(self.models)
        models["models"]["5"]["test_metrics"]["mae_percentage_points"] = 999
        with self.assertRaisesRegex(CollectionError, "재계산"):
            review.diagnose(models, self.features, self.outcomes)

    def test_previously_reviewed_period_is_recorded_without_overwriting_evidence(self):
        _, first = review.run(self.source, self.root / "diagnostics", self.root / "registry")
        _, second = review.run(self.source, self.root / "diagnostics", self.root / "registry")
        self.assertFalse(first["overlaps_previous_review"])
        self.assertTrue(second["overlaps_previous_review"])
        self.assertEqual(len(list((self.root / "registry").glob("*.json"))), 2)
        self.assertFalse(first["fresh_unseen_evidence"])

    def test_actual_up_fraction_is_reported_instead_of_assuming_half(self):
        stats = review.aggregate([
            {"actual_change_pct": actual, "predicted_change_pct": -1, "as_of_date": "2026-01-01",
             "absolute_error_pp": abs(actual + 1), "baseline_absolute_error_pp": abs(actual)}
            for actual in (1, 2, 3, -1)
        ])
        self.assertEqual(stats["actual_up_fraction"], 0.75)
        self.assertEqual(stats["always_up_direction_accuracy"], 0.75)
        self.assertEqual(stats["direction_accuracy"], 0.25)
        self.assertEqual(stats["unique_dates"], 1)

    def test_standalone_cli_uses_existing_files_without_keys_or_network(self):
        output = io.StringIO()
        with patch("dart_client.stored_key", side_effect=AssertionError("no key access")), patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no network")), redirect_stdout(output):
            status = review.main(["--root", str(self.root), "--no-open"])
        self.assertEqual(status, 0)
        self.assertIn("자동 채택 모델: 0개", output.getvalue())


if __name__ == "__main__":
    unittest.main()
