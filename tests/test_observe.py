import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import observe
from learning import index_outcomes, metrics, partition_examples, train_models
from lyupang import CollectionError, DailyBar, Page, SEOUL
from test_learning import dataset
from test_review import save_source


class Prices:
    def __init__(self, bars):
        self.bars = bars

    def fetch(self, code, end, **kwargs):
        return Page([{"dt": b.date.replace("-", ""), "open_pric": b.open, "high_pric": b.high,
                      "low_pric": b.low, "cur_prc": b.close, "trde_qty": b.volume}
                     for b in reversed(self.bars) if b.date <= end.isoformat()], False, None)


class ObserveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "learning" / "sample"
        self.features, self.outcomes = dataset()
        self.models = train_models(self.features, self.outcomes)
        bad = self.models["models"]["60"]
        bad["coefficients"] = [-100 * v for v in bad["coefficients"]]
        parts, _ = partition_examples(self.features, index_outcomes(self.outcomes), 60,
                                      self.models["split"]["validation_start"], self.models["split"]["test_start"])
        bad["test_metrics"] = metrics(bad, parts["test"])
        save_source(self.source, self.models, self.features, self.outcomes)
        summary = observe.read_json(self.source / "summary.json")
        summary["created_at"] = "2026-10-19T15:00:00+09:00"
        (self.source / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        first_day = date(2026, 10, 19) - timedelta(days=69)
        self.bars = [DailyBar("005930", (first_day + timedelta(days=i)).isoformat(),
                              100+i, 115+i, 90+i, 105+i, 1000+i, None, False, False) for i in range(78)]
        self.now = datetime(2026, 10, 20, 8, 30, tzinfo=SEOUL)
        self.prices = Prices(self.bars)
        self.dart = Mock()
        self.dart.companies.return_value = {"005930": {"corp_code": "00126380"}}
        self.dart.disclosures.return_value = []
        self.events = [{"status": "verified", "period_end": "2025-12-31", "receipt_date": "2026-03-10",
                        "receipt_no": "20260310000001", "metrics": {"scope": "CFS", "debt_ratio_pct": 40,
                        "net_margin_pct": 10, "revenue_yoy_pct": 5}}]

    def cycle(self, **kwargs):
        with patch("observe.audit_reports", return_value=self.events):
            return observe.run_cycle(self.root, self.source, source_factory=lambda: self.prices,
                                     dart_factory=lambda: self.dart, clock=lambda: self.now,
                                     progress=lambda _: None, sleep=lambda _: None, **kwargs)

    def forecast(self):
        experiment, summary = self.cycle()
        predictions = observe.records(experiment, "predictions")
        key = next(k for k, p in predictions.items() if p["horizon"] == 5)
        return experiment, key, predictions[key], observe.read_record(experiment / "commits" / (key + ".json"))

    def test_records_forward_predictions_without_refit_excludes_rejected_and_deduplicates(self):
        original = {p.name: p.read_bytes() for p in self.source.iterdir()}
        with patch("learning.fit", side_effect=AssertionError("no fitting")):
            experiment, first = self.cycle()
            saved = {p.name: p.read_bytes() for p in (experiment / "predictions").iterdir()}
            _, second = self.cycle()
        self.assertEqual(first["new_predictions"], 2)
        self.assertEqual(first["excluded_horizons"], ["60"])
        self.assertEqual(second["new_predictions"], 0)
        self.assertEqual(second["pending"], 2)
        self.assertFalse(second["automatic_promotion"])
        self.assertFalse(second["live_trading_allowed"])
        self.assertEqual(saved, {p.name: p.read_bytes() for p in (experiment / "predictions").iterdir()})
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.assertTrue(all(p["as_of_date"] == "2026-10-19" for p in observe.records(experiment, "predictions").values()))

    def test_after_open_initialization_waits_without_network_or_keys(self):
        self.now = self.now.replace(hour=11)
        with patch("observe.KiwoomSource", side_effect=AssertionError("no prices")), patch("observe.stored_key", side_effect=AssertionError("no keys")):
            experiment, summary = observe.run_cycle(self.root, self.source, clock=lambda: self.now)
        self.assertEqual(summary["total_predictions"], 0)
        self.assertEqual(summary["errors"], [])
        self.assertIn("새 예측 대기", (experiment / "report.html").read_text(encoding="utf-8"))

    def test_reconciles_next_open_to_fifth_close_once_and_keeps_forecast_bytes(self):
        experiment, key, prediction, commit = self.forecast()
        stored = (experiment / "predictions" / (key + ".json")).read_bytes()
        self.now = datetime(2026, 10, 27, 11, tzinfo=SEOUL)
        self.dart.companies.side_effect = AssertionError("no DART needed to settle")
        _, summary = self.cycle()
        result = observe.read_record(experiment / "results" / (key + ".json"))
        expected = (self.bars[74].close / self.bars[70].open - 1) * 100
        self.assertAlmostEqual(result["actual_change_pct"], expected)
        self.assertEqual(result["entry_date"], "2026-10-20")
        self.assertEqual(result["exit_date"], "2026-10-24")  # Synthetic daily series, not an exchange calendar.
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["metrics"]["5"]["rows"], 1)
        receipt = (experiment / "results" / (key + ".json")).read_bytes()
        self.cycle()
        self.assertEqual(stored, (experiment / "predictions" / (key + ".json")).read_bytes())
        self.assertEqual(receipt, (experiment / "results" / (key + ".json")).read_bytes())

    def test_late_forecasts_missing_commit_and_write_crossing_open_are_excluded(self):
        _, _, prediction, commit = self.forecast()
        self.assertEqual(observe.settle(prediction, self.bars, None)["reason"], "commit_missing")
        late = copy.deepcopy(prediction)
        late["recorded_at"] = "2026-10-20T09:01:00+09:00"
        late_commit = {"prediction_sha256": observe.digest(late), "committed_at": late["recorded_at"], "before_cutoff": True}
        self.assertEqual(observe.settle(late, self.bars, late_commit)["reason"], "recorded_after_entry")
        crossing = {**commit, "committed_at": "2026-10-20T09:00:01+09:00", "before_cutoff": False}
        self.assertEqual(observe.settle(prediction, self.bars, crossing)["reason"], "commit_after_cutoff")

    def test_revisions_missing_history_and_zero_volume_are_not_scored(self):
        _, _, prediction, commit = self.forecast()
        changed = self.bars[:]
        changed[69] = replace(changed[69], close=changed[69].close * 2)
        self.assertEqual(observe.settle(prediction, changed, commit)["reason"], "price_history_revised")
        self.assertEqual(observe.settle(prediction, self.bars[60:], commit)["reason"], "anchor_history_missing")
        changed = self.bars[:]
        changed[71] = replace(changed[71], volume=0, zero_volume=True)
        self.assertEqual(observe.settle(prediction, changed, commit)["reason"], "future_price_quality")
        self.assertIsNone(observe.settle(prediction, self.bars[:73], commit))

    def test_frozen_models_continue_if_original_source_changes_but_frozen_corruption_stops(self):
        experiment, _ = self.cycle()
        (self.source / "models.json").write_text("{}", encoding="utf-8")
        _, summary = self.cycle()
        self.assertEqual(summary["total_predictions"], 2)
        (experiment / "models.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(CollectionError, "고정한 모델"):
            self.cycle()

    def test_private_network_failures_are_redacted_and_dart_failure_still_allows_results(self):
        experiment, _, _, _ = self.forecast()
        self.now = datetime(2026, 10, 27, 8, 30, tzinfo=SEOUL)
        self.dart.companies.side_effect = RuntimeError("PRIVATE_KEY_SHOULD_NOT_APPEAR")
        _, summary = self.cycle()
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["new_predictions"], 0)
        self.assertTrue(summary["errors"])
        self.assertNotIn("PRIVATE_KEY_SHOULD_NOT_APPEAR", (experiment / "report.html").read_text(encoding="utf-8"))

    def test_clock_crossing_cutoff_during_calculation_never_backdates_prediction(self):
        self.cycle()
        experiment = self.root / "forward" / "experiment"
        manifest = observe.read_record(experiment / "experiment.json")
        models = observe.read_json(experiment / "models.json")
        saved = observe.records(experiment, "predictions")
        feature = next(iter(saved.values()))["feature"]
        feature = {**feature, "as_of_date": "2026-10-20", "available_from": "2026-10-21T00:00:00+09:00"}
        stamps = iter([datetime(2026, 10, 21, 8, 59, tzinfo=SEOUL), datetime(2026, 10, 21, 9, tzinfo=SEOUL)])
        count = observe.record_predictions(experiment, manifest, models, feature, self.bars[:71], {}, clock=lambda: next(stamps))
        self.assertEqual(count, 0)
        self.assertEqual(len(observe.records(experiment, "predictions")), 2)

    def test_disk_write_completing_after_cutoff_leaves_an_excluded_receipt(self):
        experiment, _, _, _ = self.forecast()
        manifest = observe.read_record(experiment / "experiment.json")
        models = observe.read_json(experiment / "models.json")
        original = next(iter(observe.records(experiment, "predictions").values()))
        feature = {**original["feature"], "as_of_date": "2026-10-20", "available_from": "2026-10-21T00:00:00+09:00"}
        stamps = iter([datetime(2026, 10, 21, 8, 59, 59, tzinfo=SEOUL)] * 2 +
                      [datetime(2026, 10, 21, 9, 0, 1, tzinfo=SEOUL)] * 2)
        count = observe.record_predictions(experiment, manifest, models, feature, self.bars[:71], {}, clock=lambda: next(stamps))
        self.assertEqual(count, 1)
        key = "2026-10-20_005930_5"
        prediction = observe.read_record(experiment / "predictions" / (key + ".json"))
        receipt = observe.read_record(experiment / "commits" / (key + ".json"))
        self.assertFalse(receipt["before_cutoff"])
        self.assertEqual(observe.settle(prediction, self.bars, receipt)["reason"], "commit_after_cutoff")

    def test_unlocked_lock_file_from_prior_process_does_not_block_restart(self):
        forward = self.root / "forward"
        forward.mkdir()
        (forward / ".running.lock").write_bytes(b"0")
        _, summary = self.cycle()
        self.assertEqual(summary["new_predictions"], 2)

    def test_lock_and_modified_prediction_prevent_silent_replacement(self):
        with observe.journal_lock(self.root / "forward"):
            with self.assertRaisesRegex(CollectionError, "실행 중"):
                self.cycle()
        experiment, key, _, _ = self.forecast()
        path = experiment / "predictions" / (key + ".json")
        changed = observe.read_json(path)
        changed["payload"]["predicted_change_pct"] = 99999
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(CollectionError, "해시"):
            self.cycle()

    def test_missing_financials_stop_new_predictions_and_summary_escapes_errors(self):
        self.events = []
        experiment, summary = self.cycle()
        self.assertEqual(summary["new_predictions"], 0)
        self.assertTrue(summary["errors"])
        summary["errors"] = ["<script>bad</script>"]
        self.assertNotIn("<script>bad</script>", observe.render_report(summary))

    def test_cli_initializes_after_open_and_daily_schedule_is_korean_time(self):
        output = io.StringIO()
        self.now = self.now.replace(hour=11)
        run_cycle = observe.run_cycle
        with patch("observe.run_cycle", side_effect=lambda *a, **kw: run_cycle(self.root, self.source, clock=lambda: self.now)), redirect_stdout(output):
            status = observe.main(["--root", str(self.root), "--no-open"])
        self.assertEqual(status, 0)
        self.assertIn("자동 채택 모델: 0개", output.getvalue())
        self.assertEqual(observe.next_check(self.now).isoformat(), "2026-10-21T08:30:00+09:00")
        self.assertEqual(observe.next_check(self.now.replace(hour=8, minute=10)).hour, 8)
        self.assertEqual(observe.next_check(self.now.replace(hour=8, minute=40), failed=True).minute, 45)


if __name__ == "__main__":
    unittest.main()
