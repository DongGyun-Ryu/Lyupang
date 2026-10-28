import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import train
from research import run_batch
from test_financial_history import original
from test_research_data import series, snapshot


class FakeDart:
    def __init__(self, missing_year=None):
        self.downloads = 0
        self.annual_calls = 0
        self.missing_year = missing_year

    def companies(self, codes):
        return {"005930": {"corp_code": "00126380", "corp_name": "삼성전자"}}

    def disclosures(self, corp, start, end, **kwargs):
        return [{"receipt_no": f"{year+1}0301000001", "receipt_date": f"{year+1}-03-01", "report_name": f"사업보고서 ({year}.12)"}
                for year in (2023, 2024, 2025) if start <= date(year+1, 3, 1) <= end]

    def annual_accounts(self, *args):
        self.annual_calls += 1
        return {"status": "collected", "accounts": [{"thstrm_amount": "999999999999999"}], "point_in_time_verified": False}

    def company_identifiers(self, *args):
        return {"00126380"}

    def original_xbrl(self, receipt):
        self.downloads += 1
        year = int(receipt[:4]) - 1
        return b"unsupported" if year == self.missing_year else original(year)


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        bars = series(count=600)
        snapshot(bars, self.root / "data")
        self.dart = FakeDart()
        self.input, _ = run_batch(["005930"], date.fromisoformat(bars[0].date), date.fromisoformat(bars[-1].date),
            self.root / "data", self.root / "research", source_factory=Mock(side_effect=AssertionError("cache expected")),
            dart=self.dart, progress=lambda _: None)

    def test_original_receipts_to_training_to_report_and_repeatable_cached_retraining(self):
        target, summary, models = train.run(self.input, self.root / "learning", self.root / "documents", self.dart, progress=lambda _: None)
        self.assertEqual(models["status"], "trained")
        self.assertGreater(summary["stocks"][0]["usable_rows"], 0)
        self.assertEqual(self.dart.annual_calls, 1)  # only the old research step used the current-value API.
        self.assertFalse(summary["production_training_ready"])
        self.assertFalse(summary["trading_enabled"])
        self.assertIn("연구용 모델 학습", (target / "report.html").read_text(encoding="utf-8"))
        self.assertNotIn("999999999999999", (target / "features_with_financials.csv").read_text(encoding="utf-8-sig"))
        downloads = self.dart.downloads
        second, _, second_models = train.run(self.input, self.root / "learning", self.root / "documents", self.dart, progress=lambda _: None)
        self.assertNotEqual(target, second)
        self.assertEqual(self.dart.downloads, downloads)
        self.assertEqual(models["models"], second_models["models"])
        json.loads((target / "models.json").read_text(encoding="utf-8"))

    def test_unreadable_latest_report_creates_explicit_exclusions_without_old_value_fallback(self):
        self.dart.missing_year = 2025
        target, summary, _ = train.run(self.input, self.root / "learning", self.root / "documents", self.dart, progress=lambda _: None)
        self.assertIn("일부 원문 제외", summary["stocks"][0]["warning"])
        with (target / "features_with_financials.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r["as_of_date"] >= "2026-03-02"]
        self.assertTrue(rows)
        self.assertTrue(all(r["financial_available"] == "False" and not r["debt_ratio_pct"] for r in rows))

    def test_cli_connects_existing_key_and_latest_research_without_recollecting_prices(self):
        output = io.StringIO()
        with patch.object(train, "stored_key", return_value="a" * 40), patch.object(train, "DartClient", return_value=self.dart), redirect_stdout(output):
            status = train.main(["--root", str(self.root), "--no-open"])
        self.assertEqual(status, 0)
        self.assertIn("연구용 모델 학습·평가 완료", output.getvalue())
        self.assertNotIn("a" * 40, output.getvalue())

    def test_original_connection_failure_is_not_described_as_insufficient_price_dates(self):
        self.dart.company_identifiers = Mock(side_effect=RuntimeError("PRIVATE"))
        _, summary, models = train.run(self.input, self.root / "learning", self.root / "documents", self.dart, progress=lambda _: None)
        self.assertEqual(models["status"], "blocked")
        self.assertIn("재무 원문 연결", models["reason"])
        self.assertNotIn("PRIVATE", summary["stocks"][0]["error"])


if __name__ == "__main__":
    unittest.main()
