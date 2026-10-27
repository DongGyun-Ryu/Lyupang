import json
import csv
import io
import tempfile
import unittest
from datetime import date
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from lyupang import Page
import research
from research import run_batch
from test_research_data import series, snapshot


class Source:
    def __init__(self):
        self.calls = []

    def fetch(self, code, end, **kwargs):
        self.calls.append(code)
        if code == "035420":
            raise RuntimeError("DO_NOT_PRINT_PRIVATE_RESPONSE")
        rows = [{"dt": b.date.replace("-", ""), "open_pric": str(b.open), "high_pric": str(b.high),
                 "low_pric": str(b.low), "cur_prc": str(b.close), "trde_qty": str(b.volume)} for b in reversed(series(code))]
        return Page(rows, False, None)


class ResearchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        self.reports = Path(self.temp.name) / "reports"
        bars = series()
        self.start, self.end = date.fromisoformat(bars[0].date), date.fromisoformat(bars[-1].date)
        self.existing = snapshot(bars, self.data)

    def test_reuses_verified_samsung_and_preserves_successes_when_another_stock_fails(self):
        source = Source()
        path, batch = run_batch(["005930", "005380", "035420"], self.start, self.end, self.data, self.reports,
            source_factory=lambda: source, use_dart=False, sleep=lambda _: None, progress=lambda _: None)
        self.assertEqual(source.calls, ["005380", "035420"])
        self.assertEqual(batch["status"], "partial")
        self.assertTrue(batch["stocks"][0]["reused"])
        self.assertEqual(batch["stocks"][1]["price_status"], "collected")
        self.assertEqual(batch["stocks"][2]["price_status"], "failed")
        self.assertTrue((self.existing / "daily.csv").exists())
        self.assertEqual(len(list(self.data.iterdir())), 2)
        for file in path.iterdir():
            if file.is_file():
                self.assertNotIn("DO_NOT_PRINT_PRIVATE_RESPONSE", file.read_text(encoding="utf-8-sig"))

    def test_missing_dart_key_is_reported_as_pending_not_zero_disclosures(self):
        factory = Mock(side_effect=AssertionError("must reuse cached data"))
        path, batch = run_batch(["005930"], self.start, self.end, self.data, self.reports,
                               source_factory=factory, progress=lambda _: None)
        factory.assert_not_called()
        self.assertEqual(batch["dart_state"], "waiting_key")
        self.assertEqual(batch["status"], "partial")
        self.assertFalse(batch["training_ready"])
        self.assertFalse(batch["trading_enabled"])
        self.assertIn("DART 인증키 미등록", (path / "report.html").read_text(encoding="utf-8"))

    def test_dart_financial_failure_keeps_disclosures_but_is_partial(self):
        dart = Mock()
        dart.companies.return_value = {"005930": {"corp_code": "00126380", "corp_name": "<script>unsafe</script>"}}
        dart.disclosures.return_value = [{"receipt_no": "20250301000001", "receipt_date": "2025-03-01"}]
        dart.annual_accounts.side_effect = RuntimeError("PRIVATE")
        path, batch = run_batch(["005930"], self.start, self.end, self.data, self.reports, dart=dart, progress=lambda _: None)
        self.assertEqual(batch["status"], "partial")
        payload = json.loads((path / "dart" / "005930.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["disclosures"]), 1)
        self.assertIsNone(payload["annual_financials"])
        self.assertNotIn("<script>unsafe</script>", (path / "report.html").read_text(encoding="utf-8"))
        self.assertNotIn("PRIVATE", (path / "batch.json").read_text(encoding="utf-8"))

    def test_report_write_failure_leaves_existing_data_and_no_completed_report(self):
        with patch("research.os.replace", side_effect=PermissionError("blocked")):
            with self.assertRaises(PermissionError):
                run_batch(["005930"], self.start, self.end, self.data, self.reports, use_dart=False, progress=lambda _: None)
        self.assertEqual(list(self.reports.iterdir()), [])
        self.assertTrue((self.existing / "daily.csv").exists())

    def test_successful_dart_keeps_annual_amounts_out_of_historical_features(self):
        dart = Mock()
        dart.companies.return_value = {"005930": {"corp_code": "00126380", "corp_name": "삼성전자"}}
        dart.disclosures.return_value = [{"receipt_no": "20250301000001", "receipt_date": "2025-03-01"}]
        dart.annual_accounts.return_value = {"status": "collected", "accounts": [{"thstrm_amount": "99999999999999"}],
                                            "use_in_historical_features": False}
        path, batch = run_batch(["005930"], self.start, self.end, self.data, self.reports, dart=dart, progress=lambda _: None)
        self.assertEqual(batch["status"], "collected")
        self.assertFalse(batch["training_ready"])
        with (path / "features.csv").open(encoding="utf-8-sig", newline="") as stream:
            features = list(csv.DictReader(stream))
        self.assertTrue(all(row["dart_available"] == "True" for row in features))
        self.assertNotIn("99999999999999", (path / "features.csv").read_text(encoding="utf-8-sig"))
        self.assertTrue((path / "dart" / "005930.json").exists())

    def test_cli_still_prepares_cached_prices_if_key_storage_cannot_be_read(self):
        with patch.object(research, "stored_key", side_effect=RuntimeError("PRIVATE")), redirect_stdout(io.StringIO()):
            status = research.main(["--codes", "005930", "--start", self.start.isoformat(), "--end", self.end.isoformat(),
                                    "--data-root", str(self.data), "--report-root", str(self.reports), "--no-open"])
        self.assertEqual(status, 2)
        folder, = list(self.reports.iterdir())
        batch = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
        self.assertEqual(batch["dart_state"], "error")
        self.assertGreater(batch["feature_rows"], 0)
        self.assertNotIn("PRIVATE", (folder / "report.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
