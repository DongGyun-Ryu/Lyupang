import csv
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from lyupang import Collection, DailyBar, save_collection
from research_data import build_cases, find_cached, load_snapshot


def series(code="005930", count=140):
    return [DailyBar(code, (date(2025, 1, 1) + timedelta(days=i)).isoformat(), 100 + i, 115 + i, 90 + i,
                     105 + i, 1000 + i, None, False, False) for i in range(count)]


def snapshot(bars, root):
    return save_collection(Collection(bars, 1, "requested_start_reached", bars[0].code,
                                     date.fromisoformat(bars[0].date), date.fromisoformat(bars[-1].date), True), root)


class ResearchDataTests(unittest.TestCase):
    def test_reuse_checks_date_basis_and_csv_hash(self):
        bars = series()
        start, end = date.fromisoformat(bars[0].date), date.fromisoformat(bars[-1].date)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = snapshot(bars, root)
            metadata, loaded = load_snapshot(folder)
            self.assertEqual(loaded, bars)
            self.assertEqual(find_cached(root, "005930", start, end), (folder, bars))
            self.assertIsNone(find_cached(root, "005930", start, end - timedelta(days=1)))
            with (folder / "daily.csv").open("a", encoding="utf-8") as stream:
                stream.write("corrupt")
            self.assertIsNone(find_cached(root, "005930", start, end))

    def test_features_do_not_change_when_future_prices_or_volume_change(self):
        bars = series()
        features, outcomes = build_cases(bars)
        changed = bars[:80] + [replace(b, open=b.open * 2, high=b.high * 2, low=b.low * 2,
                                      close=b.close * 2, volume=b.volume * 5) for b in bars[80:]]
        new_features, new_outcomes = build_cases(changed)
        original = next(f for f in features if f["as_of_date"] == bars[75].date)
        altered = next(f for f in new_features if f["as_of_date"] == bars[75].date)
        self.assertEqual(original, altered)
        self.assertNotEqual(next(o for o in outcomes if o["as_of_date"] == bars[75].date and o["observed_sessions"] == 20),
                            next(o for o in new_outcomes if o["as_of_date"] == bars[75].date and o["observed_sessions"] == 20))

    def test_outcomes_use_next_open_not_observation_close_and_keep_pending(self):
        bars = series()
        features, outcomes = build_cases(bars)
        first = outcomes[0]
        self.assertEqual(first["entry_date"], bars[61].date)
        self.assertEqual(first["price_return_pct"], round((bars[65].close / bars[61].open - 1) * 100, 6))
        self.assertTrue(all(row["status"] == "pending" and row["price_return_pct"] is None for row in outcomes[-3:]))
        self.assertNotIn("entry_open", features[0])
        self.assertNotIn("price_return_pct", features[0])

    def test_missing_dart_is_not_zero_and_same_day_and_future_filings_are_excluded(self):
        bars = series()
        anchor = date.fromisoformat(bars[60].date)
        filings = [{"receipt_date": (anchor + timedelta(days=shift)).isoformat()} for shift in (-8, -2, 0, 2)]
        features, _ = build_cases(bars, filings)
        self.assertEqual(features[0]["disclosures_previous_7_days"], 1)
        self.assertTrue(features[0]["dart_available"])
        missing, _ = build_cases(bars)
        self.assertIsNone(missing[0]["disclosures_previous_7_days"])
        self.assertFalse(missing[0]["dart_available"])
        empty, _ = build_cases(bars, [])
        self.assertEqual(empty[0]["disclosures_previous_7_days"], 0)
        self.assertTrue(empty[0]["dart_available"])

    def test_suspended_or_capped_future_window_is_not_a_trade_result(self):
        bars = series()
        bars[62] = replace(bars[62], volume=0, zero_volume=True)
        _, outcomes = build_cases(bars)
        self.assertEqual(outcomes[0]["status"], "quality_issue")
        self.assertIsNone(outcomes[0]["price_return_pct"])


if __name__ == "__main__":
    unittest.main()
