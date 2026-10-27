import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import collect as entry
from lyupang import CollectionError, DailyBar, Page, SEOUL, collect, default_period, save_collection


def row(day="20250908", **overrides):
    return {"dt": day, "open_pric": "69800", "high_pric": "70500", "low_pric": "69600",
            "cur_prc": "70100", "trde_qty": "9263135", "trde_prica": "648525", **overrides}


class Source:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []

    def fetch(self, code, end, *, adjusted, next_key):
        self.calls.append((code, end, adjusted, next_key))
        value = next(self.pages)
        if isinstance(value, Exception):
            raise value
        return value


START, END = date(2025, 9, 1), date(2025, 9, 8)


class CollectorTests(unittest.TestCase):
    def test_pagination_overlap_sort_filter_and_boundary_stop(self):
        source = Source([
            Page([row("20250908"), row("20250905")], True, "cursor-one"),
            Page([row("20250905"), row("20250904"), row("20250901"), row("20250829")], True, "unused"),
        ])
        sleeps = []
        result = collect(source, "005930", START, END, sleep=sleeps.append)
        self.assertEqual([b.date for b in result.bars], ["2025-09-01", "2025-09-04", "2025-09-05", "2025-09-08"])
        self.assertEqual([call[3] for call in source.calls], [None, "cursor-one"])
        self.assertEqual(sleeps, [0.6])
        self.assertEqual(result.stop_reason, "requested_start_reached")

    def test_conflicting_duplicate_is_not_accepted(self):
        source = Source([Page([row("20250908")], True, "next"),
                         Page([row("20250908", cur_prc="70000")], False, None)])
        with self.assertRaisesRegex(CollectionError, "서로 다른"):
            collect(source, "005930", START, END, delay=0)

    def test_bad_cursors_and_nonprogress_are_errors(self):
        scenarios = [
            [Page([row()], True, None)],
            [Page([row()], True, "same"), Page([row("20250905")], True, "same")],
            [Page([row()], True, "a"), Page([row()], True, "b")],
            [Page([], True, "a")],
        ]
        for pages in scenarios:
            with self.subTest(pages=pages), self.assertRaises(CollectionError):
                collect(Source(pages), "005930", START, END, delay=0)

    def test_page_limit_and_midstream_error_do_not_become_success(self):
        with self.assertRaisesRegex(CollectionError, "최대 조회"):
            collect(Source([Page([row()], True, "a")]), "005930", START, END, max_pages=1)
        with self.assertRaises(ConnectionError):
            collect(Source([Page([row()], True, "a"), ConnectionError("offline")]), "005930", START, END, delay=0)

    def test_new_listing_history_exhaustion_is_recorded_without_filling_rows(self):
        result = collect(Source([Page([row()], False, None)]), "005930", START, END)
        self.assertEqual(len(result.bars), 1)
        self.assertEqual(result.stop_reason, "source_history_exhausted")

    def test_out_of_order_and_out_of_range_responses_are_rejected(self):
        for rows in ([row("20250905"), row("20250908")], [row("20250909")]):
            with self.subTest(rows=rows), self.assertRaises(CollectionError):
                collect(Source([Page(rows, False, None)]), "005930", START, END)

    def test_empty_result_is_not_success(self):
        with self.assertRaisesRegex(CollectionError, "데이터가 없습니다"):
            collect(Source([Page([], False, None)]), "005930", START, END)

    def test_numeric_normalization_and_volume_quality_flags(self):
        bar = DailyBar.from_wire("005930", row(open_pric="-69800", cur_prc="+70100", trde_qty="4294967295"))
        self.assertEqual((bar.open, bar.close, bar.volume_capped), (69800, 70100, True))
        zero = DailyBar.from_wire("005930", row(open_pric="0", high_pric="0", low_pric="0", cur_prc="0", trde_qty="0"))
        self.assertTrue(zero.zero_volume)
        for changes in ({"trde_qty": "-1"}, {"low_pric": "80000"}, {"cur_prc": ""}, {"cur_prc": "=1+1"}, {"dt": "20250230"}):
            with self.subTest(changes=changes), self.assertRaises(CollectionError):
                DailyBar.from_wire("005930", row(**changes))

    def test_invalid_request_never_contacts_broker(self):
        for code, start, end in (("=1+1", START, END), ("005930", END, START), ("005930", START, date(9999, 1, 1))):
            source = Source([])
            with self.assertRaises(CollectionError):
                collect(source, code, start, end)
            self.assertEqual(source.calls, [])

    def test_default_period_excludes_today_and_handles_leap_day(self):
        self.assertEqual(default_period(date(2024, 3, 1)), (date(2021, 2, 28), date(2024, 2, 29)))

    def test_snapshots_csv_hash_units_and_repeatability(self):
        result = collect(Source([Page([row()], False, None)]), "005930", START, END)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            now = datetime(2025, 9, 9, 10, tzinfo=SEOUL)
            first = save_collection(result, directory, now=now)
            original = (first / "daily.csv").read_bytes()
            second = save_collection(result, directory, now=now)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "daily.csv").read_bytes(), original)
            manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["csv_sha256"], hashlib.sha256(original).hexdigest())
            self.assertFalse(manifest["point_in_time_training_ready"])
            self.assertFalse(manifest["quality"]["calendar_verified"])
            self.assertEqual(manifest["adjustment_base_date"], END.isoformat())
            with (first / "daily.csv").open(encoding="utf-8-sig", newline="") as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(saved[0]["종목코드"], "005930")
            self.assertEqual(saved[0]["거래대금_백만원"], "648525")
            self.assertEqual(len(list(directory.iterdir())), 2)

    def test_disk_failure_cleans_only_new_staging_folder(self):
        result = collect(Source([Page([row()], False, None)]), "005930", START, END)
        with tempfile.TemporaryDirectory() as tmp:
            keep = Path(tmp) / "existing.csv"
            keep.write_text("keep", encoding="utf-8")
            with patch("lyupang.os.replace", side_effect=PermissionError("blocked")):
                with self.assertRaises(PermissionError):
                    save_collection(result, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [keep])
            self.assertEqual(keep.read_text(), "keep")

    def test_cli_failure_does_not_write_or_echo_raw_sdk_secrets(self):
        output = io.StringIO()
        with patch.object(entry, "KiwoomSource", side_effect=RuntimeError("SECRET_TOKEN")), patch.object(entry, "save_collection") as save:
            with redirect_stderr(output), redirect_stdout(io.StringIO()):
                code = entry.main(["--code", "005930", "--start", START.isoformat(), "--end", END.isoformat()])
            self.assertEqual(code, 1)
            save.assert_not_called()
            self.assertNotIn("SECRET_TOKEN", output.getvalue())


if __name__ == "__main__":
    unittest.main()
