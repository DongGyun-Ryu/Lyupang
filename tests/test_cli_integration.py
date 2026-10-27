import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from kiwoom.core.client import KiwoomClient

import collect as app
from lyupang import KiwoomSource


def response(day, next_key=None):
    response = requests.Response()
    response.status_code = 200
    response.headers.update({"cont-yn": "Y" if next_key else "N", "next-key": next_key or ""})
    response._content = json.dumps({
        "stk_cd": "005930", "return_code": 0,
        "stk_dt_pole_chart_qry": [{"dt": day, "open_pric": "100", "high_pric": "110", "low_pric": "90",
                                   "cur_prc": "105", "trde_qty": "1000"}],
    }).encode()
    return response


class ConsoleWorkflowTests(unittest.TestCase):
    def source(self, responses):
        auth = Mock(mode="real")
        auth.authorization_header.return_value = "Bearer DO_NOT_EXPORT"
        client = KiwoomClient(auth)
        client.session = Mock()
        client.session.request.side_effect = responses
        return KiwoomSource(client=client)

    def test_console_to_official_sdk_to_persisted_csv(self):
        source = self.source([response("20250908", "next"), response("20250901")])
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with patch.object(app, "KiwoomSource", return_value=source), redirect_stdout(output):
                status = app.main(["--code", "005930", "--start", "2025-09-01", "--end", "2025-09-08", "--output", tmp])
            self.assertEqual(status, 0)
            self.assertIn("수집·저장 완료: 2거래일", output.getvalue())
            folder, = list(Path(tmp).iterdir())
            with (folder / "daily.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["거래일"] for r in rows], ["2025-09-01", "2025-09-08"])
            for file in folder.iterdir():
                self.assertNotIn("DO_NOT_EXPORT", file.read_text(encoding="utf-8-sig"))

    def test_midstream_sdk_failure_has_no_success_snapshot(self):
        source = self.source([response("20250908", "next"), requests.ConnectionError("DO_NOT_EXPORT")])
        with tempfile.TemporaryDirectory() as tmp:
            output, errors = io.StringIO(), io.StringIO()
            with patch.object(app, "KiwoomSource", return_value=source), redirect_stdout(output), redirect_stderr(errors):
                status = app.main(["--code", "005930", "--start", "2025-09-01", "--end", "2025-09-08", "--output", tmp])
            self.assertEqual(status, 1)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertNotIn("수집·저장 완료", output.getvalue())
            self.assertNotIn("DO_NOT_EXPORT", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
