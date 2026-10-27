"""Use the installed official SDK with mocked HTTP; never contact a live account."""
import json
import unittest
from datetime import date
from unittest.mock import Mock

import requests
from kiwoom.core.client import KiwoomClient
from kiwoom.core.errors import APIError

from lyupang import CollectionError, KiwoomSource, collect


def response(rows, *, next_key=None, code="005930"):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps({"return_code": 0, "stk_cd": code, "stk_dt_pole_chart_qry": rows}).encode()
    result.headers.update({"cont-yn": "Y" if next_key else "N", "next-key": next_key or ""})
    return result


def bar(day):
    return {"dt": day, "open_pric": "100", "high_pric": "110", "low_pric": "90", "cur_prc": "105", "trde_qty": "1000"}


class SDKContractTests(unittest.TestCase):
    def test_real_sdk_encodes_fixed_read_endpoint_and_cursor(self):
        auth = Mock(mode="real")
        auth.authorization_header.return_value = "Bearer TEST_ONLY"
        client = KiwoomClient(auth)
        client.session = Mock()
        client.session.request.side_effect = [response([bar("20250908")], next_key="cursor"), response([bar("20250901")])]
        result = collect(KiwoomSource(client=client), "005930", date(2025, 9, 1), date(2025, 9, 8), delay=0)
        self.assertEqual(len(result.bars), 2)
        calls = client.session.request.call_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(call.kwargs["url"], "https://api.kiwoom.com/api/dostk/chart")
            self.assertEqual(call.kwargs["headers"]["api-id"], "ka10081")
            self.assertEqual(call.kwargs["json"], {"stk_cd": "005930", "base_dt": "20250908", "upd_stkpc_tp": "1"})
        self.assertEqual(calls[1].kwargs["headers"]["next-key"], "cursor")
        self.assertEqual(calls[1].kwargs["headers"]["cont-yn"], "Y")

    def test_response_for_different_stock_is_rejected(self):
        client = KiwoomClient(Mock(mode="real"))
        client.session = Mock()
        client.session.request.return_value = response([bar("20250908")], code="000660")
        with self.assertRaisesRegex(CollectionError, "종목"):
            KiwoomSource(client=client).fetch("005930", date(2025, 9, 8), adjusted=True, next_key=None)

    def test_sdk_business_error_is_not_interpreted_as_empty_success(self):
        client = KiwoomClient(Mock(mode="real"))
        client.session = Mock()
        bad = requests.Response()
        bad.status_code = 200
        bad._content = b'{"return_code": 20, "return_msg": "rate limit"}'
        client.session.request.return_value = bad
        with self.assertRaises(APIError):
            KiwoomSource(client=client).fetch("005930", date(2025, 9, 8), adjusted=True, next_key=None)


if __name__ == "__main__":
    unittest.main()
