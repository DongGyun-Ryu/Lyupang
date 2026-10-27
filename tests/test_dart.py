import io
import json
import unittest
import urllib.error
import urllib.parse
import zipfile
from datetime import date
from unittest.mock import Mock

from dart_client import DartClient, DartError, NoRedirect
from setup_dart import configure

KEY = "a" * 40
CORP = "00126380"


def client_with(*items):
    opener = Mock()
    opener.open.side_effect = [io.BytesIO(json.dumps(item).encode()) if isinstance(item, dict) else io.BytesIO(item) for item in items]
    return DartClient(KEY, opener=opener, delay=0), opener


def filing(number="20250102000001", day="20250102", name="사업보고서"):
    return {"corp_code": CORP, "rcept_no": number, "rcept_dt": day, "report_nm": name}


def page(number, rows, total=2, pages=2):
    return {"status": "000", "page_no": number, "total_count": total, "total_page": pages, "list": rows}


class DartTests(unittest.TestCase):
    def test_paginated_filings_preserve_amendments_and_request_originals(self):
        client, opener = client_with(page(1, [filing()]), page(2, [filing("20250103000001", "20250103", "[기재정정]사업보고서")]))
        records = client.disclosures(CORP, date(2025, 1, 1), date(2025, 1, 31))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["report_name"], "[기재정정]사업보고서")
        self.assertEqual(records[0]["available_from"], "2025-01-03T00:00:00+09:00")
        for call in opener.open.call_args_list:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(call.args[0].full_url).query)
            self.assertEqual(query["last_reprt_at"], ["N"])
            self.assertEqual(query["crtfc_key"], [KEY])

    def test_missing_count_duplicate_or_changed_company_fails(self):
        cases = [page(1, [filing()], total=2, pages=1),
                 page(1, [filing(), filing()], total=2, pages=1),
                 page(1, [{**filing(), "corp_code": "00000001"}], total=1, pages=1)]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(DartError):
                client_with(data)[0].disclosures(CORP, date(2025, 1, 1), date(2025, 1, 31))

    def test_empty_list_is_different_from_invalid_key(self):
        self.assertEqual(client_with({"status": "013"})[0].disclosures(CORP, date(2025, 1, 1), date(2025, 1, 31)), [])
        client, _ = client_with({"status": "010", "message": "secret " + KEY})
        with self.assertRaises(DartError) as error:
            client.disclosures(CORP, date(2025, 1, 1), date(2025, 1, 31))
        self.assertNotIn(KEY, str(error.exception))

    def test_http_errors_do_not_echo_authenticated_url_and_redirects_are_blocked(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError("https://example.test/?crtfc_key=" + KEY, 403, KEY, {}, None)
        with self.assertRaises(DartError) as error:
            DartClient(KEY, opener=opener).validate_key()
        self.assertNotIn(KEY, str(error.exception))
        with self.assertRaises(DartError):
            NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.test")

    def test_proxy_binding_placeholder_is_not_rejected_as_a_raw_key(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'{"status":"000","corp_code":"00126380"}')
        DartClient("proxy_binding_placeholder", opener=opener).validate_key()
        opener.open.assert_called_once()

    def test_company_zip_maps_selected_codes_without_extracting_files(self):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as z:
            z.writestr("CORPCODE.xml", f"<result><list><corp_code>{CORP}</corp_code><stock_code>005930</stock_code><corp_name>삼성전자</corp_name></list><list><corp_code>00000001</corp_code><stock_code></stock_code></list></result>")
        client, _ = client_with(raw.getvalue())
        self.assertEqual(client.companies(["005930"]), {"005930": {"corp_code": CORP, "corp_name": "삼성전자"}})

    def test_financial_fallback_is_not_marked_as_point_in_time_input(self):
        row = {"corp_code": CORP, "rcept_no": "20260301000001", "bsns_year": "2025", "reprt_code": "11011",
               "account_id": "ifrs-full_Revenue", "thstrm_amount": "100,000"}
        client, _ = client_with({"status": "013"}, {"status": "000", "list": [row]})
        result = client.annual_accounts(CORP, 2025)
        self.assertEqual(result["fs_div"], "OFS")
        self.assertFalse(result["point_in_time_verified"])
        self.assertFalse(result["use_in_historical_features"])
        self.assertEqual(result["accounts"][0]["thstrm_amount"], "100,000")


class KeySetupTests(unittest.TestCase):
    def test_new_key_is_validated_before_secure_storage_and_not_echoed(self):
        backend = Mock()
        backend.get_password.side_effect = [None, KEY]
        factory = Mock()
        output = configure(backend, prompt=lambda _: KEY, client_factory=factory)
        factory.return_value.validate_key.assert_called_once()
        backend.set_password.assert_called_once_with("Lyupang/OpenDART", "api_key", KEY)
        self.assertNotIn(KEY, output)

    def test_bad_new_key_never_replaces_old_key(self):
        backend = Mock()
        backend.get_password.return_value = "OLD"
        factory = Mock(side_effect=DartError("invalid"))
        with self.assertRaises(DartError):
            configure(backend, replace=True, prompt=lambda _: KEY, client_factory=factory)
        backend.set_password.assert_not_called()

    def test_existing_working_key_is_reused_without_prompting(self):
        backend = Mock()
        backend.get_password.return_value = KEY
        factory, prompt = Mock(), Mock()
        output = configure(backend, client_factory=factory, prompt=prompt)
        prompt.assert_not_called()
        backend.set_password.assert_not_called()
        self.assertNotIn(KEY, output)
        factory.return_value.validate_key.assert_called_once()


if __name__ == "__main__":
    unittest.main()
