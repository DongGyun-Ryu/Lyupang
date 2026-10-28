import io
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import Mock

from financial_history import FinancialError, document_bytes, financial_as_of, parse_original, report_period


def original(year=2023, *, assets=1000, liabilities=400, revenue=200, income=20, entity="00126380", currency="KRW", extra="", scope=None, segment=False):
    xml = f'''<x:xbrl xmlns:x="http://www.xbrl.org/2003/instance" xmlns:ifrs="http://xbrl.ifrs.org/taxonomy/2024/ifrs-full" xmlns:iso="http://www.xbrl.org/2003/iso4217">
<x:context id="instant"><x:entity><x:identifier scheme="test">{entity}</x:identifier></x:entity><x:period><x:instant>{year}-12-31</x:instant></x:period></x:context>
<x:context id="duration"><x:entity><x:identifier scheme="test">{entity}</x:identifier></x:entity><x:period><x:startDate>{year}-01-01</x:startDate><x:endDate>{year}-12-31</x:endDate></x:period></x:context>
<x:context id="previous"><x:entity><x:identifier scheme="test">{entity}</x:identifier></x:entity><x:period><x:startDate>{year-1}-01-01</x:startDate><x:endDate>{year-1}-12-31</x:endDate></x:period></x:context>
<x:unit id="currency"><x:measure>iso:{currency}</x:measure></x:unit>
<ifrs:Assets contextRef="instant" unitRef="currency">{assets}</ifrs:Assets>
<ifrs:Liabilities contextRef="instant" unitRef="currency">{liabilities}</ifrs:Liabilities>
<ifrs:Revenue contextRef="duration" unitRef="currency">{revenue}</ifrs:Revenue>
<ifrs:ProfitLoss contextRef="duration" unitRef="currency">{income}</ifrs:ProfitLoss>
<ifrs:Revenue contextRef="previous" unitRef="currency">100</ifrs:Revenue>{extra}</x:xbrl>'''
    if scope or segment:
        xml = xml.replace('<x:xbrl ', '<x:xbrl xmlns:di="http://xbrl.org/2006/xbrldi" ')
        axis = "SegmentsAxis" if segment else "ConsolidatedAndSeparateFinancialStatementsAxis"
        member = "ConsolidatedMember" if scope == "CFS" else "SeparateMember"
        dimension = f'<x:segment><di:explicitMember dimension="ifrs:{axis}">ifrs:{member}</di:explicitMember></x:segment>'
        xml = xml.replace('</x:identifier>', '</x:identifier>' + dimension)
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as bundle:
        bundle.writestr("report.xbrl", xml)
    return result.getvalue()


def event(received, *, receipt="20230301000001", period="2022-12-31", status="verified", debt=40):
    return {"receipt_date": received, "receipt_no": receipt, "period_end": period, "status": status,
            "metrics": {"scope": "undimensioned", "debt_ratio_pct": debt, "net_margin_pct": 10, "revenue_yoy_pct": 20}}


class OriginalFinancialTests(unittest.TestCase):
    def test_extracts_company_period_unit_consistent_metrics(self):
        result = parse_original(original(), {"00126380"}, date(2023, 12, 31))
        self.assertEqual(result["debt_ratio_pct"], 40)
        self.assertEqual(result["net_margin_pct"], 10)
        self.assertEqual(result["revenue_yoy_pct"], 100)
        self.assertEqual(result["scope"], "undimensioned")

    def test_consolidated_scope_is_identified_and_segment_facts_are_excluded(self):
        self.assertEqual(parse_original(original(scope="CFS"), {"00126380"}, date(2023, 12, 31))["scope"], "CFS")
        with self.assertRaises(FinancialError):
            parse_original(original(segment=True), {"00126380"}, date(2023, 12, 31))

    def test_wrong_issuer_currency_period_or_conflicting_fact_is_rejected(self):
        cases = [(original(entity="99999999"), date(2023, 12, 31)),
                 (original(currency="USD"), date(2023, 12, 31)),
                 (original(), date(2022, 12, 31)),
                 (original(extra='<ifrs:Assets contextRef="instant" unitRef="currency">2000</ifrs:Assets>'), date(2023, 12, 31))]
        for payload, period in cases:
            with self.subTest(period=period), self.assertRaises(FinancialError):
                parse_original(payload, {"00126380"}, period)

    def test_future_and_same_day_correction_never_changes_earlier_feature(self):
        reports = [event("2023-03-01"), event("2024-06-01", receipt="20240601000001", debt=80)]
        before = financial_as_of(reports, date(2024, 5, 31))
        self.assertEqual(before["debt_ratio_pct"], 40)
        self.assertEqual(financial_as_of(reports, date(2024, 6, 1))["debt_ratio_pct"], 40)
        self.assertEqual(financial_as_of(reports, date(2024, 6, 2))["debt_ratio_pct"], 80)
        self.assertEqual(before["financial_receipt_no"], "20230301000001")

    def test_unreadable_latest_correction_never_falls_back_to_old_number(self):
        reports = [event("2023-03-01"), event("2024-06-01", receipt="20240601000001", status="unavailable")]
        result = financial_as_of(reports, date(2024, 6, 2))
        self.assertFalse(result["financial_available"])
        self.assertEqual(result["financial_reason"], "latest_report_unverified")
        self.assertIsNone(result["debt_ratio_pct"])

    def test_newer_financial_period_wins_and_same_day_ambiguity_is_not_guessed(self):
        reports = [event("2023-03-01"), event("2024-03-01", receipt="20240301000001", period="2023-12-31", debt=30),
                   event("2024-06-01", receipt="20240601000001", debt=80)]
        self.assertEqual(financial_as_of(reports, date(2024, 6, 2))["debt_ratio_pct"], 30)
        reports.append(event("2024-03-01", receipt="20240301000002", period="2023-12-31"))
        self.assertEqual(financial_as_of(reports, date(2024, 6, 2))["financial_reason"], "ambiguous_same_day_reports")

    def test_cache_reuse_verifies_hash_and_preserves_corrupted_original(self):
        client = Mock()
        client.original_xbrl.return_value = original()
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            first, digest = document_bytes(cache, "005930", "20240301000001", client)
            self.assertEqual(document_bytes(cache, "005930", "20240301000001", client), (first, digest))
            client.original_xbrl.assert_called_once()
            folder, = list(cache.iterdir())
            (folder / "original.zip").write_bytes(b"damaged")
            document_bytes(cache, "005930", "20240301000001", client)
            self.assertEqual(client.original_xbrl.call_count, 2)
            self.assertEqual((folder / "original.zip").read_bytes(), b"damaged")

    def test_period_parser_distinguishes_annual_report_from_other_notices(self):
        self.assertEqual(report_period("[기재정정]사업보고서 (2023.12)"), date(2023, 12, 31))
        with self.assertRaises(FinancialError):
            report_period("사업보고서 제출기한 연장신고서")


if __name__ == "__main__":
    unittest.main()
