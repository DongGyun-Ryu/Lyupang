import hashlib
import html
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import business
import facts
import report_claims as rc
from lyupang import CollectionError
from test_business import CORP, archive, filing, xml_document


class MeasurementTests(unittest.TestCase):
    def test_literal_money_scaling_is_decimal_exact_and_keeps_currencies_separate(self):
        cases = [("매출 333조 6,059억원", "333605900000000", "KRW"),
                 ("매출 1.25조원", "1250000000000.00", "KRW"),
                 ("비용 125백만원", "125000000", "KRW"),
                 ("투자 40억달러", "4000000000", "USD"),
                 ("비용 -100억원입니다.", "-10000000000", "KRW"),
                 ("판매 414만대입니다.", "4140000", "대")]
        for text, value, unit in cases:
            with self.subTest(text=text):
                q = rc.measurements(text)[0]
                self.assertEqual((q["normalized_value"], q["unit"]), (value, unit))
                self.assertEqual(text[q["start"]:q["end"]], q["text"])
        big = rc.measurements("매출 -123456789012345678901234조원입니다.")[0]
        self.assertEqual(big["normalized_value"], "-123456789012345678901234000000000000")

    def test_percent_points_and_signs_do_not_get_converted_into_each_other(self):
        values = rc.measurements("매출은 -10.9%, 이익률은 2.3%p입니다.")
        self.assertEqual([(v["normalized_value"], v["unit"]) for v in values], [("-10.9", "percent"), ("2.3", "percentage_points")])
        self.assertIsNone(rc.measurements("매출 △10.9% 감소")[0]["normalized_value"])
        self.assertEqual(facts.display_value(rc.measurements("매출 -0.5% 감소")[0]), "-0.5 %")

    def test_ranges_missing_units_and_bad_unit_order_are_not_guessed(self):
        for text in ("매출 100~120억원입니다.", "매출 100-120억원입니다.", "투자 1억 2조원입니다.", "영업권 78억 등입니다."):
            with self.subTest(text=text):
                values = rc.measurements(text)
                self.assertTrue(values)
                self.assertTrue(all(q["normalized_value"] is None for q in values))
        self.assertFalse(rc.measurements("매출 1,23억원입니다."))
        approx = rc.measurements("투자는 약 1조원입니다.")[0]
        self.assertIn("approximate_or_bound", approx["qualifiers"])
        parenthesis = rc.measurements("비용 (100억원)입니다.")[0]
        self.assertEqual(parenthesis["normalized_value"], "10000000000")
        self.assertIn("parentheses_no_negative_inference", parenthesis["qualifiers"])

    def test_metric_binding_does_not_assign_one_percentage_to_two_metrics(self):
        sample = rc.structure_sentence("2025년 당사의 매출은 100억원으로 전년 대비 10.9% 증가하였습니다.", [], "예시회사")
        self.assertEqual([q["metric"] for q in sample["measurements"]], ["revenue", "revenue"])
        self.assertEqual(sample["measurements"][0]["relation"], "reported_amount_or_count")
        self.assertEqual(sample["measurements"][1]["relation"], "relative_change_candidate")
        mixed = rc.structure_sentence("2025년 당사의 매출과 영업이익은 10% 증가하였습니다.", [], "예시회사")
        self.assertIsNone(mixed["measurements"][0]["metric"])
        self.assertEqual(mixed["status"], "needs_review")
        separate = rc.structure_sentence("2025년 당사의 매출은 100억원, 영업이익은 10억원입니다.", [], "예시회사")
        self.assertEqual([q["metric"] for q in separate["measurements"]], ["revenue", "operating_profit"])
        income = rc.structure_sentence("2025년 당사의 영업수익은 100억원, 매출총이익은 10억원입니다.", [], "예시회사")
        self.assertEqual([q["metric"] for q in income["measurements"]], ["operating_revenue", "gross_profit"])
        ratio_change = rc.structure_sentence("2025년 당사의 영업이익률은 10% 상승했습니다.", [], "예시회사")
        self.assertEqual(ratio_change["status"], "needs_review")


class ClaimTests(unittest.TestCase):
    def test_causal_background_metric_is_not_assigned_partner_cost_growth(self):
        text = "영업비용을 항목별로 나누어보면, 개발/운영비는 신규 채용에 따른 인원 증가 등으로 8.7% 상승하였고, 파트너비의 경우, 매출 증가의 영향 등으로 전년 대비 10.8% 상승하였습니다."
        result = rc.structure_sentence(text, [], "NAVER")
        quantity = result["measurements"][-1]
        self.assertEqual(quantity["normalized_value"], "10.8")
        self.assertIsNone(quantity["metric"])
        self.assertEqual(quantity["metric_binding"], "metric_in_causal_context")
        for blocker in quantity["binding_blockers"]:
            self.assertEqual(text[blocker["start"]:blocker["end"]], blocker["text"])

    def test_actual_report_patent_counts_are_not_investment_amounts(self):
        text = "2025년 중 총 38조원의 R&D투자를 통해 국내 특허 10,639건, 미국 특허 10,347건 등을 등록하였습니다."
        result = rc.structure_sentence(text, ["II. 사업의 내용"], "삼성전자")
        patents = [v for v in result["measurements"] if v["unit"] == "건"]
        self.assertEqual(len(patents), 2)
        self.assertTrue(all(v["metric"] is None for v in patents))
        self.assertIn(patents[0]["metric_binding"], {"metric_unit_mismatch", "different_quantity_role"})
        self.assertEqual(result["status"], "needs_review")
        for sample, metric in (("2025년 당사의 투자는 38조원입니다.", "investment"),
                               ("2025년 당사의 판매량은 414만대입니다.", "sales_volume")):
            quantity = rc.structure_sentence(sample, [], "삼성전자")["measurements"][0]
            self.assertEqual(quantity["metric"], metric)
        for sample in ("2025년 당사의 매출은 100명입니다.", "2025년 당사의 판매량은 100억원입니다."):
            self.assertIsNone(rc.structure_sentence(sample, [], "삼성전자")["measurements"][0]["metric"])

    def test_investment_does_not_bind_asset_increase_in_actual_report(self):
        text = "자산의 주요 증가 원인은 단기금융상품 등의 유동자산의 증가 20조 6,223억원 및반도체ㆍ디스플레이 패널 사업의 대규모 시설투자에 따른 유형자산의 증가 9조 3,596억원 등입니다."
        result = rc.structure_sentence(text, ["II. 사업의 내용"], "삼성전자")
        self.assertTrue(all(v["metric"] is None for v in result["measurements"]))
        last = result["measurements"][-1]
        self.assertEqual(last["normalized_value"], "9359600000000")
        self.assertEqual(last["metric_binding"], "different_quantity_role")
        for blocker in last["binding_blockers"]:
            self.assertEqual(text[blocker["start"]:blocker["end"]], blocker["text"])

    def test_topic_only_titles_do_not_count_as_statements(self):
        for text in ("바. 수주상황", "(2) 주요 투자 현황", "가. 매출실적"):
            self.assertIsNone(rc.structure_sentence(text, ["II. 사업의 내용"], "삼성전자"))
        self.assertIsNotNone(rc.structure_sentence("2025년 당사는 수주 100억원을 기록했습니다.", [], "삼성전자"))

    def test_reported_company_amount_and_growth_have_exact_evidence(self):
        text = "2025년 당사의 매출은 333조 6,059억원으로 전년 동기 대비 10.9% 증가하였습니다."
        result = rc.structure_sentence(text, ["II. 사업의 내용"], "삼성전자")
        self.assertEqual(result["status"], "reported_candidate")
        self.assertEqual(result["explicit_years"], [2025])
        self.assertEqual(result["measurements"][0]["normalized_value"], "333605900000000")
        for group in ("measurements", "metrics", "period_expressions", "change_expressions"):
            for record in result[group]:
                self.assertEqual(text[record["start"]:record["end"]], record["text"])
        self.assertFalse(result["externally_verified"])
        self.assertFalse(result["model_input_approved"])

    def test_plans_industry_other_company_and_segments_are_not_company_results(self):
        cases = [
            ("2025년 당사는 매출 1조원을 달성할 계획입니다.", "company", "outlook"),
            ("2025년 당사는 매출 1조원을 달성할 방침입니다.", "company", "outlook"),
            ("2025년 자동차 시장은 매출 1조원을 기록했습니다.", "industry", "reported"),
            ("2025년 현대로템은 수주 1조원을 기록했습니다.", "related_or_segment", "reported"),
            ("2025년 Alphabet은 매출 1조원을 기록했습니다.", "related_or_segment", "reported"),
            ("2025년 당사의 DX부문 매출은 1조원입니다.", "related_or_segment", "reported"),
        ]
        for text, scope, orientation in cases:
            with self.subTest(text=text):
                result = rc.structure_sentence(text, ["매출 및 수주상황"], "삼성전자")
                self.assertEqual(result["scope"], scope)
                self.assertEqual(result["time_orientation"], orientation)
                self.assertEqual(result["status"], "needs_review")
        conditional = rc.structure_sentence("2025년 환율이 같은 경우 당사의 매출은 100억원입니다.", [], "삼성전자")
        self.assertIn("conditional_context", conditional["review_reasons"])

    def test_multiple_quarters_missing_period_and_unparsed_numbers_stay_review(self):
        for text in ("당사의 매출은 100억원입니다.",
                     "2025년 1분기와 2025년 2분기 당사의 매출은 100억원입니다.",
                     "2025년 당사의 매출은 1,23억원입니다."):
            result = rc.structure_sentence(text, [], "예시회사")
            self.assertIsNotNone(result)
            self.assertEqual(result["status"], "needs_review")

    def test_gross_net_and_multiple_amounts_do_not_become_one_confirmed_metric(self):
        text = "2025년 당사의 연구개발비는 100억원이며, 자산화된 연구개발비를 제외하고 90억원을 비용으로 처리하였습니다."
        result = rc.structure_sentence(text, [], "삼성전자")
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("multiple_amounts_for_same_metric", result["review_reasons"])
        self.assertIn("adjustment_or_netting_context", result["review_reasons"])

    def test_repeated_values_are_flagged_within_receipt_not_independent_evidence(self):
        first = rc.structure_sentence("2025년 당사의 매출은 100억원입니다.", [], "삼성전자")
        second = rc.structure_sentence("2025년 당사는 매출 100억원을 기록했습니다.", [], "삼성전자")
        rows = [{**c, "code": "005930", "receipt_no": receipt} for c, receipt in
                ((first, "20260310000001"), (second, "20260310000001"), (first, "20260311000001"))]
        rc.mark_possible_repeats(rows)
        self.assertEqual([row["possible_repeat_count"] for row in rows], [2, 2, 1])

    def test_annotation_preserves_source_ranges_skips_tables_and_validates_publication(self):
        text = "2025년 당사의 매출은 100억원입니다. 2026년에는 매출 증가를 예상합니다."
        block = {"id": "b000001", "kind": "paragraph", "text": text, "headings": ["II. 사업의 내용"], "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        event = {"receipt_no": "20260310000001", "receipt_date": "2026-03-10", "available_from": "2026-03-11T00:00:00+09:00", "period_end": "2025-12-31"}
        claims, skipped = rc.extract_claims([block, {**block, "id": "b000002", "kind": "table"}], event, "005930", "삼성전자")
        self.assertEqual(len(claims), 2)
        self.assertEqual(skipped["tables_not_interpreted"], 1)
        for claim in claims:
            self.assertEqual(text[claim["start"]:claim["end"]], claim["text"])
            self.assertEqual(claim["available_from"], "2026-03-11T00:00:00+09:00")
        with self.assertRaises(CollectionError):
            rc.extract_claims([block], {**event, "available_from": "2026-03-10T00:00:00+09:00"}, "005930", "삼성전자")


class FactsWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        research = self.root / "research" / "sample"
        research.mkdir(parents=True)
        (research / "batch.json").write_text(json.dumps({"schema_version": 1, "requested_start": "2023-10-27", "requested_end": "2026-10-27", "stocks": [{"code": "005930"}]}))
        client = Mock()
        client.companies.return_value = {"005930": {"corp_code": CORP, "corp_name": "삼성전자"}}
        client.company_identifiers.return_value = {CORP}
        client.disclosures.return_value = [filing()]
        injection = "2024년 당사의 매출은 <script>alert('not executable')</script>100억원입니다."
        client.original_document.return_value = archive(xml_document(extra=f"<P>{html.escape(injection)}</P>"))
        self.source, _ = business.run(research, self.root / "business", self.root / "business_documents", client, progress=lambda _: None)
        for name in ("forward", "text_learning"):
            (self.root / name).mkdir()
            (self.root / name / "kept.txt").write_text("original model/forecast state")

    def test_offline_full_workflow_preserves_models_sources_and_exact_nested_spans(self):
        before = {p: p.read_bytes() for folder in (self.source, self.root / "forward", self.root / "text_learning") for p in folder.rglob("*") if p.is_file()}
        with patch("dart_client.stored_key", side_effect=AssertionError("no keys")), patch("kiwoom.get_client", side_effect=AssertionError("no Kiwoom")), patch("learning.fit", side_effect=AssertionError("no fitting")), patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no network")):
            target, summary = facts.run(self.source, self.root, progress=lambda _: None)
        self.assertEqual(summary["status"], "structured")
        self.assertEqual(summary["read_documents"], 1)
        self.assertGreater(summary["counts"]["claims"], 0)
        self.assertFalse(summary["model_inputs_modified"])
        self.assertFalse(summary["training_executed"])
        claims = [json.loads(line) for line in (target / "claims.jsonl").read_text().splitlines()]
        for claim in claims:
            blocks = json.loads((target / "sources" / f"{claim['code']}_{claim['receipt_no']}.json").read_text())
            block = next(b for b in blocks if b["id"] == claim["block_id"])
            self.assertEqual(block["text"][claim["start"]:claim["end"]], claim["text"])
            for q in claim["measurements"]:
                self.assertEqual(claim["text"][q["start"]:q["end"]], q["text"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(hashlib.sha256((target / "claims.jsonl").read_bytes()).hexdigest(), summary["claims_sha256"])
        self.assertNotIn("<script>alert", (target / "report.html").read_text())
        self.assertIn("&lt;script&gt;", (target / "report.html").read_text())

    def test_changed_latest_source_is_not_substituted_and_shows_local_file_reason(self):
        source_file = next((self.source / "sources").glob("*.json"))
        source_file.unlink()
        _, summary = facts.run(self.source, self.root, progress=lambda _: None)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["stocks"][0]["selected_receipt_no"], None)
        self.assertIn("사업보고서의 본문 파일", summary["stocks"][0]["errors"][0])

    def test_report_failure_leaves_no_completed_result_and_cli_reuses_existing_data(self):
        with patch("facts.render_report", side_effect=OSError("disk")), self.assertRaises(OSError):
            facts.run(self.source, self.root, progress=lambda _: None)
        self.assertEqual(list((self.root / "facts").iterdir()), [])
        output = io.StringIO()
        with redirect_stdout(output):
            code = facts.main(["--root", str(self.root), "--no-open"])
        self.assertEqual(code, 0)
        self.assertIn("원문 구조화: 1/1건", output.getvalue())


if __name__ == "__main__":
    unittest.main()
