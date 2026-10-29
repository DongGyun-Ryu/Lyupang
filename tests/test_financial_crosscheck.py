import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import business
import cross_check as cc
import facts
import financial_crosscheck as fc
import report_claims
from financial_history import parse_original
from lyupang import CollectionError
from test_business import CORP, archive, filing, xml_document
from test_financial_history import original


def finance_payload(*, revenue=200, scope="CFS", decimals="INF", extra=""):
    raw = original(year=2023, revenue=revenue, scope=scope, extra=extra)
    with zipfile.ZipFile(io.BytesIO(raw)) as z: xml = z.read("report.xbrl").decode()
    if decimals is not None: xml = xml.replace('unitRef="currency">', f'unitRef="currency" decimals="{decimals}">')
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z: z.writestr("report.xbrl", xml)
    return out.getvalue()


def financial_doc(raw=None):
    raw = raw or finance_payload()
    audited = parse_original(raw, {CORP}, date(2023, 12, 31))
    records, identity = fc.parse_facts(raw, date(2023, 12, 31), audited, CORP, "005930")
    return {"status": "read", "facts": records, **identity}


def claim(text="2023년 당사의 연결 기준 매출은 200원입니다."):
    parsed = report_claims.structure_sentence(text, ["II. 사업의 내용"], "삼성전자")
    return {"id": "c1", "code": "005930", "receipt_no": "20240312000001", "block_id": "b000001",
            "report_period_end": "2023-12-31", "text": text, **parsed}


class FinancialComparisonTests(unittest.TestCase):
    def test_exact_standard_value_preserves_fact_identity_scope_period_and_no_approval(self):
        c = claim(); doc = financial_doc()
        result = fc.compare_amount(c, c["measurements"][0], doc)
        self.assertEqual(result["status"], "exact_match")
        f = result["financial_facts"][0]
        self.assertEqual((f["scope"], f["period_start"], f["period_end"], f["context_ref"]), ("CFS", "2023-01-01", "2023-12-31", "duration"))
        self.assertIn("Revenue", f["concept"])
        self.assertFalse(result["model_input_approved"])
        self.assertFalse(doc["new_external_identity_check"])

    def test_rounding_overlap_is_distinct_from_exact_and_mismatch(self):
        c = claim("2023년 당사의 연결 기준 매출은 2백원입니다."); q = c["measurements"][0]
        self.assertEqual(fc.text_quantum(q), 100)
        self.assertEqual(fc.compare_amount(c, q, financial_doc(finance_payload(revenue=249)))["status"], "rounding_overlap")
        self.assertEqual(fc.compare_amount(c, q, financial_doc(finance_payload(revenue=251)))["status"], "value_mismatch")
        self.assertEqual(fc.compare_amount(c, q, financial_doc(finance_payload(revenue=249, decimals=None)))["reasons"], ["rounding_unknown"])
        large = report_claims.measurements("매출 333조 6,059억원")[0]
        self.assertEqual(fc.text_quantum(large), 100000000)

    def test_scope_missing_separate_and_undimensioned_are_not_silently_consolidated(self):
        missing = claim("2023년 당사의 매출은 200원입니다.")
        self.assertIn("basis_unclear", fc.compare_amount(missing, missing["measurements"][0], financial_doc())["reasons"])
        c = claim()
        for scope in (None, "OFS"):
            result = fc.compare_amount(c, c["measurements"][0], financial_doc(finance_payload(scope=scope)))
            self.assertEqual(result["status"], "not_comparable")
            self.assertIn("fact_missing", result["reasons"])
        separate = claim("2023년 당사의 별도 기준 매출은 200원입니다.")
        self.assertEqual(fc.compare_amount(separate, separate["measurements"][0], financial_doc(finance_payload(scope="OFS")))["status"], "exact_match")

    def test_period_plan_segment_percent_and_loss_roles_are_not_amount_matches(self):
        for text, reason in (("2022년 당사의 연결 기준 매출은 200원입니다.", "period_mismatch"),
            ("2023년 당사의 연결 기준 매출은 200원으로 예상합니다.", "statement_requires_review"),
            ("2023년 당사의 DX부문 연결 기준 매출은 200원입니다.", "statement_requires_review"),
            ("2023년 당사의 연결 기준 매출은 전년 대비 10% 증가했습니다.", "unsupported_amount"),
            ("2023년 당사의 연결 기준 순손실은 200원입니다.", "loss_sign_unclear")):
            with self.subTest(text=text):
                c = claim(text)
                self.assertIn(reason, fc.compare_amount(c, c["measurements"][0], financial_doc())["reasons"])

    def test_conflicting_facts_are_not_selected_to_fit_the_text(self):
        c = claim(); doc = financial_doc(); other = copy.deepcopy(doc["facts"][0]); other["value"] = "999"
        doc["facts"].append(other)
        self.assertEqual(fc.compare_amount(c, c["measurements"][0], doc)["reasons"], ["fact_conflict"])

    def test_change_amount_and_interim_period_do_not_match_annual_revenue(self):
        for text, reason in (("2023년 당사의 연결 기준 매출은 전년 대비 200원 증가하였습니다.", "amount_change_context"),
                             ("2023년 당사의 연결 기준 매출 증가액은 200원입니다.", "amount_change_context"),
                             ("2023년 누적 3분기 당사의 연결 기준 매출은 200원입니다.", "period_unclear")):
            with self.subTest(text=text):
                c = claim(text); result = fc.compare_amount(c, c["measurements"][0], financial_doc())
                self.assertEqual(result["status"], "not_comparable")
                self.assertIn(reason, result["reasons"])

    def test_standard_operating_profit_is_read_but_foreign_namespace_is_not(self):
        extra = '<ifrs:ProfitLossFromOperatingActivities contextRef="duration" unitRef="currency">50</ifrs:ProfitLossFromOperatingActivities>'
        c = claim("2023년 당사의 연결 기준 영업이익은 50원입니다.")
        doc = financial_doc(finance_payload(extra=extra))
        self.assertEqual(fc.compare_amount(c, c["measurements"][0], doc)["status"], "exact_match")
        fake = extra.replace('ifrs:ProfitLossFromOperatingActivities', 'fake:ProfitLossFromOperatingActivities').replace(' contextRef', ' xmlns:fake="https://ifrs.org.evil.test/x" contextRef', 1)
        doc = financial_doc(finance_payload(extra=fake))
        self.assertEqual(fc.compare_amount(c, c["measurements"][0], doc)["reasons"], ["fact_missing"])

    def test_partial_sales_receivables_and_attributable_profit_are_not_total_values(self):
        for noun in ("국내 매출", "매출채권", "지배주주 순이익", "세전 순이익", "공장 매출"):
            c = claim(f"2023년 당사의 연결 기준 {noun}은 200원입니다.")
            result = fc.compare_amount(c, c["measurements"][0], financial_doc())
            self.assertEqual(result["status"], "not_comparable")
            self.assertIn("submetric_context", result["reasons"])

    def test_original_audit_reproduction_and_single_issuer_are_required(self):
        raw = finance_payload(); metrics = parse_original(raw, {CORP}, date(2023, 12, 31)); metrics["revenue_krw"] = "999"
        with self.assertRaises(CollectionError): fc.parse_facts(raw, date(2023, 12, 31), metrics, CORP, "005930")
        extra = '<x:context id="other"><x:entity><x:identifier scheme="test">99999999</x:identifier></x:entity><x:period><x:instant>2023-12-31</x:instant></x:period></x:context>'
        raw = finance_payload(extra=extra); metrics = parse_original(raw, {CORP}, date(2023, 12, 31))
        with self.assertRaises(CollectionError): fc.parse_facts(raw, date(2023, 12, 31), metrics, CORP, "005930")

    def test_duplicate_context_and_unsupported_currency_rejected(self):
        raw = finance_payload(extra='<x:context id="duration"/>')
        with self.assertRaises(CollectionError): fc.parse_facts(raw, date(2023, 12, 31), {}, CORP, "005930")
        raw = original(currency="USD", scope="CFS")
        with self.assertRaises(CollectionError): fc.parse_facts(raw, date(2023, 12, 31), {}, CORP, "005930")


class CrosscheckWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        research = self.root / "research/sample"; (research / "dart").mkdir(parents=True)
        (research / "batch.json").write_text(json.dumps({"schema_version": 1, "requested_start": "2023-10-27", "requested_end": "2026-10-27", "stocks": [{"code": "005930"}]}))
        (research / "dart/005930.json").write_text(json.dumps({"company": {"corp_code": CORP}}))
        client = Mock(); client.companies.return_value = {"005930": {"corp_code": CORP, "corp_name": "삼성전자"}}
        client.company_identifiers.return_value = {CORP}
        client.disclosures.return_value = [filing(receipt="20240312000001", day="2024-03-12", title="사업보고서 (2023.12)")]
        text = "2023년 당사의 연결 기준 매출은 200원입니다."
        client.original_document.return_value = archive(xml_document(extra=f"<P>{text}</P>"), receipt="20240312000001")
        source, _ = business.run(research, self.root / "business", self.root / "business_documents", client, progress=lambda _: None)
        self.source, _ = facts.run(source, self.root, progress=lambda _: None)
        self.learning = self.root / "learning/20261028_114327_12345678"; self.learning.mkdir(parents=True)
        (self.learning / "summary.json").write_text(json.dumps({"schema_version": 1, "source_research": str(research)}))
        self.raw = finance_payload(); self.digest = hashlib.sha256(self.raw).hexdigest()
        event = {"receipt_no": "20240312000001", "receipt_date": "2024-03-12", "report_name": "사업보고서 (2023.12)", "period_end": "2023-12-31", "status": "verified", "sha256": self.digest, "metrics": parse_original(self.raw, {CORP}, date(2023, 12, 31))}
        self.audit = {"005930": [event]}; (self.learning / "financial_audit.json").write_text(json.dumps(self.audit))
        self.cache = self.root / "financial_documents/005930_20240312000001_12345678"; self.cache.mkdir(parents=True)
        (self.cache / "original.zip").write_bytes(self.raw)
        self.meta = {"provider": "OpenDART", "endpoint": "fnlttXbrl.xml", "code": "005930", "receipt_no": "20240312000001", "sha256": self.digest}
        (self.cache / "source.json").write_text(json.dumps(self.meta))

    def test_full_offline_comparison_preserves_inputs_and_source_provenance(self):
        for folder in ("forward", "text_learning"):
            p = self.root / folder / "model.txt"; p.parent.mkdir(); p.write_text("preserve")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch("kiwoom.get_client", side_effect=AssertionError("no Kiwoom")), patch("dart_client.stored_key", side_effect=AssertionError("no key")), patch("learning.fit", side_effect=AssertionError("no fitting")), patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no network")):
            target, summary = cc.run(self.source, self.root, self.learning, progress=lambda _: None)
        self.assertEqual(summary["financial_read"], 1)
        self.assertEqual(summary["comparison_counts"]["exact_match"], 1)
        self.assertFalse(summary["learning_ready"])
        self.assertFalse(summary["independent_accuracy_verified"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual((target / "financial_sources/005930_20240312000001/original.zip").read_bytes(), self.raw)
        self.assertEqual((target / "input_claims.jsonl").read_bytes(), (self.source / "claims.jsonl").read_bytes())
        rows = [json.loads(s) for s in (target / "comparisons.jsonl").read_text().splitlines()]
        self.assertEqual(sum(r["status"] == "exact_match" for r in rows), 1)

    def test_context_pipeline_keeps_baseline_and_cites_same_report_without_rewriting_claims(self):
        research = self.root / "research/sample"
        client = Mock(); client.companies.return_value = {"005930": {"corp_code": CORP, "corp_name": "삼성전자"}}
        client.company_identifiers.return_value = {CORP}
        client.disclosures.return_value = [filing(receipt="20240312000001", day="2024-03-12", title="사업보고서 (2023.12)")]
        extra = '<P>아래 실적은 2023년 연결기준으로 작성하였습니다.</P><P>2023년 당사의 매출은 200원입니다.</P>'
        client.original_document.return_value = archive(xml_document(extra=extra), receipt="20240312000001")
        source, _ = business.run(research, self.root / "business", self.root / "business_context_documents", client, progress=lambda _: None)
        input_facts, _ = facts.run(source, self.root, progress=lambda _: None)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        target, summary = cc.run(input_facts, self.root, self.learning, progress=lambda _: None)
        self.assertEqual(summary["baseline_compared"], 0)
        self.assertEqual(summary["context_compared"], 1)
        self.assertEqual(summary["context_evidence_compared"], 1)
        baseline = [json.loads(x) for x in (target / "baseline_comparisons.jsonl").read_text().splitlines()]
        rows = [json.loads(x) for x in (target / "comparisons.jsonl").read_text().splitlines()]
        self.assertEqual(len(baseline), len(rows))
        compared = next(r for r in rows if r["status"] == "exact_match")
        self.assertTrue(compared["context_resolution"]["evidence"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertIn("기간·기준 근거 원문", (target / "report.html").read_text())
        self.assertFalse(summary["learning_ready"])

    def test_wrong_receipt_or_corrupt_cache_does_not_use_other_receipts(self):
        (self.cache / "original.zip").write_bytes(self.raw+b"changed")
        other = self.cache.parent / "005930_20240313000002_87654321"; other.mkdir()
        (other / "original.zip").write_bytes(self.raw)
        (other / "source.json").write_text(json.dumps({**self.meta, "receipt_no": "20240313000002"}))
        target, summary = cc.run(self.source, self.root, self.learning, progress=lambda _: None)
        self.assertEqual(summary["financial_read"], 0)
        self.assertEqual(summary["comparison_counts"]["exact_match"], 0)
        self.assertIn("같은 해시", (target / "report.html").read_text())
        self.assertEqual(summary["status"], "review_required")

    def test_audit_receipt_period_and_status_must_match(self):
        for key, value in (("receipt_no", "20240313000002"), ("period_end", "2022-12-31"), ("status", "unavailable")):
            audit = copy.deepcopy(self.audit); audit["005930"][0][key] = value
            (self.learning / "financial_audit.json").write_text(json.dumps(audit))
            _, summary = cc.run(self.source, self.root, self.learning, progress=lambda _: None)
            self.assertEqual(summary["financial_read"], 0)

    def test_missing_audit_produces_waiting_report_without_authentication_request(self):
        (self.learning / "financial_audit.json").unlink()
        target, summary = cc.run(self.source, self.root, self.learning, progress=lambda _: None)
        self.assertEqual(summary["financial_read"], 0)
        self.assertTrue(summary["setup_errors"])
        self.assertTrue((target / "report.html").exists())
        self.assertEqual(summary["status"], "review_required")

    def test_pinned_learning_source_used_without_model_read_and_bad_pin_not_replaced(self):
        p = self.root / "forward/experiment/experiment.json"; p.parent.mkdir(parents=True)
        payload = {"policy": "forward_journal_v1", "source_learning": str(self.learning)}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        p.write_text(json.dumps({"payload": payload, "sha256": hashlib.sha256(raw).hexdigest()}))
        self.assertEqual(cc.learning_source(self.root), self.learning)
        p.write_text(json.dumps({"payload": payload, "sha256": "0"*64}))
        with self.assertRaises(CollectionError): cc.learning_source(self.root)

    def test_cli_success_and_waiting_exit_statuses(self):
        args = ["--root", str(self.root), "--input", str(self.source), "--learning", str(self.learning), "--no-open"]
        with redirect_stdout(io.StringIO()): self.assertEqual(cc.main(args), 0)
        (self.cache / "original.zip").unlink()
        with redirect_stdout(io.StringIO()): self.assertEqual(cc.main(args), 2)

    def test_failed_report_only_cleans_new_pending_output(self):
        parent = self.root / "financial_crosschecks"; parent.mkdir(); keep = parent / "keep.txt"; keep.write_text("keep")
        with patch.object(cc, "render_report", side_effect=OSError("disk full")), self.assertRaises(OSError):
            cc.run(self.source, self.root, self.learning, progress=lambda _: None)
        self.assertEqual(list(parent.iterdir()), [keep])


if __name__ == "__main__": unittest.main()
