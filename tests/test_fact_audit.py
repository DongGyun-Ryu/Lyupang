import copy
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
import check_facts as audit
import facts
from lyupang import CollectionError
from test_business import CORP, archive, filing, xml_document


class FactAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        research = self.root / "research/sample"; research.mkdir(parents=True)
        (research / "batch.json").write_text(json.dumps({"schema_version": 1, "requested_start": "2023-10-27", "requested_end": "2026-10-27", "stocks": [{"code": "005930"}]}))
        client = Mock()
        client.companies.return_value = {"005930": {"corp_code": CORP, "corp_name": "삼성전자"}}
        client.company_identifiers.return_value = {CORP}
        client.disclosures.return_value = [filing()]
        revenue = "2024년 당사의 매출은 <script>alert('x')</script>100억원으로 전년 대비 10% 증가하였습니다."
        cash = "투자활동 현금유출은 전년 대비 44.5% 감소한 (-) 7,437억 원입니다."
        client.original_document.return_value = archive(xml_document(extra=f"<P>{html.escape(revenue)}</P><P>{cash}</P>"))
        source, _ = business.run(research, self.root / "business", self.root / "business_documents", client, progress=lambda _: None)
        self.source, _ = facts.run(source, self.root, progress=lambda _: None)
        self.data = audit.load_facts(self.source)
        self.revenue = next(c for c in self.data[1] if "100억원" in c["text"])
        self.cash = next(c for c in self.data[1] if "7,437" in c["text"])

    def case(self, claim=None):
        claim = claim or self.revenue
        doc = self.data[2][claim["code"], claim["receipt_no"]]["metadata"]
        return {"id": "fixture", "title": "합성 검증 자료", "code": claim["code"], "receipt_no": claim["receipt_no"],
            "block_id": claim["block_id"], "quote_sha256": claim["text_sha256"], "source_blocks_sha256": doc["source_blocks_sha256"],
            "quantities": [{"text": "100억원", "normalized_value": "10000000000", "unit": "KRW", "metric_equals": "revenue"}],
            "context": {"scope": "company", "time_orientation": "reported"}, "rationale": "가격 결과를 사용하지 않는 합성 사례입니다."}

    def rewrite(self, claims):
        raw = ("\n".join(json.dumps(c, ensure_ascii=False) for c in claims)+"\n").encode()
        (self.source / "claims.jsonl").write_bytes(raw)
        summary = json.loads((self.source / "summary.json").read_text())
        summary["claims_sha256"] = hashlib.sha256(raw).hexdigest()
        summary["counts"] = facts.tally(claims)
        for stock in summary["stocks"]:
            stock["counts"] = facts.tally([c for c in claims if c["code"] == stock["code"]])
        (self.source / "summary.json").write_text(json.dumps(summary))

    def test_reference_file_is_fixed_development_criteria_not_human_or_holdout_truth(self):
        reference, raw = audit.load_reference()
        self.assertEqual(len(reference["cases"]), 12)
        self.assertFalse(reference["independent_human_reviewed"])
        self.assertFalse(reference["fresh_unseen_evaluation"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), audit.REFERENCE_SHA256)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "reference.json"; p.write_bytes(raw+b" ")
            with patch.object(audit, "REFERENCE", p), self.assertRaises(CollectionError):
                audit.load_reference()

    def test_compare_numeric_and_binding_errors_separately_without_learning_approval(self):
        case = self.case()
        self.assertEqual(audit.compare_case(case, self.data[1], self.data[2])["status"], "matched")
        wrong = copy.deepcopy(self.data[1])
        c = next(c for c in wrong if c["id"] == self.revenue["id"])
        q = next(q for q in c["measurements"] if q["text"] == "100억원")
        q["metric"] = "investment"
        result = audit.compare_case(case, wrong, self.data[2])
        self.assertEqual(result["status"], "mismatch")
        self.assertTrue(result["quantity_checks"][0]["number_match"])
        self.assertFalse(result["quantity_checks"][0]["binding_match"])
        q["metric"] = "revenue"; q["normalized_value"] = "10000000001"
        self.assertFalse(audit.compare_case(case, wrong, self.data[2])["quantity_checks"][0]["number_match"])
        q["normalized_value"] = "10000000000.00"; q["unit"] = "USD"
        self.assertFalse(audit.compare_case(case, wrong, self.data[2])["quantity_checks"][0]["number_match"])

    def test_existing_negative_sign_error_is_detected_not_filled_from_program_output(self):
        case = self.case(self.cash)
        case["quantities"] = [{"text": "7,437억 원", "normalized_value": "-743700000000", "unit": "KRW"},
                              {"text": "44.5%", "normalized_value": "44.5", "unit": "percent", "metric_forbidden": ["investment"]}]
        case["context"] = {"status": "needs_review"}
        result = audit.compare_case(case, self.data[1], self.data[2])
        self.assertEqual(result["status"], "mismatch")
        self.assertFalse(result["quantity_checks"][0]["number_match"])
        self.assertFalse(result["quantity_checks"][1]["binding_match"])

    def test_absence_changed_source_and_duplicate_quotes_are_never_passes(self):
        case = self.case()
        self.assertEqual(audit.compare_case(case, [], self.data[2])["status"], "missing")
        self.assertEqual(audit.compare_case(case, self.data[1], {})["status"], "missing")
        altered = copy.deepcopy(self.data[2]); altered[case["code"], case["receipt_no"]]["metadata"]["source_blocks_sha256"] = "0"*64
        self.assertEqual(audit.compare_case(case, self.data[1], altered)["status"], "source_mismatch")
        self.assertEqual(audit.compare_case(case, self.data[1]+[self.revenue], self.data[2])["status"], "ambiguous")
        missing_value = copy.deepcopy(case); missing_value["quantities"][0]["text"] = "존재하지 않는 수치"
        result = audit.compare_case(missing_value, self.data[1], self.data[2])
        self.assertEqual(result["status"], "mismatch")

    def test_unparsed_range_matches_none_only_and_context_mismatch_is_failure(self):
        case = self.case(); case["quantities"] = [{"text": "100억원", "normalized_value": None, "unit": "KRW"}]
        self.assertEqual(audit.compare_case(case, self.data[1], self.data[2])["status"], "mismatch")
        changed = copy.deepcopy(self.data[1])
        c = next(c for c in changed if c["id"] == self.revenue["id"])
        next(q for q in c["measurements"] if q["text"] == "100억원")["normalized_value"] = None
        self.assertEqual(audit.compare_case(case, changed, self.data[2])["status"], "matched")
        case["context"] = {"scope": "related_or_segment"}
        self.assertEqual(audit.compare_case(case, changed, self.data[2])["status"], "mismatch")

    def test_corrupt_claim_file_and_source_copy_are_rejected(self):
        path = self.source / "claims.jsonl"; original = path.read_bytes(); path.write_bytes(original+b" ")
        with self.assertRaises(CollectionError): audit.load_facts(self.source)
        path.write_bytes(original)
        copied = next((self.source / "sources").glob("*.json")); copied.write_bytes(copied.read_bytes()+b" ")
        with self.assertRaises(CollectionError): audit.load_facts(self.source)

    def test_changed_offsets_approval_flags_and_counts_rejected_after_rehash(self):
        for alteration in ("offset", "flag", "claim_hash", "count"):
            with self.subTest(alteration=alteration):
                claims = copy.deepcopy(self.data[1])
                if alteration == "offset":
                    quantity = next(c for c in claims if c["id"] == self.revenue["id"])["measurements"][0]
                    quantity.update(start=0, end=1, text="wrong")
                if alteration == "flag": claims[0]["model_input_approved"] = True
                if alteration == "claim_hash": claims[0]["text_sha256"] = "0"*64
                self.rewrite(claims)
                if alteration == "count":
                    s = json.loads((self.source / "summary.json").read_text()); s["counts"]["claims"] += 1
                    (self.source / "summary.json").write_text(json.dumps(s))
                with self.assertRaises(CollectionError): audit.load_facts(self.source)

    def test_publication_and_headings_preserved(self):
        for alteration in ("publication", "heading"):
            claims = copy.deepcopy(self.data[1])
            if alteration == "publication": claims[0]["available_from"] = "2025-03-11T00:00:00+09:00"
            else: claims[0]["headings"] = ["unverified scope"]
            self.rewrite(claims)
            with self.assertRaises(CollectionError): audit.load_facts(self.source)

    def test_full_offline_run_preserves_files_and_never_approves_even_when_all_match(self):
        for folder in ("forward", "text_learning", "learning"):
            p = self.root / folder / "existing-model.txt"; p.parent.mkdir(exist_ok=True); p.write_text("unchanged")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        reference = {"reference_version": "test_only", "cases": [self.case()]}
        raw = json.dumps(reference).encode()
        with patch.object(audit, "load_reference", return_value=(reference, raw)), patch("kiwoom.get_client", side_effect=AssertionError("no Kiwoom")), patch("learning.fit", side_effect=AssertionError("no fitting")), patch("dart_client.stored_key", side_effect=AssertionError("no keys")), patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no network")):
            target, summary = audit.run(self.source, self.root, progress=lambda _: None)
        self.assertEqual(summary["status"], "development_checks_matched")
        self.assertFalse(summary["learning_ready"])
        self.assertFalse(summary["independent_accuracy_verified"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        report = (target / "report.html").read_text()
        self.assertNotIn("<script>alert", report)
        self.assertIn("&lt;script&gt;", report)
        self.assertEqual((target / "input_claims.jsonl").read_bytes(), (self.source / "claims.jsonl").read_bytes())

    def test_real_reference_missing_from_fixture_is_not_scored_as_correct(self):
        _, summary = audit.run(self.source, self.root, progress=lambda _: None)
        self.assertEqual(summary["case_counts"]["matched"], 0)
        self.assertEqual(summary["case_counts"]["missing"], 12)
        self.assertEqual(summary["status"], "review_required")

    def test_latest_corrupt_input_is_not_replaced_and_cli_preserves_exit_status(self):
        newer = self.root / "facts/20991231_235959_12345678"; newer.mkdir()
        (newer / "summary.json").write_text("not json")
        self.assertEqual(audit.latest_facts(self.root / "facts"), newer)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            self.assertEqual(audit.main(["--root", str(self.root), "--no-open"]), 1)
            self.assertEqual(audit.main(["--input", str(self.source), "--root", str(self.root), "--no-open"]), 2)

    def test_failed_report_only_removes_its_pending_output(self):
        parent = self.root / "fact_checks"; parent.mkdir()
        keep = parent / "existing.txt"; keep.write_text("keep")
        with patch.object(audit, "render_report", side_effect=OSError("disk failure")), self.assertRaises(OSError):
            audit.run(self.source, self.root, progress=lambda _: None)
        self.assertEqual(list(parent.iterdir()), [keep])


if __name__ == "__main__":
    unittest.main()
