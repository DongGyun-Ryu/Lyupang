import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import business
import cross_check
import evidence_table as table
import facts
from financial_crosscheck import compare_amount
from financial_history import parse_original
from lyupang import CollectionError
from test_business import CORP, archive, filing, xml_document
from test_financial_crosscheck import claim, finance_payload, financial_doc


def example():
    c = claim(); q = c["measurements"][0]
    row = compare_amount(c, q, financial_doc())
    row["context_resolution"] = {"year": 2023, "scope": "CFS", "year_method": "sentence_explicit", "basis_method": "sentence_explicit", "evidence": [], "conflicts": []}
    coverage = [{"code": c["code"], "company_name": "삼성전자", "receipt_no": c["receipt_no"], "receipt_date": "2024-03-12", "available_from": "2024-03-13T00:00:00+09:00", "period_end": "2023-12-31", "financial_status": "read"}]
    return row, coverage


class EvidenceRecordTests(unittest.TestCase):
    def test_table_keeps_exact_reference_and_original_text_values_separate(self):
        row, coverage = example()
        row["status"] = "rounding_overlap"; row["quantity"]["normalized_value"] = "199"
        before = copy.deepcopy(row)
        records = table.build_records([row], coverage)
        self.assertEqual(records[0]["reference_value_krw"], "200")
        self.assertEqual(records[0]["evidence"][0]["quantity"]["normalized_value"], "199")
        self.assertFalse(records[0]["model_input_approved"])
        self.assertEqual(row, before)

    def test_duplicate_statements_merge_evidence_not_independent_records(self):
        row, coverage = example(); duplicate = copy.deepcopy(row); duplicate["claim_id"] = "another"
        records = table.build_records([row, duplicate], coverage)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["evidence"]), 2)

    def test_separate_scopes_and_receipts_are_not_merged(self):
        row, coverage = example(); separate = copy.deepcopy(row)
        separate["context_resolution"]["scope"] = "OFS"
        for f in separate["financial_facts"]: f["scope"] = "OFS"
        later = copy.deepcopy(row); later["receipt_no"] = "20240314000002"
        later_doc = {**coverage[0], "receipt_no": later["receipt_no"], "receipt_date": "2024-03-14", "available_from": "2024-03-15T00:00:00+09:00"}
        records = table.build_records([row, separate, later], coverage+[later_doc])
        self.assertEqual(len(records), 3)

    def test_mismatches_and_unresolved_values_never_enter_table(self):
        row, coverage = example()
        for status in ("value_mismatch", "not_comparable"):
            row["status"] = status
            self.assertEqual(table.build_records([row], coverage), [])

    def test_conflicting_reference_values_do_not_choose_one(self):
        row, coverage = example(); duplicate = copy.deepcopy(row); duplicate["financial_facts"][0]["value"] = "201"
        with self.assertRaises(CollectionError): table.build_records([row, duplicate], coverage)

    def test_publication_day_is_not_eligible_and_missing_is_not_zero(self):
        row, coverage = example(); records = table.build_records([row], coverage)
        before = table.snapshot_as_of(records, coverage, {"005930"}, "2024-03-12")
        after = table.snapshot_as_of(records, coverage, {"005930"}, "2024-03-13")
        self.assertIsNone(before[0]["selected_receipt_no"])
        self.assertEqual(after[0]["values"]["revenue"], "200")
        self.assertIsNone(after[0]["values"]["net_income"])
        self.assertIn("net_income", after[0]["missing_metrics"])

    def test_latest_unmatched_revision_blocks_old_values(self):
        row, coverage = example(); records = table.build_records([row], coverage)
        later = {**coverage[0], "receipt_no": "20240601000001", "receipt_date": "2024-06-01", "available_from": "2024-06-02T00:00:00+09:00", "financial_status": "unavailable"}
        snapshots = table.snapshot_as_of(records, coverage+[later], {"005930"}, "2024-06-02")
        self.assertEqual(snapshots[0]["selected_receipt_no"], later["receipt_no"])
        self.assertTrue(all(v is None for v in snapshots[0]["values"].values()))

    def test_partial_latest_receipt_does_not_mix_old_metric_into_new_values(self):
        row, coverage = example(); records = table.build_records([row], coverage)
        old_income = {**records[0], "id": "old-income", "metric": "net_income", "reference_value_krw": "20"}
        later = {**coverage[0], "receipt_no": "20240601000001", "receipt_date": "2024-06-01", "available_from": "2024-06-02T00:00:00+09:00"}
        later_revenue = {**records[0], "id": "new-revenue", "receipt_no": later["receipt_no"]}
        snapshot = table.snapshot_as_of(records+[old_income, later_revenue], coverage+[later], {"005930"}, "2024-06-02")
        self.assertEqual(snapshot[0]["values"]["revenue"], "200")
        self.assertIsNone(snapshot[0]["values"]["net_income"])

    def test_zero_is_preserved_and_untrusted_company_name_is_escaped_in_report(self):
        row, coverage = example(); records = table.build_records([row], coverage)
        records[0]["reference_value_krw"] = "0"
        records[0]["company_name"] = "<script>alert('x')</script>"
        snapshot = table.snapshot_as_of(records, coverage, {"005930"}, "2024-03-13")
        self.assertEqual(snapshot[0]["values"]["revenue"], "0")
        self.assertEqual(table.money("0"), "0 원")
        summary = {"source_rows": 1, "included_evidence": 1, "records": 1, "companies_with_records": 1, "source_companies": 1, "receipts_with_records": 1, "as_of": "2024-03-13", "reference_date": "2024-03-13"}
        report = table.render_report(summary, records, snapshot)
        self.assertIn("&lt;script&gt;", report)
        self.assertNotIn("<script>", report)

    def test_later_correction_to_old_year_does_not_replace_new_fiscal_year(self):
        row, coverage = example(); records = table.build_records([row], coverage)
        older = {**coverage[0], "receipt_no": "20240601000001", "receipt_date": "2024-06-01", "available_from": "2024-06-02T00:00:00+09:00", "period_end": "2022-12-31"}
        snapshots = table.snapshot_as_of(records, coverage+[older], {"005930"}, "2024-06-02")
        self.assertEqual(snapshots[0]["selected_receipt_no"], row["receipt_no"])

    def test_same_day_receipt_ties_and_incomplete_coverage_are_held(self):
        row, coverage = example(); records = table.build_records([row], coverage)
        tied = {**coverage[0], "receipt_no": "20240312000002"}
        snapshot = table.snapshot_as_of(records, coverage+[tied], {"005930"}, "2024-03-13")
        self.assertIsNone(snapshot[0]["selected_receipt_no"])
        self.assertIn("여러 제출본", snapshot[0]["state"])
        snapshot = table.snapshot_as_of(records, coverage, {"005930"}, "2024-03-13", completeness={"005930": False})
        self.assertIsNone(snapshot[0]["selected_receipt_no"])
        self.assertIn("미완료", snapshot[0]["state"])

    def test_company_without_records_is_still_shown_for_both_scopes(self):
        row, coverage = example(); records = table.build_records([row], coverage)
        snapshots = table.snapshot_as_of(records, coverage, {"005930", "005380"}, "2024-03-13")
        missing = [r for r in snapshots if r["code"] == "005380"]
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(all(v is None for v in r["values"].values()) for r in missing))


class EvidenceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); research = self.root / "research/sample"; (research / "dart").mkdir(parents=True)
        (research / "batch.json").write_text(json.dumps({"schema_version": 1, "requested_start": "2023-10-27", "requested_end": "2026-10-27", "stocks": [{"code": "005930"}]}))
        (research / "dart/005930.json").write_text(json.dumps({"company": {"corp_code": CORP}}))
        client = Mock(); client.companies.return_value = {"005930": {"corp_code": CORP, "corp_name": "삼성전자"}}
        client.company_identifiers.return_value = {CORP}; client.disclosures.return_value = [filing(receipt="20240312000001", day="2024-03-12", title="사업보고서 (2023.12)")]
        client.original_document.return_value = archive(xml_document(extra='<P>2023년 당사의 연결 기준 매출은 200원입니다.</P>'), receipt="20240312000001")
        source, _ = business.run(research, self.root / "business", self.root / "business_documents", client, progress=lambda _: None)
        self.facts, _ = facts.run(source, self.root, progress=lambda _: None)
        learning = self.root / "learning/20261028_114327_12345678"; learning.mkdir(parents=True)
        (learning / "summary.json").write_text(json.dumps({"schema_version": 1, "source_research": str(research)}))
        raw = finance_payload(); digest = hashlib.sha256(raw).hexdigest()
        event = {"receipt_no": "20240312000001", "receipt_date": "2024-03-12", "period_end": "2023-12-31", "report_name": "사업보고서 (2023.12)", "status": "verified", "sha256": digest, "metrics": parse_original(raw, {CORP}, date(2023, 12, 31))}
        (learning / "financial_audit.json").write_text(json.dumps({"005930": [event]}))
        cache = self.root / "financial_documents/005930_20240312000001_12345678"; cache.mkdir(parents=True)
        (cache / "original.zip").write_bytes(raw); (cache / "source.json").write_text(json.dumps({"provider": "OpenDART", "endpoint": "fnlttXbrl.xml", "code": "005930", "receipt_no": "20240312000001", "sha256": digest}))
        self.source, _ = cross_check.run(self.facts, self.root, learning, progress=lambda _: None)

    def test_full_offline_workflow_keeps_sources_models_and_both_value_types(self):
        for name in ("forward", "text_learning"):
            p = self.root / name / "preserve.txt"; p.parent.mkdir(); p.write_text("preserve")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch("kiwoom.get_client", side_effect=AssertionError("no Kiwoom")), patch("dart_client.stored_key", side_effect=AssertionError("no key")), patch("learning.fit", side_effect=AssertionError("no fitting")), patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no network")):
            target, summary = table.run(self.source, self.root, progress=lambda _: None)
        self.assertEqual(summary["records"], 1)
        self.assertEqual(summary["receipts_with_records"], 1)
        self.assertFalse(summary["learning_ready"])
        self.assertFalse(summary["historical_point_in_time_certified"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertTrue((target / "evidence_table.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertEqual((target / "source_comparisons.jsonl").read_bytes(), (self.source / "comparisons.jsonl").read_bytes())

    def test_altered_comparison_is_detected_without_relying_on_a_parent_row_hash(self):
        path = self.source / "comparisons.jsonl"; rows = [json.loads(s) for s in path.read_text().splitlines()]
        next(r for r in rows if r["status"] == "exact_match")["quantity"]["normalized_value"] = "201"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        with self.assertRaises(CollectionError): table.verify_source(self.source)

    def test_changed_financial_fact_or_zip_is_rejected(self):
        path = self.source / "documents.json"; saved = path.read_bytes(); docs = json.loads(saved)
        docs[0]["facts"][0]["value"] = "201"; path.write_text(json.dumps(docs))
        with self.assertRaises(CollectionError): table.verify_source(self.source)
        path.write_bytes(saved)
        original = next((self.source / "financial_sources").glob('*/original.zip')); original.write_bytes(original.read_bytes()+b"changed")
        with self.assertRaises(CollectionError): table.verify_source(self.source)

    def test_parent_approval_flags_or_input_changes_are_not_trusted(self):
        p = self.source / "summary.json"; saved = p.read_bytes(); summary = json.loads(saved); summary["learning_ready"] = True; p.write_text(json.dumps(summary))
        with self.assertRaises(CollectionError): table.verify_source(self.source)
        p.write_bytes(saved)
        claims = self.source / "input_claims.jsonl"; claims.write_bytes(claims.read_bytes()+b" ")
        with self.assertRaises(CollectionError): table.verify_source(self.source)

    def test_latest_corrupt_parent_is_not_silently_replaced(self):
        newer = self.root / "financial_crosschecks/20991231_235959_abcdef12"; newer.mkdir()
        (newer / "summary.json").write_text("invalid json")
        self.assertEqual(table.latest_crosscheck(newer.parent), newer)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            self.assertEqual(table.main(["--root", str(self.root), "--no-open"]), 1)

    def test_cli_and_atomic_failed_output_cleanup(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(table.main(["--root", str(self.root), "--input", str(self.source), "--as-of", "2024-03-12", "--no-open"]), 0)
        parent = self.root / "evidence_tables"; before = set(parent.iterdir())
        with patch.object(table, "render_report", side_effect=OSError("disk failure")), self.assertRaises(OSError):
            table.run(self.source, self.root, progress=lambda _: None)
        self.assertEqual(set(parent.iterdir()), before)


if __name__ == "__main__": unittest.main()
