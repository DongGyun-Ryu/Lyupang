import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import cross_check
import evidence_table
import finance_learning as fl
import learn_finance as cli
from financial_history import parse_original
from learning import train_models
from lyupang import CollectionError
import test_evidence_table as evidence_fixtures
from test_evidence_table import example
from test_financial_crosscheck import CORP, finance_payload, financial_doc
from test_learning import dataset
from test_review import save_source

OP = '<ifrs:ProfitLossFromOperatingActivities contextRef="duration" unitRef="currency" decimals="INF">40</ifrs:ProfitLossFromOperatingActivities><ifrs:ProfitLossFromOperatingActivities contextRef="previous" unitRef="currency" decimals="INF">10</ifrs:ProfitLossFromOperatingActivities>'


def fixture():
    row, coverage = example()
    doc = financial_doc(finance_payload(extra=OP)); doc["source_sha256"] = "a"*64
    finance = {("005930", row["receipt_no"]): doc}
    narrative = evidence_table.build_records([row], coverage)
    records = fl.financial_records(finance, coverage, narrative, [row])
    return records, coverage, finance, narrative, row


def features_for_fixture(count=500):
    features, outcomes = dataset(count)
    for f in features:
        f["as_of_date"] = (date.fromisoformat(f["as_of_date"])+timedelta(days=100)).isoformat()
        f["available_from"] = (date.fromisoformat(f["as_of_date"])+timedelta(days=1)).isoformat()+"T00:00:00+09:00"
        f.update(financial_receipt_no="20240312000001", financial_receipt_date="2024-03-12",
            financial_available_from="2024-03-13T00:00:00+09:00", financial_scope="CFS", financial_period_end="2023-12-31")
    for o in outcomes:
        for k in ("as_of_date", "entry_date", "exit_date"):
            o[k] = (date.fromisoformat(o[k])+timedelta(days=100)).isoformat()
    return features, outcomes


class FinancialInputTests(unittest.TestCase):
    def test_direct_income_is_distinct_from_narrative_validation_and_duplicates(self):
        records, coverage, finance, narrative, row = fixture()
        income = next(r for r in records if r["metric"] == "net_income" and r["scope"] == "CFS" and r["period_end"] == "2023-12-31")
        self.assertEqual(income["value_krw"], "20")
        self.assertIsNone(income["narrative_record_id"])
        self.assertFalse(income["narrative_verified"])
        self.assertFalse(income["production_input_approved"])
        self.assertEqual(len(narrative), 1)
        for doc in finance.values(): doc["facts"] *= 2
        repeated = fl.financial_records(finance, coverage, narrative, [row])
        self.assertEqual(len(records), len(repeated))
        self.assertEqual(sum(r["status"] == "available" for r in records), sum(r["status"] == "available" for r in repeated))

    def test_conflicts_unknown_precision_and_statement_mismatches_are_held(self):
        _, coverage, finance, narrative, row = fixture()
        facts = finance["005930", row["receipt_no"]]["facts"]
        other = {**facts[0], "value": "999"}; facts.append(other)
        records = fl.financial_records(finance, coverage, narrative, [row])
        self.assertEqual(records[0]["status"], "fact_conflict")
        facts.pop(); facts[0]["quantum"] = None
        self.assertEqual(fl.financial_records(finance, coverage, narrative, [row])[0]["status"], "precision_unresolved")
        facts[0]["quantum"] = "0"; row["status"] = "value_mismatch"
        self.assertEqual(fl.financial_records(finance, coverage, narrative, [row])[0]["status"], "statement_value_mismatch")

    def test_scope_currency_and_period_never_mix_and_zero_differs_from_missing(self):
        records, coverage, finance, narrative, row = fixture()
        facts = finance["005930", row["receipt_no"]]["facts"]
        income = next(f for f in facts if f["metric"] == "net_income")
        income["value"] = "0"
        facts.append({**facts[0], "value": "999", "currency": "USD"})
        records = fl.financial_records(finance, coverage, narrative, [row])
        self.assertEqual(records[0]["value_krw"], "200")
        self.assertEqual(next(r for r in records if r["metric"] == "net_income" and r["scope"] == "CFS" and r["period_end"] == "2023-12-31")["value_krw"], "0")
        self.assertTrue(all(r["value_krw"] is None for r in records if r["scope"] == "OFS"))

    def test_ratio_preserves_negative_operating_result_and_rejects_zero_denominator(self):
        records, _, _, _, _ = fixture()
        revenue, operating = records[:2]
        self.assertEqual(fl.margin(revenue, operating), 20)
        operating["value_krw"] = "-40"
        self.assertEqual(fl.margin(revenue, operating), -20)
        revenue["value_krw"] = "0"
        self.assertIsNone(fl.margin(revenue, operating))
        revenue.update(value_krw="1", quantum="10")
        self.assertIsNone(fl.margin(revenue, operating))

    def test_receipt_day_blocked_next_day_eligible_with_same_document_comparatives(self):
        records, coverage, _, _, _ = fixture()
        features, _ = features_for_fixture(2)
        features[0]["as_of_date"] = "2024-03-12"; features[1]["as_of_date"] = "2024-03-13"
        paired, excluded = fl.attach_financials(features, records, coverage, {"005930": True})
        self.assertEqual(len(paired), 1); self.assertEqual(sum(excluded.values()), 1)
        self.assertEqual([paired[0][k] for k in fl.EXTRA_FEATURES], [20, 10])
        self.assertEqual(paired[0]["source_financial_document_id"], "005930_20240312000001")

    def test_unreadable_or_partial_latest_never_borrows_old_values(self):
        records, coverage, _, _, _ = fixture()
        features, _ = features_for_fixture(1)
        later = {**coverage[0], "receipt_no": "20240401000001", "receipt_date": "2024-04-01", "available_from": "2024-04-02T00:00:00+09:00", "financial_status": "unavailable"}
        paired, excluded = fl.attach_financials(features, records, coverage+[later], {"005930": True})
        self.assertEqual(paired, []); self.assertEqual(excluded["original_and_selected_receipt_differ"], 1)
        current_op = next(r for r in records if r["metric"] == "operating_income" and r["period_end"] == "2023-12-31" and r["scope"] == "CFS")
        current_op.update(status="fact_missing", value_krw=None)
        paired, _ = fl.attach_financials(features, records, coverage, {"005930": True})
        self.assertIsNone(paired[0][fl.EXTRA_FEATURES[0]])
        self.assertIsNone(paired[0][fl.EXTRA_FEATURES[1]])

    def test_no_previous_margin_in_same_receipt_does_not_borrow_another_receipt(self):
        records, coverage, _, _, _ = fixture()
        prior_op = next(r for r in records if r["metric"] == "operating_income" and r["period_end"] == "2022-12-31" and r["scope"] == "CFS")
        other = {**prior_op, "receipt_no": "20230312000001"}
        prior_op.update(status="fact_missing", value_krw=None)
        features, _ = features_for_fixture(1)
        paired, _ = fl.attach_financials(features, records+[other], coverage, {"005930": True})
        self.assertEqual(paired[0][fl.EXTRA_FEATURES[0]], 20)
        self.assertIsNone(paired[0][fl.EXTRA_FEATURES[1]])

    def test_incomplete_same_day_and_undimensioned_are_excluded(self):
        records, coverage, _, _, _ = fixture(); features, _ = features_for_fixture(1)
        for docs, completeness in ((coverage, {"005930": False}),
                (coverage+[{**coverage[0], "receipt_no": "20240312000002"}], {"005930": True})):
            paired, _ = fl.attach_financials(features, records, docs, completeness)
            self.assertEqual(paired, [])
        features[0]["financial_scope"] = "undimensioned"
        self.assertEqual(fl.attach_financials(features, records, coverage, {"005930": True})[0], [])


class ResearchModelTests(unittest.TestCase):
    def test_added_profitability_input_can_learn_a_synthetic_signal(self):
        rows = []
        for i in range(40):
            signal = -5.0 if i % 2 else 5.0
            f = dict.fromkeys(fl.MODEL_FEATURES, 0.0)
            f.update({fl.EXTRA_FEATURES[0]: signal, fl.EXTRA_FEATURES[1]: None, "as_of_date": "2024-01-01"})
            rows.append((f, {"price_return_pct": signal, "exit_date": "2024-01-10"}))
        base = fl.fit_model(rows, 0.1, fl.MODEL_FEATURES)
        extra = fl.fit_model(rows, 0.1, fl.MODEL_FEATURES+fl.EXTRA_FEATURES)
        self.assertEqual(fl.score_model(base, rows)["mae_pp"], 5)
        self.assertLess(fl.score_model(extra, rows)["mae_pp"], 0.02)
        self.assertGreater(extra["coefficients"][-2], 0)

    def examples(self):
        records, coverage, _, _, _ = fixture(); features, outcomes = features_for_fixture()
        original = train_models(features, outcomes)
        paired, _ = fl.attach_financials(features, records, coverage, {"005930": True})
        return paired, outcomes, original

    def test_same_cases_baseline_reproduces_original_and_no_promotion(self):
        features, outcomes, original = self.examples()
        result = fl.train_comparison(features, outcomes, original["split"], ["5", "20"], minimum_documents=1)
        self.assertEqual(result["status"], "trained")
        for horizon, c in result["models"].items():
            self.assertEqual(c["numeric"]["coefficients"], original["models"][horizon]["coefficients"])
            self.assertEqual(c["numeric"]["test_metrics"]["rows"], c["numeric_and_finance"]["test_metrics"]["rows"])
            self.assertLess(c["numeric_and_finance"]["training_label_end"], original["split"]["test_start"])
            self.assertGreater(c["excluded"]["cross_boundary"], 0)
            self.assertEqual(c["disposition"], "rejected")
        self.assertFalse(result["automatic_promotion"]); self.assertFalse(result["fresh_unseen_evidence"])

    def test_later_inputs_and_labels_cannot_change_fitting_or_tuning(self):
        features, outcomes, original = self.examples(); split = original["split"]
        # A synthetic varying input exercises the extra column rather than just constant columns.
        for i, f in enumerate(features): f[fl.EXTRA_FEATURES[0]] = None if i % 5 == 0 else i % 13
        first = fl.train_comparison(features, outcomes, split, ["5", "20"], minimum_documents=1)
        changed, labels = copy.deepcopy(features), copy.deepcopy(outcomes)
        for f in changed:
            if f["as_of_date"] >= split["test_start"]:
                f[fl.EXTRA_FEATURES[0]] = 999999; f["return_5_pct"] = -999999
        for o in labels:
            if o["as_of_date"] >= split["test_start"]: o["price_return_pct"] = -12345
        second = fl.train_comparison(changed, labels, split, ["5", "20"], minimum_documents=1)
        for h in first["models"]:
            for kind in ("numeric", "numeric_and_finance"):
                for k in ("alpha", "median", "mean", "scale", "coefficients", "intercept"):
                    self.assertEqual(first["models"][h][kind][k], second["models"][h][kind][k])
                self.assertNotEqual(first["models"][h][kind]["test_metrics"], second["models"][h][kind]["test_metrics"])

    def test_all_training_inputs_missing_or_too_few_documents_block(self):
        features, outcomes, original = self.examples()
        result = fl.train_comparison(features, outcomes, original["split"], ["5"])
        self.assertEqual(result["status"], "blocked")
        for f in features:
            if f["as_of_date"] < original["split"]["validation_start"]:
                for k in fl.EXTRA_FEATURES: f[k] = None
        result = fl.train_comparison(features, outcomes, original["split"], ["5"], minimum_documents=1)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("추가 재무 입력", result["models"]["5"]["reason"])

    def test_missing_entire_column_is_not_learned_from_later_values(self):
        features, outcomes, original = self.examples()
        for f in features:
            f[fl.EXTRA_FEATURES[1]] = None if f["as_of_date"] < original["split"]["test_start"] else 12345
        result = fl.train_comparison(features, outcomes, original["split"], ["5"], minimum_documents=1)
        model = result["models"]["5"]["numeric_and_finance"]
        self.assertEqual(model["input_observed_counts"][fl.EXTRA_FEATURES[1]], 0)
        self.assertEqual(model["coefficients"][-1], 0)

    def test_future_additional_metadata_and_excluded_horizon_rejected(self):
        features, outcomes, original = self.examples()
        with self.assertRaises(CollectionError): fl.train_comparison(features, outcomes, original["split"], ["60"])
        features[0]["source_financial_available_from"] = "2099-01-01T00:00:00+09:00"
        with self.assertRaises(CollectionError): fl.train_comparison(features, outcomes, original["split"], ["5"])


class FinancialResearchWorkflowTests(unittest.TestCase):
    def setUp(self):
        helper = evidence_fixtures.EvidenceWorkflowTests("test_full_offline_workflow_keeps_sources_models_and_both_value_types")
        helper.setUp(); self.addCleanup(helper.doCleanups)
        self.root, self.facts = helper.root, helper.facts
        self.learning = self.root / "learning/20261028_114327_12345678"
        original_summary = json.loads((self.learning / "summary.json").read_text())
        features, outcomes = features_for_fixture(); models = train_models(features, outcomes)
        saved = self.root / "synthetic_model"
        save_source(saved, models, features, outcomes)
        for p in saved.iterdir():
            if p.name != "summary.json": (self.learning / p.name).write_bytes(p.read_bytes())
        original_summary.update(json.loads((saved / "summary.json").read_text()))
        (self.learning / "summary.json").write_text(json.dumps(original_summary))
        raw = finance_payload(extra=OP); digest = hashlib.sha256(raw).hexdigest()
        event = {"receipt_no": "20240312000001", "receipt_date": "2024-03-12", "period_end": "2023-12-31", "report_name": "사업보고서 (2023.12)", "status": "verified", "sha256": digest, "metrics": parse_original(raw, {CORP}, date(2023, 12, 31))}
        (self.learning / "financial_audit.json").write_text(json.dumps({"005930": [event]}))
        cache = next((self.root / "financial_documents").iterdir())
        (cache / "original.zip").write_bytes(raw)
        meta = json.loads((cache / "source.json").read_text()); meta["sha256"] = digest
        (cache / "source.json").write_text(json.dumps(meta))
        self.source, _ = cross_check.run(self.facts, self.root, self.learning, progress=lambda _: None)

    def test_full_offline_workflow_preserves_all_old_files_and_source_proofs(self):
        for name in ("forward", "evidence_tables", "text_learning"):
            p = self.root / name / "preserve.txt"; p.parent.mkdir(); p.write_text("preserve")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch("kiwoom.get_client", side_effect=AssertionError("no Kiwoom")), patch("dart_client.stored_key", side_effect=AssertionError("no key")), patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no network")):
            target, summary, models = cli.run(self.source, self.learning, self.root, minimum_documents=1, progress=lambda _: None)
        self.assertEqual(summary["status"], "trained"); self.assertTrue(summary["research_training_executed"])
        self.assertEqual(summary["paired_rows"], 500); self.assertEqual(summary["narrative_records"], 1)
        self.assertEqual(summary["available_current_records"], 3)
        self.assertFalse(summary["existing_model_modified"]); self.assertFalse(summary["production_input_approved"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertTrue((target / "feature_joins.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertTrue(next((target / "financial_sources").glob("*/facts.html")).is_file())
        self.assertIn("같은 사례에서의 비교", (target / "report.html").read_text())
        for name, expected in summary["output_sha256"].items(): self.assertEqual(hashlib.sha256((target / name).read_bytes()).hexdigest(), expected)

    def test_changed_parent_or_different_learning_cannot_train(self):
        with self.assertRaises(CollectionError): cli.run(self.source, self.root / "other_learning", self.root, progress=lambda _: None)
        (self.learning / "financial_audit.json").write_text("{}")
        with self.assertRaises(CollectionError): cli.run(self.source, self.learning, self.root, progress=lambda _: None)

    def test_source_change_during_training_discards_new_output(self):
        original_train = cli.train_comparison
        def change_after_fit(*args, **kwargs):
            result = original_train(*args, **kwargs)
            path = self.learning / "models.json"
            path.write_bytes(path.read_bytes()+b" ")
            return result
        with patch.object(cli, "train_comparison", side_effect=change_after_fit), self.assertRaises(CollectionError):
            cli.run(self.source, self.learning, self.root, minimum_documents=1, progress=lambda _: None)
        self.assertEqual(list((self.root / "finance_learning").iterdir()), [])

    def test_report_escapes_source_text_and_missing_is_not_zero(self):
        target, summary, models = cli.run(self.source, self.learning, self.root, minimum_documents=1, progress=lambda _: None)
        records = json.loads((target / "financial_records.json").read_text()); snapshots = json.loads((target / "as_of.json").read_text())
        summary["exclusions"] = {"<script>alert('x')</script>": 1}
        report = cli.render_report(summary, records, snapshots, models)
        self.assertNotIn("<script>", report); self.assertIn("&lt;script&gt;", report)
        self.assertTrue(all(v is None for v in snapshots[1]["values"].values()))

    def test_failed_new_report_cleanup_and_cli_blocked_exit(self):
        target, _, _ = cli.run(self.source, self.learning, self.root, minimum_documents=1, progress=lambda _: None)
        before = set(target.parent.iterdir())
        with patch.object(cli, "render_report", side_effect=OSError("disk")), self.assertRaises(OSError):
            cli.run(self.source, self.learning, self.root, minimum_documents=1, progress=lambda _: None)
        self.assertEqual(before, set(target.parent.iterdir()))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["--root", str(self.root), "--learning", str(self.learning), "--input", str(self.source), "--no-open"]), 2)


if __name__ == "__main__": unittest.main()
