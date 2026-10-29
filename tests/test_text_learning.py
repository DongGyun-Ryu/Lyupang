import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import business
import learn_text
import text_learning as tl
from learning import train_models
from lyupang import CollectionError
from observe import append_record, digest
from test_business import archive, xml_document
from test_learning import dataset
from test_review import save_source


def document(text, received):
    return {"text": text, "terms": tl.fragments(text), "sha256": hashlib.sha256(text.encode()).hexdigest(), "receipt_date": received}


class TextModelTests(unittest.TestCase):
    def test_text_columns_can_learn_a_signal_when_numeric_columns_cannot(self):
        docs = {key: document("수주확대 성장개선" if key.startswith("g") else "수요감소 손실위험", "2023-03-10")
                for key in ("g1", "g2", "r1", "r2")}
        rows = []
        for key in docs:
            for _ in range(10):
                feature = dict.fromkeys(tl.MODEL_FEATURES, 0.0)
                feature["text_document_id"] = key
                feature["as_of_date"] = "2024-01-01"
                rows.append((feature, {"price_return_pct": 5.0 if key.startswith("g") else -5.0, "exit_date": "2024-01-10"}))
        vocabulary = tl.fit_vocabulary(docs, docs)
        numeric = tl.fit_candidate(rows, 1.0, None, docs)
        combined = tl.fit_candidate(rows, 1.0, vocabulary, docs)
        self.assertEqual(tl.score_candidate(numeric, rows, docs)["mae_pp"], 5.0)
        self.assertLess(tl.score_candidate(combined, rows, docs)["mae_pp"], 1.0)
        self.assertEqual(len(combined["coefficients"]), len(tl.MODEL_FEATURES)+len(vocabulary["vocabulary"]))

    def examples(self):
        base_features, base_outcomes = dataset()
        features, outcomes, histories, docs = [], [], {}, {}
        for n, code in enumerate(("005930", "005380", "035420")):
            features.extend({**f, "code": code} for f in base_features)
            outcomes.extend({**o, "code": code} for o in base_outcomes)
            events = []
            for year in (2023, 2024, 2025):
                receipt = f"{year}0310{code[-4:]}01"
                day = f"{year}-03-10"
                events.append({"receipt_no": receipt, "receipt_date": day, "period_end": f"{year-1}-12-31", "status": "read",
                               "available_from": f"{year}-03-11T00:00:00+09:00"})
                docs[code+"_"+receipt] = document(f"제품시장 서비스수요 성장위험 기술투자 " + ("반도체생산" if n == 0 else "자동차판매" if n == 1 else "인터넷광고") + f" 미래전망{year}", day)
            histories[code] = events
        original = train_models(features, outcomes)
        paired, _ = tl.attach_text(features, histories, docs)
        return paired, outcomes, docs, original["split"], histories

    def test_publication_and_unreadable_or_same_day_correction_never_fall_back(self):
        features = [{"code": "005930", "as_of_date": "2024-03-10"}, {"code": "005930", "as_of_date": "2024-03-11"}]
        events = [
            {"receipt_no": "20230310000001", "receipt_date": "2023-03-10", "period_end": "2022-12-31", "status": "read", "available_from": "2023-03-11T00:00:00+09:00"},
            {"receipt_no": "20240310000001", "receipt_date": "2024-03-10", "period_end": "2023-12-31", "status": "read", "available_from": "2024-03-11T00:00:00+09:00"}]
        docs = {"005930_"+e["receipt_no"]: document("제품시장 성장과 매출", e["receipt_date"]) for e in events}
        paired, _ = tl.attach_text(features, {"005930": events}, docs)
        self.assertEqual([row["text_document_id"] for row in paired], ["005930_20230310000001", "005930_20240310000001"])
        broken = copy.deepcopy(events)
        broken[-1]["status"] = "unavailable"
        paired, reasons = tl.attach_text(features, {"005930": broken}, docs)
        self.assertEqual(len(paired), 1)
        self.assertEqual(reasons["latest_report_unreadable"], 1)
        ambiguous = events + [{**events[-1], "receipt_no": "20240310000002"}]
        paired, reasons = tl.attach_text(features, {"005930": ambiguous}, docs)
        self.assertEqual(len(paired), 1)
        self.assertEqual(reasons["ambiguous_same_day_reports"], 1)

    def test_vocabulary_counts_unique_receipts_not_repeated_daily_rows(self):
        docs = {"a": document("반도체기술 회사매출", "2023-03-10"), "b": document("반도체수요 회사매출", "2023-03-11"),
                "c": document("자동차수요 회사매출", "2023-03-12")}
        first = tl.fit_vocabulary(["a", "b", "c"], docs)
        second = tl.fit_vocabulary(["a"]*100+["b", "c"], docs)
        self.assertEqual(first, second)
        self.assertNotIn("회사", first["vocabulary"])
        self.assertNotIn("2023", tl.fragments("2023년 매출 100억원"))

    def test_future_test_text_labels_and_numeric_values_do_not_change_fitted_models(self):
        features, outcomes, docs, split, _ = self.examples()
        first = tl.train_comparison(features, outcomes, docs, split, ["5", "20"])
        self.assertEqual(first["status"], "trained")
        altered_features, altered_outcomes, altered_docs = copy.deepcopy(features), copy.deepcopy(outcomes), copy.deepcopy(docs)
        train_ids = set(first["models"]["5"]["numeric_and_text"]["text_vocabulary"]["training_document_ids"])
        test_only = {f["text_document_id"] for f in features if f["as_of_date"] >= split["test_start"]}-train_ids
        self.assertTrue(test_only)
        for key in test_only:
            altered_docs[key] = document("완전히새로운단어만 존재하는 비교원문", docs[key]["receipt_date"])
        for feature in altered_features:
            if feature["as_of_date"] >= split["test_start"]:
                feature["return_5_pct"] = 99999
                feature["text_sha256"] = altered_docs[feature["text_document_id"]]["sha256"]
        for outcome in altered_outcomes:
            if outcome["as_of_date"] >= split["test_start"]:
                outcome["price_return_pct"] = -9999
        second = tl.train_comparison(altered_features, altered_outcomes, altered_docs, split, ["5", "20"])
        for horizon in first["models"]:
            for kind in ("numeric", "numeric_and_text"):
                left, right = first["models"][horizon][kind], second["models"][horizon][kind]
                for key in ("alpha", "median", "mean", "scale", "coefficients", "intercept", "text_vocabulary"):
                    self.assertEqual(left[key], right[key])
                self.assertNotEqual(left["test_metrics"], right["test_metrics"])
                self.assertLess(left["training_label_end"], split["test_start"])
        self.assertFalse(first["automatic_promotion"])
        self.assertFalse(first["fresh_unseen_evidence"])

    def test_insufficient_documents_and_no_held_horizons_do_not_fake_text_training(self):
        features, outcomes, docs, split, _ = self.examples()
        result = tl.train_comparison(features, outcomes, docs, split, ["5"], minimum_documents=100)
        self.assertEqual(result["status"], "blocked")
        self.assertNotIn("numeric_and_text", result["models"]["5"])
        empty = tl.train_comparison(features, outcomes, docs, split, [])
        self.assertIn("탈락 모델", empty["reason"])

    def test_core_rejects_wrong_document_hash_or_future_text_availability(self):
        features, outcomes, docs, split, _ = self.examples()
        for key, value in (("text_sha256", "changed"), ("text_available_from", "2030-01-01T00:00:00+09:00")):
            bad = copy.deepcopy(features)
            bad[0][key] = value
            with self.assertRaises(CollectionError):
                tl.train_comparison(bad, outcomes, docs, split, ["5"])


class TextWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        features, outcomes, _, _, _ = TextModelTests().examples()
        # The original numeric source does not contain any text-only columns.
        self.original = train_models(features, outcomes)
        self.learning = self.root / "learning" / "sample"
        save_source(self.learning, self.original, features, outcomes)
        research = self.root / "research" / "sample"
        research.mkdir(parents=True)
        codes = ("005930", "005380", "035420")
        (research / "batch.json").write_text(json.dumps({"schema_version": 1, "requested_start": "2024-01-01", "requested_end": "2025-05-14",
                                                       "stocks": [{"code": code} for code in codes]}))
        companies = {code: {"corp_code": f"0000000{i+1}", "corp_name": f"테스트기업{i+1}"} for i, code in enumerate(codes)}
        payloads, filings = {}, {}
        for code, company in companies.items():
            filings[company["corp_code"]] = []
            for year in (2023, 2024, 2025):
                receipt = f"{year}0310{code[-4:]}01"
                filings[company["corp_code"]].append({"receipt_no": receipt, "receipt_date": f"{year}-03-10", "report_name": f"사업보고서 ({year-1}.12)"})
                sector = {"005930": "반도체생산 기술투자", "005380": "자동차판매 수출확대", "035420": "인터넷광고 서비스성장"}[code]
                text = f"<P>제품시장 서비스수요 기술투자 성장위험 {sector} 사업을 운영하고 있습니다.</P>"
                payloads[receipt] = archive(xml_document(corp=company["corp_code"], extra=text), receipt=receipt)
        client = Mock()
        client.companies.return_value = companies
        client.disclosures.side_effect = lambda corp, *args, **kwargs: filings[corp]
        client.original_document.side_effect = lambda receipt: payloads[receipt]
        self.business, summary = business.run(research, self.root / "business", self.root / "business_documents", client, progress=lambda _: None)
        self.assertEqual(summary["read_reports"], 9)
        experiment = self.root / "forward" / "experiment"
        experiment.mkdir(parents=True)
        append_record(experiment / "experiment.json", {"source_learning": str(self.learning), "model_sha256": digest(self.original)})
        (experiment / "kept.txt").write_text("existing fixed forecasts")

    def run_text(self):
        return learn_text.run(self.learning, self.business, self.root, progress=lambda _: None)

    def test_full_training_is_offline_pairs_same_cases_and_preserves_fixed_experiment(self):
        before = {p: p.read_bytes() for folder in (self.learning, self.business, self.root / "forward") for p in folder.rglob("*") if p.is_file()}
        with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no network")), patch("dart_client.stored_key", side_effect=AssertionError("no keys")):
            target, summary, models = self.run_text()
        self.assertEqual(models["status"], "trained")
        self.assertEqual(summary["paired_rows"], summary["source_rows"])
        self.assertEqual(summary["horizons"], ["5", "20"])
        self.assertFalse(summary["frozen_observer_modified"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        for model in models["models"].values():
            self.assertEqual(model["numeric"]["test_metrics"]["rows"], model["numeric_and_text"]["test_metrics"]["rows"])
        report = (target / "report.html").read_text()
        self.assertIn("이미 확인한 과거 자료", report)
        self.assertIn("새 미사용 검증은 아닙니다", report)
        self.assertIn("같은 사례에서의 비교", report)
        self.assertTrue((target / "text_joins.csv").exists())
        index = learn_text.read_json(target / "document_index.json")
        for example in summary["text_examples"]:
            text = (target / "sources" / (example["document_id"]+".txt")).read_text()
            self.assertEqual(text[example["start"]:example["end"]], example["excerpt"])
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), example["text_sha256"])
            self.assertEqual(index[example["document_id"]]["sha256"], example["text_sha256"])
        self.assertEqual(hashlib.sha256((target / "historical_comparisons.csv").read_bytes()).hexdigest(), summary["comparison_sha256"])

    def test_corrupt_latest_text_excludes_its_period_without_old_source_substitution(self):
        data = learn_text.read_json(self.business / "evidence.json")
        event = data["stocks"][0]["events"][-1]
        (self.business / event["source_blocks"]).write_text("changed")
        _, summary, _ = self.run_text()
        self.assertLess(summary["paired_rows"], summary["source_rows"])
        self.assertFalse(summary["source_complete"])
        self.assertTrue(any(item.get("reason") for item in summary["document_audit"]))

    def test_missing_company_report_list_is_not_claimed_complete(self):
        data = learn_text.read_json(self.business / "evidence.json")
        missing = data["stocks"].pop()["code"]
        (self.business / "evidence.json").write_text(json.dumps(data))
        _, summary, _ = self.run_text()
        self.assertFalse(summary["source_complete"])
        self.assertEqual(summary["missing_business_codes"], [missing])
        self.assertLess(summary["paired_rows"], summary["source_rows"])

    def test_learning_source_is_pinned_and_mismatched_original_is_not_silently_replaced(self):
        self.assertEqual(learn_text.select_learning(self.root), self.learning)
        (self.learning / "models.json").write_text("{}")
        with self.assertRaises(CollectionError):
            learn_text.select_learning(self.root)

    def test_atomic_report_failure_keeps_sources_and_no_completed_result(self):
        with patch("learn_text.render_report", side_effect=OSError("disk")), self.assertRaises(OSError):
            self.run_text()
        self.assertEqual(list((self.root / "text_learning").iterdir()), [])

    def test_cli_uses_existing_sources_without_prompting_or_replacing_observer(self):
        out = io.StringIO()
        with redirect_stdout(out):
            result = learn_text.main(["--root", str(self.root), "--no-open"])
        self.assertEqual(result, 0)
        self.assertIn("연구용 학습·과거 비교 완료", out.getvalue())
        self.assertIn("기존 고정 모델·예측 기록은 유지", out.getvalue())


if __name__ == "__main__":
    unittest.main()
