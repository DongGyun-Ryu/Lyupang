import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import business
import contextualize
from evidence_context import annotate, classify, mark_overlaps, release_outcomes, sentence_spans
from lyupang import DailyBar
from test_business import CORP, archive, filing
from test_research_data import series, snapshot


def event(day="2025-01-03", receipt="20250103000001"):
    return {"receipt_no": receipt, "receipt_date": day, "period_end": "2024-12-31",
            "available_from": (date.fromisoformat(day) + timedelta(days=1)).isoformat() + "T00:00:00+09:00"}


def prices():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(160)]
    bars = [DailyBar("005930", day.isoformat(), 100+i, 115+i, 90+i, 105+i, 1000+i, None, False, False)
            for i, day in enumerate(d for d in days if d.weekday() < 5)]
    end = bars[-1].date
    return {"code": "005930", "requested_start": bars[0].date, "requested_end": end,
            "adjustment_base_date": end, "adjusted": True}, bars


class ContextRulesTests(unittest.TestCase):
    def test_accounting_and_management_terms_do_not_establish_a_forecast(self):
        # Authored regression examples, not a held-out accuracy benchmark.
        texts = [
            "연결회사는 기대신용손실 모형으로 충당금을 설정하고 있습니다.",
            "당사는 기대 신용 손실을 측정하였습니다.",
            "당사는 회계추정의 변경을 반영하였습니다.",
            "당사는 추정 내용연수와 추정 잔존가치를 검토합니다.",
            "당사는 추정공정가치에 따라 자산을 측정합니다.",
            "당사는 예상신용손실을 인식하였습니다.",
            "당사는 자금수지 예측으로 유동성을 관리합니다.",
            "당사는 현금 흐름 예측을 정기적으로 수행합니다.",
            "계획 표준 생산 시간 대비 실제 가동시간을 계산합니다.",
            "연결회사는 확정급여계획을 운영합니다.",
            "당사는 퇴직 급여 계획을 운영합니다.",
        ]
        for text in texts:
            with self.subTest(text=text):
                result = classify(text, [], "삼성전자")
                self.assertNotIn(result["time_orientation"], {"outlook", "mixed"})
                self.assertNotIn("outlook", result["matched_expressions"])
                self.assertTrue(result["excluded_expressions"])
                self.assertEqual(result["status"], "needs_review")
                self.assertIn("technical_term_requires_context", result["review_reasons"])
                for cue in result["excluded_expressions"]:
                    self.assertEqual(text[cue["start"]:cue["end"]], cue["text"])
                    self.assertEqual(text[cue["context_start"]:cue["context_end"]], cue["context_text"])

    def test_independent_forecasts_survive_a_technical_term_in_the_same_sentence(self):
        texts = [
            "당사는 기대신용손실 증가를 예상합니다.",
            "당사는 회계추정 변경으로 내년 이익이 늘 것으로 전망합니다.",
            "당사는 자금수지 예측 시스템을 확대할 계획입니다.",
            "당사는 확정급여계획을 변경할 예정입니다.",
        ]
        for text in texts:
            with self.subTest(text=text):
                result = classify(text, [], "삼성전자")
                self.assertEqual(result["time_orientation"], "outlook")
                self.assertTrue(result["matched_expressions"]["outlook"])
                self.assertTrue(result["excluded_expressions"])
                self.assertEqual(result["status"], "needs_review")
        for text in ("당사는 내년 매출 증가를 기대합니다.", "내년 당사 매출은 10조원으로 추정됩니다.",
                     "당사는 생산설비를 확대할 계획입니다."):
            result = classify(text, [], "삼성전자")
            self.assertEqual(result["time_orientation"], "outlook")
            self.assertEqual(result["excluded_expressions"], [])

    def test_sales_destinations_and_company_actions_are_not_industry_subjects(self):
        cases = [
            ("당사는 국내시장에서 판매망을 확대하였습니다.", "company"),
            ("국내시장에서는 판매대리점을 통해 제품을 공급합니다.", "unclear"),
            ("글로벌 시장과 소통을 강화하였습니다.", "unclear"),
            ("당사는 세계 IT업계에서 위상을 높이고 있습니다.", "company"),
            ("아시아 시장을 중심으로 AI 프로젝트를 진행합니다.", "unclear"),
            ("대표 시장의 판매가격을 평균하여 산정하였습니다.", "unclear"),
            ("자동차 산업은 산업 연관 효과가 큽니다.", "industry"),
            ("TV 업계는 신제품 경쟁을 하고 있습니다.", "industry"),
            ("세계 반도체 시장은 성장하였습니다.", "industry"),
            ("시장 규모가 확대될 것으로 예상합니다.", "industry"),
            ("당사는 시장 수요가 증가할 것으로 예상합니다.", "mixed"),
        ]
        for text, subject in cases:
            with self.subTest(text=text):
                self.assertEqual(classify(text, [], "삼성전자")["subject"], subject)

    def test_company_industry_future_and_mixed_are_kept_distinct_with_rule_spans(self):
        cases = [
            ("당사의 매출은 전년 대비 10.9% 증가하였습니다.", "company", "reported"),
            ("세계 반도체 시장은 전년 대비 성장하였습니다.", "industry", "reported"),
            ("당사는 매출 성장을 예상하고 있습니다.", "company", "outlook"),
            ("당사의 매출은 증가하였으며 내년에는 시장 수요 확대를 예상합니다.", "mixed", "mixed"),
        ]
        for text, subject, orientation in cases:
            with self.subTest(text=text):
                result = classify(text, ["II. 사업의 내용"], "삼성전자")
                self.assertEqual(result["subject"], subject)
                self.assertEqual(result["time_orientation"], orientation)
                self.assertFalse(result["externally_verified"])
                for matches in result["matched_expressions"].values():
                    for match in matches:
                        self.assertEqual(match["text"], text[match["start"]:match["end"]])
        self.assertEqual(classify(cases[-1][0], [], "삼성전자")["status"], "needs_review")

    def test_prior_year_comparison_in_a_forecast_is_not_realized_performance(self):
        result = classify("당사 매출은 전년 대비 20% 증가할 전망입니다.", ["매출 및 수주상황"], "삼성전자")
        self.assertEqual(result["time_orientation"], "outlook")
        planned = classify("당사는 매출 확대를 추진했습니다.", [], "삼성전자")
        self.assertEqual(planned["time_orientation"], "mixed")
        self.assertEqual(planned["status"], "needs_review")
        unclear = classify("다양한 기술을 활용하는 새로운 서비스입니다.", [], "삼성전자")
        self.assertEqual(unclear["subject"], "unclear")
        self.assertEqual(unclear["status"], "needs_review")

    def test_sentence_offsets_preserve_decimal_and_joined_korean_sentences(self):
        text = "  당사 매출은 10.9% 증가하였습니다.또한 내년 20.5% 성장을 예상합니다.  "
        pieces = [text[a:b] for a, b in sentence_spans(text)]
        self.assertEqual(pieces, ["당사 매출은 10.9% 증가하였습니다.", "또한 내년 20.5% 성장을 예상합니다."])

    def test_annotation_uses_source_offsets_availability_and_skips_numeric_tables(self):
        text = "당사의 매출은 증가하였습니다. 향후 성장을 예상합니다."
        block = {"id": "b000001", "kind": "paragraph", "text": text, "headings": ["II. 사업의 내용"],
                 "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        table = {**block, "id": "b000002", "kind": "table"}
        statements, skipped = annotate([block, table], event(), "005930", "삼성전자")
        self.assertEqual(skipped, 1)
        self.assertEqual(len(statements), 2)
        for row in statements:
            self.assertEqual(row["text"], text[row["start"]:row["end"]])
            self.assertEqual(row["available_from"], "2025-01-04T00:00:00+09:00")
            self.assertNotIn("price_change_pct", row)

    def test_prices_enter_after_publication_and_use_next_observed_open_not_receipt_close(self):
        metadata, bars = prices()
        rows = release_outcomes(event(), "005930", metadata, bars, metadata["requested_end"])
        five = rows[0]
        self.assertEqual(five["entry_date"], "2025-01-06")
        self.assertEqual(five["exit_date"], "2025-01-10")
        first = next(b for b in bars if b.date == five["entry_date"])
        last = next(b for b in bars if b.date == five["exit_date"])
        self.assertAlmostEqual(five["price_change_pct"], (last.close / first.open - 1) * 100)
        self.assertFalse(five["calendar_verified"])
        self.assertFalse(five["strategy_return"])

    def test_pre_coverage_receipt_never_moves_to_late_first_price(self):
        metadata, bars = prices()
        rows = release_outcomes(event("2024-01-03", "20240103000001"), "005930", metadata, bars, metadata["requested_end"])
        self.assertTrue(all(r["status"] == "outside_coverage" and r["price_change_pct"] is None for r in rows))
        # Even a requested older start cannot turn missing old history into entry.
        metadata["requested_start"] = "2023-01-01"
        rows = release_outcomes(event("2024-01-03", "20240103000001"), "005930", metadata, bars, metadata["requested_end"])
        self.assertTrue(all(r["status"] == "outside_coverage" for r in rows))

    def test_reference_date_limits_future_bars_and_bad_volume_is_not_scored(self):
        metadata, bars = prices()
        rows = release_outcomes(event(), "005930", metadata, bars, "2025-01-07")
        self.assertTrue(all(r["status"] == "pending" for r in rows))
        bad = [replace(b, volume=0, zero_volume=True) if b.date == "2025-01-07" else b for b in bars]
        rows = release_outcomes(event(), "005930", metadata, bad, metadata["requested_end"])
        self.assertTrue(all(r["status"] == "quality_issue" and r["price_change_pct"] is None for r in rows))

    def test_same_day_and_overlapping_corrections_are_not_independent_cases(self):
        metadata, bars = prices()
        first = release_outcomes(event(), "005930", metadata, bars, metadata["requested_end"])
        second = release_outcomes(event(receipt="20250103000002"), "005930", metadata, bars, metadata["requested_end"])
        mark_overlaps(first + second)
        self.assertTrue(all(len(r["same_day_receipts"]) == 2 and r["overlaps_other_receipt"] for r in first + second))


class ContextWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        bars = series(count=200)
        self.snapshot = snapshot(bars, self.root / "data")
        self.research = self.root / "research" / "sample"
        self.research.mkdir(parents=True)
        batch = {"schema_version": 1, "requested_start": bars[0].date, "requested_end": bars[-1].date,
                 "stocks": [{"code": "005930", "source_snapshot": str(self.snapshot)}]}
        (self.research / "batch.json").write_text(json.dumps(batch), encoding="utf-8")
        client = Mock()
        client.companies.return_value = {"005930": {"corp_code": CORP, "corp_name": "삼성전자"}}
        client.company_identifiers.return_value = {CORP}
        client.disclosures.return_value = [filing()]
        client.original_document.return_value = archive()
        self.business, _ = business.run(self.research, self.root / "business", self.root / "business_documents", client, progress=lambda _: None)
        (self.root / "forward").mkdir()
        (self.root / "forward" / "frozen.json").write_text("fixed journal", encoding="utf-8")

    def run_context(self):
        return contextualize.run(self.business, self.root, progress=lambda _: None)

    def test_end_to_end_preserves_sources_and_frozen_journal_and_links_once_per_receipt(self):
        original = {p: p.read_bytes() for folder in (self.business, self.research, self.snapshot, self.root / "forward")
                    for p in folder.rglob("*") if p.is_file()}
        with patch("learning.fit", side_effect=AssertionError("no training")), patch("dart_client.stored_key", side_effect=AssertionError("no key")), patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no network")):
            target, summary = self.run_context()
        self.assertEqual(summary["status"], "linked")
        self.assertEqual(summary["outcome_counts"], {"observed": 3})
        outcomes = json.loads((target / "release_outcomes.json").read_text())
        statements = [json.loads(line) for line in (target / "statements.jsonl").read_text().splitlines()]
        self.assertGreater(len(statements), len(outcomes))
        self.assertEqual(len(outcomes), 3)
        self.assertTrue(all("price_change_pct" not in row for row in statements))
        self.assertEqual(hashlib.sha256((target / "statements.jsonl").read_bytes()).hexdigest(), summary["statement_sha256"])
        self.assertFalse(summary["live_trading_allowed"])
        self.assertFalse(summary["fresh_unseen_evidence"])
        self.assertEqual(summary["classification_policy"], "sentence_context_rules_v2")
        report = (target / "report.html").read_text()
        self.assertIn("용어 혼동 점검", report)
        self.assertIn("정확도가 높아졌다고 판단하지 않습니다", report)
        self.assertEqual(original, {p: p.read_bytes() for p in original})

    def test_modified_text_hash_blocks_classification_without_inventing_replacement(self):
        source = next((self.business / "sources").glob("*.json"))
        source.write_text(source.read_text() + " ", encoding="utf-8")
        _, summary = self.run_context()
        self.assertEqual(summary["read_documents"], 0)
        self.assertEqual(summary["status"], "partial")
        self.assertIn("해시", " ".join(summary["stocks"][0]["errors"]))

    def test_price_corruption_keeps_text_but_marks_all_outcomes_unavailable(self):
        with (self.snapshot / "daily.csv").open("a", encoding="utf-8") as stream:
            stream.write("changed")
        _, summary = self.run_context()
        self.assertEqual(summary["read_documents"], 1)
        self.assertGreater(summary["counts"]["sentences"], 0)
        self.assertEqual(summary["outcome_counts"], {"source_error": 3})
        self.assertEqual(summary["status"], "partial")

    def test_price_changes_do_not_change_statement_classification(self):
        first, _ = self.run_context()
        # Swap in a separately verified price snapshot and update only its provenance.
        changed = [replace(b, open=b.open*2, high=b.high*2, low=b.low*2, close=b.close*2) for b in series(count=200)]
        new_snapshot = snapshot(changed, self.root / "data")
        batch = json.loads((self.research / "batch.json").read_text())
        batch["stocks"][0]["source_snapshot"] = str(new_snapshot)
        (self.research / "batch.json").write_text(json.dumps(batch), encoding="utf-8")
        evidence = json.loads((self.business / "evidence.json").read_text())
        evidence["source_batch_sha256"] = hashlib.sha256((self.research / "batch.json").read_bytes()).hexdigest()
        (self.business / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        second, _ = self.run_context()
        self.assertEqual((first / "statements.jsonl").read_bytes(), (second / "statements.jsonl").read_bytes())

    def test_report_failure_leaves_no_completed_folder_and_cli_uses_existing_files(self):
        with patch("contextualize.render_report", side_effect=OSError("disk")), self.assertRaises(OSError):
            self.run_context()
        self.assertEqual(list((self.root / "context").iterdir()), [])
        output = io.StringIO()
        with redirect_stdout(output):
            code = contextualize.main(["--root", str(self.root), "--no-open"])
        self.assertEqual(code, 0)
        self.assertIn("설명 구분·가격 연결 완료", output.getvalue())


if __name__ == "__main__":
    unittest.main()
