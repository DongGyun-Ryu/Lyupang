import hashlib
import html
import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import business
import contextualize
import textcheck
import text_validation as tv
from lyupang import CollectionError
from test_business import CORP, archive, filing, xml_document
from test_research_data import series, snapshot


def labels_for(pack, answers):
    manifest, _, _ = tv.load_pack(pack)
    return {"schema_version": 1, "pack_id": manifest["pack_id"], "pack_hash": manifest["pack_hash"],
            "rubric": tv.RUBRIC, "annotation_source": "human_entered", "reviewer_attestation": True, "answers": answers}


def answer(item, subject="company", orientation="reported", state="answered"):
    return {"id": item["id"], "text_sha256": item["text_sha256"], "state": state,
            "subject": subject if state == "answered" else None,
            "time_orientation": orientation if state == "answered" else None, "note": ""}


class AgreementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pack = self.root / "pack"
        self.pack.mkdir()
        self.items = [{"id": str(i), "company_name": "테스트 회사", "text": f"기능 검사용 문장 {i}",
                       "text_sha256": tv.digest(f"기능 검사용 문장 {i}".encode())} for i in range(4)]
        axes = [("company", "reported"), ("industry", "outlook"), ("mixed", "mixed"), ("company", "outlook")]
        predictions = [{"id": item["id"], "text_sha256": item["text_sha256"], "subject": axis[0],
                        "time_orientation": axis[1], "sample_group": "general" if i < 2 else "challenge",
                        "routing": "rule_based_candidate"} for i, (item, axis) in enumerate(zip(self.items, axes))]
        manifest = {"schema_version": 1, "pack_id": "a"*24, "rubric": tv.RUBRIC, "classification_policy": "sentence_context_rules_v2",
                    "sample_size": 4, "items_sha256": tv.digest(tv.canonical(self.items)),
                    "predictions_sha256": tv.digest(tv.canonical(predictions))}
        manifest["pack_hash"] = tv.digest(tv.canonical(manifest))
        for name, data in (("manifest.json", manifest), ("items.json", self.items), ("predictions.json", predictions)):
            (self.pack / name).write_bytes(tv.canonical(data))
        self.labels = self.root / "answers.json"

    def write_answers(self, answers):
        content = labels_for(self.pack, answers)
        self.labels.write_bytes(tv.canonical(content))
        return content

    def test_hand_counted_agreement_keeps_strata_abstentions_and_missing_answers_separate(self):
        # Test-only invented human answers, never evidence that someone reviewed data.
        self.write_answers([answer(self.items[0]), answer(self.items[1]), answer(self.items[2], state="abstain")])
        summary, comparisons = tv.evaluate(self.pack, self.labels)
        general, challenge = summary["groups"]
        self.assertEqual((summary["answered"], summary["abstained"], summary["unanswered"]), (2, 1, 1))
        self.assertEqual(general["subject_agreement_pct"], 50.0)
        self.assertEqual(general["time_orientation_agreement_pct"], 50.0)
        self.assertEqual(general["joint_agreement_pct"], 50.0)
        self.assertEqual(general["subject_confusion"]["company"]["industry"], 1)
        self.assertEqual(general["candidate_joint_disagreements"], 1)
        self.assertIsNone(challenge["joint_agreement_pct"])
        for flag in ("fresh_unseen_evidence", "population_accuracy_established", "model_approved",
                     "training_executed", "live_trading_allowed", "ground_truth_independently_verified"):
            self.assertIs(summary[flag], False)
        page = tv.evaluation_html(summary, comparisons, self.items)
        self.assertIn("50.0%", page)
        self.assertIn("유보·미답변은 정답으로 채우거나", page)
        self.assertIn("새 미사용 평가가 아니며", page)

    def test_all_abstentions_produce_no_agreement_percentage(self):
        self.write_answers([answer(item, state="abstain") for item in self.items])
        summary, _ = tv.evaluate(self.pack, self.labels)
        self.assertEqual(summary["answered"], 0)
        self.assertTrue(all(group["joint_agreement_pct"] is None for group in summary["groups"]))

    def test_wrong_pack_duplicate_unknown_labels_and_missing_attestation_are_rejected(self):
        baseline = self.write_answers([answer(self.items[0])])
        invalids = [
            {**baseline, "pack_hash": "wrong"}, {**baseline, "reviewer_attestation": False},
            {**baseline, "annotation_source": "automatically_generated"},
            {**baseline, "answers": baseline["answers"]*2},
            {**baseline, "answers": [{**baseline["answers"][0], "text_sha256": "wrong"}]},
            {**baseline, "answers": [{**baseline["answers"][0], "subject": "anything"}]},
            {**baseline, "answers": [{**baseline["answers"][0], "state": "abstain"}]},
            {**baseline, "answers": []},
        ]
        for content in invalids:
            self.labels.write_bytes(tv.canonical(content))
            with self.subTest(content=content), self.assertRaises(CollectionError):
                tv.evaluate(self.pack, self.labels)

    def test_changed_predictions_and_duplicate_json_keys_are_not_accepted(self):
        self.write_answers([answer(self.items[0])])
        with (self.pack / "predictions.json").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaises(CollectionError):
            tv.evaluate(self.pack, self.labels)
        self.labels.write_text('{"schema_version":1,"schema_version":2}')
        with self.assertRaises(CollectionError):
            tv.read_json(self.labels)

    def test_newest_broken_download_does_not_fall_back_to_older_answers(self):
        old = self.root / ("Lyupang-text-labels-" + "a"*24 + ".json")
        new = self.root / ("Lyupang-text-labels-" + "a"*24 + " (1).json")
        old.write_text("valid older file")
        new.write_text("broken latest file")
        os.utime(old, ns=(1_000, 1_000))
        os.utime(new, ns=(2_000, 2_000))
        self.assertEqual(textcheck.downloaded_labels(self.pack, [self.root]), new)


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        bars = series(count=200)
        snap = snapshot(bars, self.root / "data")
        research = self.root / "research" / "sample"
        research.mkdir(parents=True)
        (research / "batch.json").write_text(json.dumps({"schema_version": 1, "requested_start": bars[0].date,
            "requested_end": bars[-1].date, "stocks": [{"code": "005930", "source_snapshot": str(snap)}]}))
        variants = ["당사는 매출이 증가하였습니다.", "자동차 산업은 수요 증가를 예상합니다.",
                    "당사는 기대신용손실 모형을 사용합니다.", "새로운 제품을 개발하고 있습니다."]
        paragraphs = [f"<P>항목{i}: {html.escape(variants[i%4])}</P>" for i in range(60)]
        paragraphs.append("<P>" + html.escape('당사는 </script><script>globalThis.injected=1</script> 문자를 표시합니다.') + "</P>")
        client = Mock()
        client.companies.return_value = {"005930": {"corp_code": CORP, "corp_name": "삼성전자"}}
        client.company_identifiers.return_value = {CORP}
        client.disclosures.return_value = [filing()]
        client.original_document.return_value = archive(xml_document(extra="".join(paragraphs)))
        source, _ = business.run(research, self.root / "business", self.root / "business_documents", client, progress=lambda _: None)
        self.source, _ = contextualize.run(source, self.root, progress=lambda _: None)
        (self.root / "forward").mkdir()
        (self.root / "forward" / "frozen.json").write_text("existing predictions")

    def test_pack_is_reused_blind_and_price_free_and_sources_stay_unchanged(self):
        before = {p: p.read_bytes() for path in (self.source, self.root / "forward") for p in path.rglob("*") if p.is_file()}
        with patch("evidence_context.classify", side_effect=AssertionError("no reclassification")), patch("learning.fit", side_effect=AssertionError("no fit")), patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no network")), patch("dart_client.stored_key", side_effect=AssertionError("no key")):
            pack = textcheck.prepare(self.source, self.root)
            old = {p: p.read_bytes() for p in pack.iterdir()}
            self.assertEqual(textcheck.prepare(self.source, self.root), pack)
        self.assertEqual(old, {p: p.read_bytes() for p in old})
        manifest, items, predictions = tv.load_pack(pack)
        self.assertEqual(len(items), 24)
        self.assertEqual(len({p["dedup_group"] for p in predictions}), 24)
        self.assertEqual(sum(p["sample_group"] == "general" for p in predictions), 12)
        page = (pack / "review.html").read_text()
        embedded = json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', page, re.S)[1])
        for forbidden in ("predictions", "classification", "price_change_pct", "routing", "sample_group"):
            self.assertNotIn(forbidden, json.dumps(embedded))
        self.assertNotIn("reviewer_answers", {p.name for p in pack.iterdir()})
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_sampling_deduplicates_whitespace_and_is_stable(self):
        _, rows, _, _ = tv.source_rows(self.source)
        extra = {**rows[0], "id": "duplicate", "text": "  " + rows[0]["text"].replace(" ", "  ")}
        first, count = tv.sample_rows(rows + [extra], "fixed")
        second, other_count = tv.sample_rows(list(reversed(rows + [extra])), "fixed")
        self.assertEqual(first, second)
        self.assertEqual(count, other_count)
        _, original_count = tv.sample_rows(rows, "fixed")
        self.assertEqual(count, original_count)

    def test_embedded_source_text_cannot_end_the_data_script_or_run_markup(self):
        _, rows, _, _ = tv.source_rows(self.source)
        attack = next(row for row in rows if "globalThis.injected" in row["text"])
        with patch("text_validation.sample_rows", return_value=([(attack, "general", "group")], len(rows))):
            tv.make_pack(self.source, self.root / "attack-pack")
        page = (self.root / "attack-pack" / "review.html").read_text()
        self.assertNotIn("</script><script>globalThis.injected", page)
        embedded = json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', page, re.S)[1])
        self.assertEqual(embedded["items"][0]["text"], attack["text"])
        self.assertIn("$('quote').textContent=item.text", page)

    def test_corrupt_statements_or_source_blocks_block_sampling(self):
        path = self.source / "statements.jsonl"
        old = path.read_bytes()
        path.write_bytes(old + b" ")
        with self.assertRaises(CollectionError):
            textcheck.prepare(self.source, self.root)
        path.write_bytes(old)
        block = next((self.source / "sources").glob("*.json"))
        block.write_text("[]")
        with self.assertRaises(CollectionError):
            textcheck.prepare(self.source, self.root)

    def test_label_round_trip_archives_answers_without_changing_frozen_pack(self):
        pack = textcheck.prepare(self.source, self.root)
        _, items, _ = tv.load_pack(pack)
        labels = self.root / "answers.json"
        labels.write_bytes(tv.canonical(labels_for(pack, [answer(items[0]), answer(items[1], state="abstain")])))
        before = {p: p.read_bytes() for p in pack.iterdir()}
        result, summary = textcheck.record_evaluation(pack, labels, self.root)
        self.assertEqual(summary["answered"], 1)
        self.assertEqual(summary["abstained"], 1)
        self.assertEqual(summary["unanswered"], 22)
        self.assertEqual((result / "reviewer_answers.json").read_bytes(), labels.read_bytes())
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_atomic_pack_and_report_failure_do_not_leave_completed_results(self):
        with patch("pathlib.Path.read_text", side_effect=OSError("template unavailable")), self.assertRaises(OSError):
            textcheck.prepare(self.source, self.root)
        self.assertEqual(list((self.root / "text_validation" / "packs").iterdir()), [])
        pack = textcheck.prepare(self.source, self.root)
        _, items, _ = tv.load_pack(pack)
        labels = self.root / "answers.json"
        labels.write_bytes(tv.canonical(labels_for(pack, [answer(items[0])])))
        with patch("textcheck.evaluation_html", side_effect=OSError("disk")), self.assertRaises(OSError):
            textcheck.record_evaluation(pack, labels, self.root)
        self.assertEqual(list((self.root / "text_validation" / "results").iterdir()), [])

    def test_cli_opens_unanswered_review_and_accepts_explicit_old_pack_without_context(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = textcheck.main(["--root", str(self.root), "--no-open", "--downloads", str(self.root / "empty")])
        self.assertEqual(code, 0)
        self.assertIn("아직 사람의 답을 입력하지 않았습니다", output.getvalue())
        pack = next((self.root / "text_validation" / "packs").iterdir())
        _, items, _ = tv.load_pack(pack)
        labels = self.root / "answers.json"
        labels.write_bytes(tv.canonical(labels_for(pack, [answer(items[0])])))
        with patch("textcheck.latest_context", side_effect=AssertionError("old pinned pack")), redirect_stdout(io.StringIO()):
            code = textcheck.main([str(labels), "--root", str(self.root), "--no-open"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
