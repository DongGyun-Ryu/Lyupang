import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import business
import cross_check
import facts
import learn_finance
import observe
import observe_finance as shadow
from finance_learning import EXTRA_FEATURES
from financial_history import financial_as_of, parse_original
from learning import MODEL_FEATURES, index_outcomes, partition_examples, train_models
from lyupang import CollectionError, DailyBar, SEOUL
from test_business import CORP, archive, filing, xml_document
from test_financial_history import original
from test_observe import Prices
from test_review import save_source


def held_study(root):
    """Synthetic sources/returns with a profitability signal, never market evidence."""
    research = root / "research/fixture"; (research / "dart").mkdir(parents=True)
    (research / "batch.json").write_text(json.dumps({"schema_version": 1, "requested_start": "2023-04-01", "requested_end": "2026-07-13", "stocks": [{"code": "005930"}]}))
    (research / "dart/005930.json").write_text(json.dumps({"company": {"corp_code": CORP}}))
    filings, payloads, audit = [], {}, []
    op = {2021: 40, 2022: 20, 2023: -10, 2024: 60, 2025: 10}
    for year in (2022, 2023, 2024, 2025):
        receipt, received = f"{year+1}0312000001", f"{year+1}-03-12"
        event = filing(receipt=receipt, day=received, title=f"사업보고서 ({year}.12)"); filings.append(event)
        extra = f'<ifrs:ProfitLossFromOperatingActivities contextRef="duration" unitRef="currency" decimals="INF">{op[year]}</ifrs:ProfitLossFromOperatingActivities><ifrs:ProfitLossFromOperatingActivities contextRef="previous" unitRef="currency" decimals="INF">{op[year-1]}</ifrs:ProfitLossFromOperatingActivities>'
        raw = original(year=year, scope="CFS", extra=extra)
        # Add declared precision to the other monetary facts as well.
        import zipfile
        with zipfile.ZipFile(io.BytesIO(raw)) as z: xml = z.read("report.xbrl").replace(b'unitRef="currency">', b'unitRef="currency" decimals="INF">')
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z: z.writestr("report.xbrl", xml)
        raw = out.getvalue(); sha = hashlib.sha256(raw).hexdigest()
        metrics = parse_original(raw, {CORP}, date(year, 12, 31))
        audit.append({**event, "period_end": f"{year}-12-31", "status": "verified", "sha256": sha, "metrics": metrics})
        payloads[receipt] = archive(xml_document(extra=f'<P>{year}년 당사의 연결 기준 매출은 200원입니다.</P>'), receipt=receipt)
        cache = root / "financial_documents" / f"005930_{receipt}_12345678"; cache.mkdir(parents=True)
        (cache / "original.zip").write_bytes(raw)
        (cache / "source.json").write_text(json.dumps({"provider": "OpenDART", "endpoint": "fnlttXbrl.xml", "code": "005930", "receipt_no": receipt, "sha256": sha}))
    client = Mock(); client.companies.return_value = {"005930": {"corp_code": CORP, "corp_name": "삼성전자"}}
    client.company_identifiers.return_value = {CORP}; client.disclosures.return_value = filings
    client.original_document.side_effect = lambda receipt: payloads[receipt]
    source, _ = business.run(research, root / "business", root / "business_documents", client, progress=lambda _: None)
    text, _ = facts.run(source, root, progress=lambda _: None)
    features, outcomes = [], []
    first, last = date(2023, 4, 1), date(2026, 7, 13)
    for i in range((last-first).days+1):
        day = first+timedelta(days=i); x = math.sin(i/11)
        f = dict.fromkeys(MODEL_FEATURES)
        f.update(code="005930", as_of_date=day.isoformat(), available_from=(day+timedelta(days=1)).isoformat()+"T00:00:00+09:00",
            quality_issue=False, return_5_pct=x, return_20_pct=2*x, return_60_pct=3*x, volume_ratio_previous_20=1, disclosures_previous_7_days=0)
        f.update(financial_as_of(audit, day)); features.append(f)
        target = x*3 + op[int(f["financial_period_end"][:4])]/200*100*0.1
        for h in (5, 20, 60):
            outcomes.append({"code": "005930", "as_of_date": day.isoformat(), "observed_sessions": h,
                "status": "observed" if day+timedelta(days=h) <= last else "pending", "entry_date": (day+timedelta(days=1)).isoformat(),
                "exit_date": (day+timedelta(days=h)).isoformat(), "price_return_pct": target})
    learning = root / "learning/20261029_100000_12345678"
    save_source(learning, train_models(features, outcomes), features, outcomes)
    summary = json.loads((learning / "summary.json").read_text()); summary.update(schema_version=1, source_research=str(research), created_at="2026-10-19T10:00:00+09:00")
    (learning / "summary.json").write_text(json.dumps(summary)); (learning / "financial_audit.json").write_text(json.dumps({"005930": audit}))
    parent, _ = cross_check.run(text, root, learning, progress=lambda _: None)
    study, _, models = learn_finance.run(parent, learning, root, minimum_documents=1, progress=lambda _: None)
    meta = json.loads((study/"summary.json").read_text()); meta["created_at"] = "2026-10-19T16:00:00+09:00"
    (study/"summary.json").write_text(json.dumps(meta))
    assert all(c["disposition"] == "hold_for_new_validation" for c in models["models"].values())
    return study, audit


class PairedForwardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.source, self.events = held_study(self.root)
        self.now = datetime(2026, 10, 20, 8, 30, tzinfo=SEOUL)
        first = date(2026, 10, 19)-timedelta(days=69)
        self.bars = [DailyBar("005930", (first+timedelta(days=i)).isoformat(), 100+i, 115+i, 90+i, 105+i, 1000+i, None, False, False) for i in range(78)]
        self.prices = Prices(self.bars); self.dart = Mock()
        self.dart.companies.return_value = {"005930": {"corp_code": CORP}}
        self.dart.disclosures.return_value = []

    def cycle(self):
        with patch.object(shadow, "audit_reports", return_value=self.events):
            return shadow.run_cycle(self.root, self.source, source_factory=lambda: self.prices, dart_factory=lambda: self.dart,
                clock=lambda: self.now, progress=lambda _: None, sleep=lambda _: None)

    def forecast(self):
        experiment, summary = self.cycle()
        self.assertEqual(summary["new_predictions"], 2)
        predictions = observe.records(experiment, "predictions")
        key = next(k for k, p in predictions.items() if p["horizon"] == 5)
        return experiment, key, predictions[key]

    def test_pin_source_reproduction_no_fit_and_separate_journal(self):
        old = self.root / "forward/experiment/preserve"; old.parent.mkdir(parents=True); old.write_text("old numeric forecast")
        before = {p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch("finance_learning.fit_model", side_effect=AssertionError("no fitting")), patch("learning.fit", side_effect=AssertionError("no fitting")):
            experiment, summary = self.cycle()
        self.assertEqual(summary["new_predictions"], 2); self.assertEqual(summary["pending"], 2)
        self.assertEqual(experiment, self.root / "forward_finance/experiment")
        self.assertEqual(before, {p:p.read_bytes() for p in before})
        self.assertTrue(list((experiment / "sources").glob("*.zip")))
        self.assertFalse(summary["automatic_promotion"]); self.assertFalse(summary["existing_observer_modified"])

    def test_same_case_contains_two_predictions_and_deduplicates(self):
        experiment, _, p = self.forecast()
        self.assertIn("numeric_predicted_change_pct", p); self.assertIn("predicted_change_pct", p)
        self.assertTrue(all(p["feature"][k] is not None for k in EXTRA_FEATURES))
        before = {p:p.read_bytes() for p in (experiment/"predictions").iterdir()}
        _, summary = self.cycle()
        self.assertEqual(summary["new_predictions"], 0); self.assertEqual(summary["total_predictions"], 2)
        self.assertEqual(before, {p:p.read_bytes() for p in before})

    def test_paired_settlement_uses_same_future_prices_and_preserves_bytes(self):
        experiment, key, p = self.forecast()
        original = (experiment / "predictions" / (key+".json")).read_bytes()
        self.now = datetime(2026, 10, 27, 11, tzinfo=SEOUL)
        self.dart.companies.side_effect = AssertionError("settlement needs no DART")
        _, summary = self.cycle()
        result = observe.read_record(experiment/"results"/(key+".json"))
        actual = (self.bars[74].close/self.bars[70].open-1)*100
        self.assertAlmostEqual(result["actual_change_pct"], actual)
        self.assertEqual(result["entry_date"], "2026-10-20")  # Synthetic days, not exchange sessions.
        self.assertEqual(result["exit_date"], "2026-10-24")
        m = summary["paired_metrics"]["5"]
        self.assertAlmostEqual(m["numeric_mae_pp"], abs(actual-p["numeric_predicted_change_pct"]))
        self.assertAlmostEqual(m["finance_mae_pp"], abs(actual-p["predicted_change_pct"]))
        self.assertEqual(summary["completed"], 1); self.assertEqual(summary["pending"], 1)
        self.cycle(); self.assertEqual(original, (experiment/"predictions"/(key+".json")).read_bytes())

    def test_after_open_first_run_needs_no_keys_or_network(self):
        self.now = self.now.replace(hour=11)
        with patch.object(shadow, "stored_key", side_effect=AssertionError("no key")):
            target, summary = shadow.run_cycle(self.root, self.source, source_factory=lambda: (_ for _ in ()).throw(AssertionError("no API")), clock=lambda:self.now)
        self.assertEqual(summary["total_predictions"], 0); self.assertEqual(summary["errors"], [])
        self.assertIn("새 예측 대기", (target/"report.html").read_text())

    def test_original_study_can_change_but_frozen_model_cannot(self):
        experiment, _, _ = self.forecast()
        (self.source/"models.json").write_text("{}")
        self.cycle()
        (experiment/"models.json").write_text("{}")
        with self.assertRaises((CollectionError, KeyError)): self.cycle()

    def test_tampered_study_does_not_initialize(self):
        p = self.source/"models.json"; p.write_bytes(p.read_bytes()+b" ")
        with self.assertRaises(CollectionError): self.cycle()
        self.assertFalse((self.root/"forward_finance/experiment").exists())

    def test_rejected_candidate_is_not_revived_for_forward_observation(self):
        models = observe.read_json(self.source/"models.json")
        _, _, outcomes = shadow.load_source(self.source/"inputs")
        paired = observe.read_json(self.source/"feature_joins.json")
        for h, c in models["models"].items():
            candidate = {**copy.deepcopy(c["numeric"]), "features": list(MODEL_FEATURES+EXTRA_FEATURES)}
            for k, tail in (("median", [0,0]), ("mean", [0,0]), ("scale", [1,1]), ("coefficients", [0,0])):
                candidate[k] += tail
            parts, _ = partition_examples(paired,index_outcomes(outcomes),int(h),models["split"]["validation_start"],models["split"]["test_start"])
            candidate["test_metrics"] = shadow.score_model(candidate,parts["test"])
            c.update(numeric_and_finance=candidate,disposition="rejected",relative_error_reduction_pct=0)
        (self.source/"models.json").write_text(json.dumps(models))
        meta = observe.read_json(self.source/"summary.json")
        meta["output_sha256"]["models.json"] = hashlib.sha256((self.source/"models.json").read_bytes()).hexdigest()
        (self.source/"summary.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(CollectionError,"보류 후보가 없습니다"):
            shadow.load_candidate(self.source)

    def test_corrupt_latest_study_is_not_replaced_with_earlier_success(self):
        newest = self.root/"finance_learning/20991231_235959_abcdef12"; newest.mkdir()
        (newest/"summary.json").write_text("invalid JSON")
        self.assertEqual(shadow.latest_study(self.root),newest)
        with self.assertRaises(ValueError):
            shadow.run_cycle(self.root,clock=lambda:self.now)
        self.assertFalse((self.root/"forward_finance/experiment").exists())

    def test_missing_extra_input_defers_pair_instead_of_imputing(self):
        with patch.object(shadow, "live_financials", side_effect=CollectionError("전년 자료 없음")):
            _, summary = self.cycle()
        self.assertEqual(summary["total_predictions"], 0); self.assertTrue(summary["errors"])

    def test_history_revision_excludes_whole_pair(self):
        experiment, key, _ = self.forecast()
        self.prices.bars = self.bars[:]; self.prices.bars[69] = replace(self.bars[69], close=self.bars[69].close+1)
        self.now = datetime(2026, 10, 27, 11, tzinfo=SEOUL)
        _, summary = self.cycle()
        result = observe.read_record(experiment/"results"/(key+".json"))
        self.assertEqual(result["reason"], "price_history_revised")
        self.assertEqual(summary["paired_metrics"], {})

    def test_missing_commit_and_crossed_cutoff_do_not_count_as_validation(self):
        experiment, key, p = self.forecast()
        commit = observe.read_record(experiment/"commits"/(key+".json"))
        self.assertEqual(observe.settle(p,self.bars,None)["status"], "excluded")
        late = {**commit, "before_cutoff": False, "committed_at": "2026-10-20T09:00:01+09:00"}
        self.assertEqual(observe.settle(p,self.bars,late)["reason"], "commit_after_cutoff")

    def test_no_backfill_or_clock_rollback(self):
        experiment, _, p = self.forecast(); manifest = observe.read_record(experiment/"experiment.json"); models = observe.read_record(experiment/"models.json")
        late = lambda: self.now.replace(hour=9, minute=0)
        self.assertEqual(shadow.record_predictions(experiment,manifest,models,p["feature"],self.bars[:70],{},clock=late),0)
        with self.assertRaises(CollectionError): shadow.summary_for(experiment,manifest,now=self.now-timedelta(days=1))

    def test_private_errors_redacted_but_existing_results_continue(self):
        experiment, _, _ = self.forecast(); self.now = datetime(2026,10,27,8,30,tzinfo=SEOUL)
        self.dart.companies.side_effect = RuntimeError("PRIVATE_API_KEY_MUST_NOT_APPEAR")
        _, summary = self.cycle()
        self.assertEqual(summary["completed"],1); self.assertEqual(summary["new_predictions"],0)
        self.assertNotIn("PRIVATE_API_KEY", (experiment/"report.html").read_text())

    def test_same_day_ambiguity_no_older_financial_fallback(self):
        self.events.append({**self.events[-1], "receipt_no":"20260312000002"})
        _, summary = self.cycle()
        self.assertEqual(summary["total_predictions"],0); self.assertTrue(summary["errors"])

    def test_os_lock_prevents_parallel_writer_but_not_old_observer(self):
        with observe.journal_lock(self.root/"forward"):
            self.cycle()
        with observe.journal_lock(self.root/"forward_finance"), self.assertRaises(CollectionError): self.cycle()

    def test_html_escapes_messages_and_cli_resumes_without_refit(self):
        experiment, _, _ = self.forecast()
        summary = shadow.summary_for(experiment,observe.read_record(experiment/"experiment.json"),now=self.now,errors=["<script>bad</script>"])
        self.assertNotIn("<script>",shadow.render_report(summary)); self.assertIn("&lt;script&gt;",shadow.render_report(summary))
        self.now = self.now.replace(hour=11)
        resumed = self.cycle()
        with patch.object(shadow,"local_now",return_value=self.now), patch.object(shadow,"run_cycle",return_value=resumed), redirect_stdout(io.StringIO()):
            self.assertEqual(shadow.main(["--root",str(self.root),"--no-open"]),0)

    def test_write_crossing_cutoff_excludes_both_model_predictions(self):
        experiment, _, p = self.forecast()
        manifest = observe.read_record(experiment/"experiment.json"); models = observe.read_record(experiment/"models.json")
        feature = {**p["feature"], "as_of_date": "2026-10-20", "available_from": "2026-10-21T00:00:00+09:00"}
        before = datetime(2026,10,21,8,59,59,tzinfo=SEOUL); after = before.replace(hour=9,minute=0,second=1)
        times = iter([before,before,after,after])
        self.assertEqual(shadow.record_predictions(experiment,manifest,models,feature,self.bars[:71],{},clock=lambda:next(times)),1)
        pred = observe.read_record(experiment/"predictions/2026-10-20_005930_5.json")
        commit = observe.read_record(experiment/"commits/2026-10-20_005930_5.json")
        self.assertEqual(observe.settle(pred,self.bars,commit)["reason"],"commit_after_cutoff")
        self.assertFalse((experiment/"predictions/2026-10-20_005930_20.json").exists())


if __name__ == "__main__": unittest.main()
