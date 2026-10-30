"""A separate paired forward journal; never fits, promotes, or places orders."""
from __future__ import annotations

import argparse
import hashlib
import html
import math
import os
import shutil
import sys
import tempfile
import time
import webbrowser
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from check_facts import content, decoded, require
from dart_client import DartClient, stored_key
from evidence_table import build_records, latest_crosscheck, verify_source
from finance_learning import (ALPHAS, EXTRA_FEATURES, POLICY as FINANCE_POLICY, attach_financials,
    financial_records, predict_model, score_model)
from financial_crosscheck import cached_xbrl, parse_facts
from financial_history import audit_reports, financial_as_of
from learning import MODEL_FEATURES, index_outcomes, partition_examples
from lyupang import CollectionError, KiwoomSource, SEOUL, collect, parse_date, save_collection
from observe import (append_record, canonical, digest, journal_lock, local_now, next_check,
    prediction_window, read_json, read_record, records, settle, summarize, timestamp)
from research import safe_reason
from research_data import build_cases, load_snapshot
from review import diagnose, load_source

POLICY = "paired_finance_forward_journal_v1"


def latest_study(root):
    try: return latest_crosscheck(root / "finance_learning")
    except CollectionError:
        raise CollectionError("저장된 LEARN_FINANCE 연구 결과가 없습니다. 원래 자료 폴더를 확인해주세요.") from None


def validate_pair(comparison):
    require(comparison.get("status") == "trained")
    for kind, columns in (("numeric", MODEL_FEATURES), ("numeric_and_finance", MODEL_FEATURES+EXTRA_FEATURES)):
        m = comparison[kind]
        require(m["features"] == list(columns) and m["alpha"] in ALPHAS)
        for key in ("median", "mean", "scale", "coefficients"):
            require(len(m[key]) == len(columns) and all(type(v) in (int, float) and math.isfinite(v) for v in m[key]))
        require(all(v > 0 for v in m["scale"]) and type(m["intercept"]) in (int, float) and math.isfinite(m["intercept"]))
    require(comparison["numeric"]["intercept"] == comparison["numeric_and_finance"]["intercept"])


def load_candidate(source):
    """Reproduce source joins and held scores without calling any fitting function."""
    summary = decoded(content(source / "summary.json"))
    require(summary["schema_version"] == 1 and summary["policy"] == FINANCE_POLICY)
    require(summary["research_training_executed"] is True and summary["source_complete"] is True)
    require(all(summary.get(k) is False for k in ("production_input_approved", "automatic_promotion", "live_trading_allowed", "existing_model_modified", "fresh_unseen_evaluation")))
    expected_outputs = {"financial_records.json", "coverage.json", "as_of.json", "feature_joins.json", "models.json", "historical_comparisons.csv", "financial_records.csv", "feature_joins.csv"}
    require(set(summary["output_sha256"]) == expected_outputs)
    for name, sha in summary["output_sha256"].items(): require(hashlib.sha256(content(source / name)).hexdigest() == sha)
    expected_inputs = {"summary.json", "financial_audit.json", "models.json", "features_with_financials.csv", "future_outcomes.csv",
        "crosscheck_summary.json", "crosscheck_documents.json", "crosscheck_comparisons.jsonl", "crosscheck_baseline_comparisons.jsonl"}
    require(set(summary["input_sha256"]) == expected_inputs)
    for name, sha in summary["input_sha256"].items(): require(hashlib.sha256(content(source / "inputs" / name)).hexdigest() == sha)
    parent = Path(summary["source_crosscheck"])
    require(hashlib.sha256(content(parent / "summary.json")).hexdigest() == summary["source_crosscheck_sha256"])
    _, original, comparisons, coverage, _, finance, _ = verify_source(parent)
    require(content(parent / "financial_summary.json") == content(source / "inputs/summary.json")
        and content(parent / "financial_audit.json") == content(source / "inputs/financial_audit.json"))
    for name in ("summary.json", "documents.json", "comparisons.jsonl", "baseline_comparisons.jsonl"):
        require(content(parent / name) == content(source / "inputs" / ("crosscheck_"+name)))
    direct = financial_records(finance, coverage, build_records(comparisons, coverage), comparisons)
    require(direct == decoded(content(source / "financial_records.json")) and coverage == decoded(content(source / "coverage.json")))
    complete = {s["code"]: s["read_documents"] == s["target_documents"] > 0 and s["selection_status"] == "available" for s in original["stocks"]}
    old, features, outcomes = load_source(source / "inputs")
    diagnosis, _ = diagnose(old, features, outcomes)
    paired, excluded = attach_financials(features, direct, coverage, complete)
    require(paired == decoded(content(source / "feature_joins.json")) and excluded == summary["exclusions"])
    require(len(paired) == summary["paired_rows"] and len(features) == summary["source_rows"])
    models = decoded(content(source / "models.json"))
    require(models["policy"] == FINANCE_POLICY and models["split"] == old["split"] and models["research_only"] is True
        and models["automatic_promotion"] is False and models["live_trading_allowed"] is False and models["fresh_unseen_evidence"] is False)
    eligible, indexed = [], index_outcomes(outcomes)
    for horizon, comparison in models["models"].items():
        require(horizon in {"5", "20"})
        if comparison["status"] != "trained": continue
        require(diagnosis["models"][horizon]["disposition"] == "hold")
        validate_pair(comparison)
        parts, _ = partition_examples(paired, indexed, int(horizon), old["split"]["validation_start"], old["split"]["test_start"])
        counts = {k: len(v) for k, v in parts.items()}
        require(counts == comparison["counts"] and all(counts.values()))
        for kind in ("numeric", "numeric_and_finance"):
            m = comparison[kind]
            require(m["training_rows"] == counts["train"]+counts["validation"]
                and m["training_label_end"] == max(o["exit_date"] for _, o in parts["train"]+parts["validation"])
                and m["training_label_end"] < old["split"]["test_start"])
            computed = score_model(m, parts["test"])
            require(computed.keys() == m["test_metrics"].keys() and all(math.isclose(v, m["test_metrics"][k], rel_tol=1e-8, abs_tol=1e-8) for k, v in computed.items()))
        base, extra = (comparison[k]["test_metrics"] for k in ("numeric", "numeric_and_finance"))
        held = extra["mae_pp"] < base["mae_pp"] and extra["mae_pp"] < base["constant_mae_pp"]
        require(comparison["disposition"] == ("hold_for_new_validation" if held else "rejected"))
        if held: eligible.append(horizon)
    require(eligible, "새 예측을 관찰할 보류 후보가 없습니다. 탈락 모델은 자동으로 되살리지 않습니다.")
    return summary, models, sorted(eligible, key=int), sorted({f["code"] for f in paired}), max(f["as_of_date"] for f in features)


def initialize(root, source=None, *, clock=local_now):
    experiment = root / "experiment"
    if experiment.exists():
        manifest = read_record(experiment / "experiment.json")
        models = read_record(experiment / "models.json")
        require(manifest["policy"] == POLICY and digest(models) == manifest["model_sha256"], "고정한 재무 관찰 모델이 달라졌습니다. 기존 기록을 보존해주세요.")
        require(manifest["features"] == list(MODEL_FEATURES+EXTRA_FEATURES) and set(manifest["horizons"]) <= {"5", "20"} and manifest["horizons"])
        require(manifest["automatic_promotion"] is False and manifest["live_trading_allowed"] is False)
        for h in manifest["horizons"]: validate_pair(models["models"][h])
        return experiment, manifest, models
    require(source is not None, "먼저 저장한 재무 추가 연구 결과가 필요합니다.")
    summary, models, horizons, codes, last_day = load_candidate(source)
    created = clock().astimezone(SEOUL)
    require(timestamp(summary["created_at"]) <= created, "PC 시각이 연구 결과 생성 시각보다 이전입니다.")
    manifest = {"schema_version": 1, "policy": POLICY, "created_at": created.isoformat(), "source_study": str(source.resolve()),
        "source_summary_sha256": hashlib.sha256(content(source / "summary.json")).hexdigest(), "model_sha256": digest(models),
        "horizons": horizons, "excluded_horizons": [h for h in ("5", "20", "60") if h not in horizons], "codes": codes,
        "features": list(MODEL_FEATURES+EXTRA_FEATURES), "source_last_as_of": last_day,
        "target": "next_observed_session_open_to_horizon_close_before_costs", "automatic_promotion": False, "live_trading_allowed": False,
        "trusted_timestamp": False, "exchange_calendar_verified": False, "source_test_already_consumed": True}
    staging = Path(tempfile.mkdtemp(prefix=".initializing-", dir=root))
    try:
        append_record(staging / "models.json", models); append_record(staging / "experiment.json", manifest)
        (staging / "source_summary.json").write_bytes(content(source / "summary.json"))
        os.replace(staging, experiment)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return experiment, manifest, models


def live_financials(root, experiment, code, corp, feature, events):
    receipt = feature["financial_receipt_no"]
    chosen = [e for e in events if e["receipt_no"] == receipt]
    require(feature["financial_available"] is True and len(chosen) == 1 and chosen[0]["status"] == "verified",
            "추가 재무 입력의 최신 접수번호를 확인하지 못했습니다.")
    event = chosen[0]
    raw, metadata = cached_xbrl(root / "financial_documents", code, receipt, event["sha256"])
    facts, identity = parse_facts(raw, parse_date(event["period_end"]), event["metrics"], corp, code)
    doc = {"status": "read", "facts": facts, "source_sha256": event["sha256"], **identity}
    coverage = [{"code": code, "company_name": code, "receipt_no": receipt, "receipt_date": event["receipt_date"],
        "period_end": event["period_end"], "available_from": feature["financial_available_from"], "financial_status": "read"}]
    direct = financial_records({(code, receipt): doc}, coverage, [], [])
    joined, _ = attach_financials([feature], direct, coverage, {code: True})
    require(len(joined) == 1 and all(joined[0][k] is not None and math.isfinite(joined[0][k]) for k in EXTRA_FEATURES),
            "영업이익률 또는 같은 제출본의 전년 자료가 없어 이번 예측을 보류합니다.")
    folder = experiment / "sources"; folder.mkdir(exist_ok=True)
    path = folder / (code+"_"+receipt+"_"+event["sha256"]+".zip")
    if path.exists(): require(content(path) == raw)
    else:
        with path.open("xb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return joined[0], {"original_path": str(path.resolve()), "sha256": event["sha256"], "identity": identity,
        "facts": direct, "provider": metadata["provider"], "endpoint": metadata["endpoint"]}


def record_predictions(experiment, manifest, models, feature, bars, evidence, *, clock=local_now):
    now = clock().astimezone(SEOUL)
    if not prediction_window(now): return 0
    require(feature["code"] in manifest["codes"] and feature["as_of_date"] >= manifest["source_last_as_of"])
    require(parse_date(feature["as_of_date"]) < now.date() and timestamp(feature["available_from"]) <= now
        and timestamp(manifest["created_at"]) <= now)
    require(not feature.get("quality_issue") and feature.get("financial_available") is True and feature.get("dart_available") is True)
    require(feature["financial_receipt_date"] < feature["as_of_date"] and timestamp(feature["financial_available_from"]) <= now)
    require(feature["financial_available_from"] == (parse_date(feature["financial_receipt_date"])+timedelta(days=1)).isoformat()+"T00:00:00+09:00")
    require(feature["source_financial_available_from"] == feature["financial_available_from"]
        and feature["source_financial_document_id"] == feature["code"]+"_"+feature["financial_receipt_no"]
        and feature["source_financial_scope"] == feature["financial_scope"])
    require(all(feature[k] is not None and type(feature[k]) in (int, float) and math.isfinite(feature[k]) for k in EXTRA_FEATURES))
    require(len(bars) >= 61 and bars[-1].date == feature["as_of_date"] and bars[-1].code == feature["code"])
    count = 0
    for h in manifest["horizons"]:
        pair = models["models"][h]
        require(all(pair[k]["training_label_end"] < feature["as_of_date"] for k in ("numeric", "numeric_and_finance")))
        path = experiment / "predictions" / f"{feature['as_of_date']}_{feature['code']}_{h}.json"
        if path.exists(): read_record(path); continue
        numeric, extra = (float(predict_model(pair[k], [(feature, {})])[0]) for k in ("numeric", "numeric_and_finance"))
        recorded = clock().astimezone(SEOUL)
        if not prediction_window(recorded): break
        require(recorded >= now, "예측 저장 중 PC 시각이 이전으로 바뀌었습니다.")
        payload = {"schema_version": 1, "recorded_at": recorded.isoformat(), "code": feature["code"], "as_of_date": feature["as_of_date"],
            "horizon": int(h), "model_sha256": manifest["model_sha256"], "predicted_change_pct": extra,
            "numeric_predicted_change_pct": numeric, "baseline_change_pct": pair["numeric"]["intercept"],
            "feature": feature, "evidence": evidence, "anchor_bars": [asdict(b) for b in bars[-61:]], "live_trading_allowed": False}
        append_record(path, payload)
        committed = clock().astimezone(SEOUL)
        append_record(experiment / "commits" / path.name, {"prediction_sha256": digest(payload),
            "committed_at": committed.isoformat(), "before_cutoff": prediction_window(committed)})
        count += 1
    return count


def summary_for(experiment, manifest, *, now, new=0, errors=None, notices=None):
    require(timestamp(manifest["created_at"]) <= now, "PC 시각이 모델 고정 시각보다 이전입니다.")
    summary = summarize(experiment, manifest, now=now, new=new, errors=errors, notices=notices)
    predictions, results = records(experiment, "predictions"), records(experiment, "results")
    models = read_record(experiment / "models.json")
    require(digest(models) == manifest["model_sha256"])
    for key, p in predictions.items():
        require(str(p["horizon"]) in manifest["horizons"] and p["code"] in manifest["codes"]
            and key == f"{p['as_of_date']}_{p['code']}_{p['horizon']}")
        pair = models["models"][str(p["horizon"])]
        for kind, field in (("numeric", "numeric_predicted_change_pct"), ("numeric_and_finance", "predicted_change_pct")):
            require(math.isclose(p[field], float(predict_model(pair[kind], [(p["feature"], {})])[0]), rel_tol=1e-12, abs_tol=1e-12))
        if key in results and results[key]["status"] == "observed":
            require(results[key]["numeric_predicted_change_pct"] == p["numeric_predicted_change_pct"])
    paired = {}
    for h in manifest["horizons"]:
        cases = [r for r in results.values() if r["status"] == "observed" and str(r["horizon"]) == h]
        if not cases: continue
        extra = sum(r["absolute_error_pp"] for r in cases)/len(cases)
        numeric = sum(abs(r["actual_change_pct"]-r["numeric_predicted_change_pct"]) for r in cases)/len(cases)
        constant = sum(r["baseline_absolute_error_pp"] for r in cases)/len(cases)
        paired[h] = {"rows": len(cases), "unique_dates": len({r['as_of_date'] for r in cases}), "finance_mae_pp": extra,
            "numeric_mae_pp": numeric, "constant_mae_pp": constant, "relative_error_reduction_pct": (numeric-extra)/numeric*100 if numeric else None,
            "direction_accuracy": sum((r["actual_change_pct"]>0) == (r["predicted_change_pct"]>0) for r in cases)/len(cases),
            "actual_up_fraction": sum(r["actual_change_pct"]>0 for r in cases)/len(cases)}
    return {**summary, "policy": POLICY, "paired_metrics": paired, "source_study": manifest["source_study"],
        "model_sha256": manifest["model_sha256"], "existing_observer_modified": False, "model_retrained": False}


def render_report(summary):
    esc = lambda v: html.escape(str(v))
    rows = []
    for h in summary["horizons"]:
        m = summary["paired_metrics"].get(h)
        if not m: rows.append(f"<tr><td>{h}일봉</td><td colspan='7'>결과 대기</td></tr>"); continue
        gain = m["relative_error_reduction_pct"]
        rows.append(f"<tr><td>{h}일봉</td><td>{m['rows']} / {m['unique_dates']}</td><td>{m['numeric_mae_pp']:.3f}</td>"
            f"<td>{m['finance_mae_pp']:.3f}</td><td>{m['constant_mae_pp']:.3f}</td><td>{'—' if gain is None else f'{gain:.2f}%'}</td>"
            f"<td>{m['direction_accuracy']*100:.1f}% / {m['actual_up_fraction']*100:.1f}%</td><td>관찰 중·자동 채택 없음</td></tr>")
    messages = "".join(f"<li>{esc(s)}</li>" for s in summary["notices"]+summary["errors"])
    reason_names = {"commit_missing": "예측 저장 완료 기록 없음", "commit_after_cutoff": "오전 9시 이후 저장 완료",
        "anchor_history_missing": "기존 가격 이력 누락", "price_history_revised": "기존 가격 이력 변경",
        "recorded_after_entry": "관찰 시작 뒤 저장된 예측", "future_price_quality": "이후 가격·거래량 품질 문제"}
    exclusions = " · ".join(f"{reason_names.get(k,k)}: {v}쌍" for k,v in summary["exclusion_reasons"].items()) or "없음"
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>재무 추가 모델의 새 예측 관찰</title><style>body{{font-family:system-ui;line-height:1.8;max-width:1200px;margin:30px auto;padding:20px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:18px;background:#fff2d5}}</style>
<h1>재무 추가 모델의 새 예측 관찰</h1><p class="note">같은 연구에서 고정한 수치 모델과 재무 추가 모델을 같은 날·같은 입력·같은 이후 가격으로 비교합니다. 기존 OBSERVE는 별도로 유지합니다. 실제 주문·재학습·자동 채택은 실행하지 않습니다.</p>
<p>모델 고정: {esc(summary['experiment_created_at'])}<br>마지막 확인: {esc(summary['updated_at'])}</p>
<p>관찰 구간: {esc(', '.join(summary['horizons']))}일봉 · 제외한 구간: {esc(', '.join(summary['excluded_horizons']) or '없음')}.</p>
<h2>기록 상태</h2><table><tr><th>이번 새 예측 쌍</th><th>누적 예측 쌍</th><th>결과 대기</th><th>결과 확인</th><th>검증 제외</th></tr><tr><td>{summary['new_predictions']}</td><td>{summary['total_predictions']}</td><td>{summary['pending']}</td><td>{summary['completed']}</td><td>{summary['excluded']}</td></tr></table>
<p>한 쌍은 같은 종목·기준일·5일 또는 20일 구간에 대한 두 모델의 예측입니다. 모델 두 개를 별도 독립 사례로 세지 않습니다.</p><ul>{messages}</ul><p>검증 제외 사유: {esc(exclusions)}</p>
<h2>같은 새 사례에서의 비교</h2><p>평균 오차(%p)는 작을수록 좋습니다. 개선율은 예측 오차의 감소이며 투자 수익률이 아닙니다. 평균 기준은 같은 학습 사례의 평균 변화만 답하는 방식입니다.</p>
<table><tr><th>구간</th><th>예측 쌍 / 날짜</th><th>수치 모델 오차</th><th>재무 추가 오차</th><th>평균 기준 오차</th><th>오차 개선율</th><th>재무 모델 상승 일치 / 실제 상승 비율</th><th>상태</th></tr>{''.join(rows)}</table>
<h2>실행 시간</h2><p>이 검은 실행 창과 PC를 켜두면 한국 시간 매일 오전 8시 30분에 확인합니다. 첫 실행 때도 한 번 확인합니다. 오전 9시 이후에는 기존 결과만 확인하며 새 예측을 만들지 않습니다. 꺼져 있거나 잠들어 놓친 날은 뒤늦게 채우지 않습니다. 브라우저만 열어둬서는 실행되지 않습니다.</p>
<p>처음 검증한 연구 모델을 복사해 고정합니다. 이후 연구 결과가 추가되거나 원래 폴더가 바뀌어도 자동 교체·재학습하지 않습니다. 기존 OBSERVE 모델과 예측은 이 기록과 합치지 않습니다.</p>
<h2>검증의 범위</h2><p>기록·저장 완료 이후 시작한 다음 일봉 시가에서 5·20개 일봉의 마지막 종가까지 비교합니다. 두 예측을 함께 저장하고 같은 가격 결과에서 함께 평가합니다. 공개 시점과 명시적 연결/별도 범위를 맞춘 영업이익률·동일 제출본 비교 전기 항목을 사용합니다. 두 추가 입력 중 하나라도 확인하지 못하면 그날 예측 쌍을 보류합니다.</p>
<p>원래 가격 이력이 변하거나 이후 가격·거래량 품질 문제가 있으면 쌍 전체를 제외합니다. 새 기간의 기록이 쌓여도 자동 승인하지 않습니다. 겹치는 미래 구간·같은 날짜의 종목은 독립 표본이 아니며, 소수 기업·반복 연간 원문으로 시장 전체 성능이나 통계적 유의성을 입증하지 않습니다.</p>
<p>수수료·세금·체결·모의매매·거래소 달력·특별 개장·거래정지·과거 당시 수정주가·독립된 원문 정답·외부 시각 인증은 검증하지 못합니다. 로컬 해시는 외부 위변조 방지 증명이 아닙니다. PC 시각이 정확해야 합니다. 주가 상승 원인·사업/뉴스의 정확한 의미를 학습한 결과가 아닙니다.</p></html>"""


def write_report(experiment, summary):
    for name, raw in (("summary.json", canonical(summary)), ("report.html", render_report(summary).encode())):
        path = experiment / ("."+name+".tmp"); path.write_bytes(raw); os.replace(path, experiment / name)


def run_cycle(root, source_study=None, *, source_factory=KiwoomSource, dart_factory=None, clock=local_now, progress=print, sleep=time.sleep):
    forward = root / "forward_finance"
    with journal_lock(forward):
        experiment, manifest, models = initialize(forward, source_study if source_study else (None if (forward/"experiment").exists() else latest_study(root)), clock=clock)
        now = clock().astimezone(SEOUL)
        # Validate the journal before adding new records.
        summary_for(experiment, manifest, now=now)
        prior, results = records(experiment, "predictions"), records(experiment, "results")
        pending = {k:p for k,p in prior.items() if k not in results}
        errors, notices, new = [], [], 0
        wants_new = prediction_window(now)
        if not wants_new: notices.append("새 예측 대기 — 오전 9시 이후입니다. 다음 자동 확인은 한국 시간 오전 8시 30분입니다.")
        source, dart, companies = None, None, {}
        if wants_new:
            try:
                key = None if dart_factory else stored_key()
                require(dart_factory or key, "기존 DART 인증정보를 찾지 못했습니다.")
                dart = dart_factory() if dart_factory else DartClient(key)
                companies = dart.companies(manifest["codes"])
            except Exception as exc:
                errors.append("재무·공시 연결 확인 필요: "+safe_reason(exc)); wants_new = False
        for code in manifest["codes"]:
            items = {k:p for k,p in pending.items() if p["code"] == code}
            if not wants_new and not items: continue
            try:
                progress(f"종목 {code} · 재무 추가 모델의 새 예측과 기록 확인 중...")
                end = now.date()-timedelta(days=1); start = end-timedelta(days=400)
                if items: start = min(start, min(parse_date(p["anchor_bars"][0]["date"]) for p in items.values()))
                if source is None: source = source_factory()
                collection = collect(source, code, start, end, sleep=sleep)
                snapshot = save_collection(collection, root/"data", now=clock())
                metadata, bars = load_snapshot(snapshot)
                evidence = {"price_snapshot": str(snapshot.resolve()), "price_sha256": metadata["csv_sha256"]}
                for key, p in items.items():
                    commit_path = experiment/"commits"/(key+".json")
                    outcome = settle(p, bars, read_record(commit_path) if commit_path.exists() else None)
                    if outcome is not None:
                        outcome.update(prediction_sha256=digest(p), checked_at=clock().isoformat(), evidence=evidence)
                        if outcome["status"] == "observed": outcome["numeric_predicted_change_pct"] = p["numeric_predicted_change_pct"]
                        append_record(experiment/"results"/(key+".json"), outcome)
                if wants_new and prediction_window(clock()):
                    latest = bars[-1].date
                    if all((experiment/"predictions"/f"{latest}_{code}_{h}.json").exists() for h in manifest["horizons"]): continue
                    company = companies.get(code); require(company, "DART 기업번호를 확인하지 못했습니다.")
                    as_of = parse_date(latest)
                    filings = dart.disclosures(company["corp_code"], as_of-timedelta(days=7), as_of)
                    events = audit_reports(code, company["corp_code"], as_of, end, dart, root/"financial_documents", progress=progress)
                    features, _ = build_cases(bars, filings); require(features, "61개 일봉으로 새 입력을 만들지 못했습니다.")
                    feature = {**features[-1], **financial_as_of(events, as_of)}
                    feature, financial = live_financials(root, experiment, code, company["corp_code"], feature, events)
                    new += record_predictions(experiment, manifest, models, feature, bars,
                        {**evidence, "filings": filings, "financial_audit": events, "financial_original": financial}, clock=clock)
            except Exception as exc: errors.append(code+": "+safe_reason(exc))
            finally: sleep(0.6)
        summary = summary_for(experiment, manifest, now=clock(), new=new, errors=errors, notices=notices)
        write_report(experiment, summary)
        return experiment, summary


def main(argv=None):
    default = Path(os.environ["LOCALAPPDATA"])/"Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="재무 추가 후보와 같은 연구의 수치 모델을 새 가격에서 함께 관찰")
    parser.add_argument("--root", type=Path, default=default)
    parser.add_argument("--input", type=Path, help="처음 고정할 LEARN_FINANCE 결과; 이후에는 고정 모델 유지")
    parser.add_argument("--profile", default="kiwoom-real")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv); opened = False
    try:
        while True:
            target, summary = run_cycle(args.root, args.input, source_factory=lambda: KiwoomSource(args.profile))
            print(f"\n새 예측 쌍 {summary['new_predictions']}개 · 대기 {summary['pending']}개 · 결과 확인 {summary['completed']}개 · 제외 {summary['excluded']}개")
            for message in summary["notices"]+summary["errors"]: print(message)
            print("기존 OBSERVE 유지 · 자동 채택 0개 · 실제 주문 없음")
            print(f"재무 관찰 보고서: {target/'report.html'}")
            if not opened and sys.platform == "win32" and not args.no_open:
                webbrowser.open((target/"report.html").resolve().as_uri()); opened = True
            if not args.watch: return 2 if summary["errors"] else 0
            scheduled = next_check(local_now(), bool(summary["errors"]))
            print(f"다음 자동 확인: {scheduled:%Y-%m-%d %H:%M} 한국 시간 · 이 창과 PC를 켜두세요. 종료: Ctrl+C", flush=True)
            while local_now() < scheduled: time.sleep(min(30, max(0, (scheduled-local_now()).total_seconds())))
    except (KeyboardInterrupt, EOFError):
        print("재무 관찰을 종료했습니다. 다음 실행에서 같은 기록을 이어갑니다."); return 130
    except CollectionError as exc: print(f"재무 관찰 확인 필요: {exc}", file=sys.stderr)
    except Exception: print("재무 관찰 기록을 확인하지 못했습니다. 마지막 메시지를 알려주세요. 기존 기록은 덮어쓰지 않습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__": sys.exit(main())
