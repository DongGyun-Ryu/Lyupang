"""Use stored financial originals in an isolated research comparison."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import sys
import tempfile
import uuid
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path

from check_facts import content, require
from cross_check import financial_html
from evidence_table import METRICS, SCOPES, build_records, latest_crosscheck, money, snapshot_as_of, verify_source, write_csv
from finance_learning import EXTRA_FEATURES, FEATURE_NAMES, POLICY, attach_financials, financial_records, predict_model, train_comparison
from learn_text import select_learning
from learning import index_outcomes, partition_examples
from lyupang import CollectionError, SEOUL, parse_date
from review import diagnose, load_source


def render_report(summary, records, snapshots, models, *, show_records=False):
    esc = lambda v: html.escape(str(v))
    rows, repeated = [], []
    for horizon, comparison in models["models"].items():
        if comparison["status"] != "trained":
            rows.append(f"<tr><td>{esc(horizon)}일봉</td><td colspan='7'>학습 대기: {esc(comparison['reason'])}</td></tr>")
            continue
        base, extra = (comparison[k]["test_metrics"] for k in ("numeric", "numeric_and_finance"))
        gain = comparison["relative_error_reduction_pct"]
        verdict = "탈락 — 수치 모델 또는 평균 기준보다 오차가 크거나 같음" if comparison["disposition"] == "rejected" else "보류 — 새 자료 검증 필요"
        rows.append(f"<tr><td>{horizon}일봉</td><td>{extra['rows']} / {extra['unique_dates']}</td><td>{base['mae_pp']:.3f}</td>"
            f"<td>{base['constant_mae_pp']:.3f}</td><td>{extra['mae_pp']:.3f}</td><td>{'—' if gain is None else f'{gain:.2f}%'}</td>"
            f"<td>{extra['direction_accuracy']*100:.1f}% / {extra['actual_up_fraction']*100:.1f}%</td><td>{verdict}</td></tr>")
        repeated.append(f"<p>{horizon}일봉: 최종 학습 원문 {comparison['training_documents']}건 · 비교 원문 {comparison['test_documents']}건 · "
            f"학습에서 읽지 않은 비교 원문 {comparison['unseen_test_documents']}건 · 학습에서도 읽은 원문을 쓰는 비교 사례 {comparison['rows_using_seen_documents']}개.</p>")
        repeated.append("<p>추가 입력의 학습 / 비교 구간 값 있음: " + " · ".join(
            f"{FEATURE_NAMES[k]} {comparison['numeric_and_finance']['input_observed_counts'][k]} / {comparison['test_observed_counts'][k]}사례"
            for k in EXTRA_FEATURES) + "</p>")
    current = [r for r in records if r["period_end"] == r["report_period_end"]]
    coverage = []
    for stock in summary["codes"]:
        selected = [r for r in current if r["code"] == stock]
        coverage.append(f"<tr><td>{stock}</td><td>{len({r['receipt_no'] for r in selected})}</td>"
            + "".join(f"<td>{sum(r['status']=='available' and r['metric']==m for r in selected)}</td>" for m in METRICS)
            + f"<td>{sum(r['status']!='available' for r in selected)}</td></tr>")
    latest = "".join(f"<tr><td>{r['code']} · {SCOPES[r['scope']]}</td><td>{esc(r['selected_receipt_no'] or '없음')}</td>"
        + "".join(f"<td>{esc(money(r['values'][m]))}</td>" for m in METRICS)+f"<td>{esc(r['state'])}</td></tr>" for r in snapshots)
    state_names = {"available": "원문 표기값 있음", "source_unavailable": "재무 원문 미확인", "fact_missing": "해당 항목 없음",
        "fact_conflict": "원문 값 충돌", "precision_unresolved": "표시 정밀도 미확정", "statement_value_mismatch": "설명 금액과 다름·보류"}
    details = "".join(f"<tr><td>{r['code']}</td><td>{r['receipt_no']}<br>{esc(r['available_from'][:10])}</td>"
        f"<td>{r['period_start']} ~ {r['period_end']}<br>{SCOPES[r['scope']]}</td><td>{METRICS[r['metric']]}</td>"
        f"<td>{esc(money(r['value_krw']))}</td><td>{state_names[r['status']]}</td>"
        f"<td>{'설명 대조 근거 있음' if r['narrative_record_id'] else '재무 원문에서 직접 읽음·설명 대조 없음'}<br>"
        + " ".join(f"<a href='financial_sources/{r['code']}_{r['receipt_no']}/facts.html#{esc(f['id'])}'>{esc(f['concept'].rsplit('}',1)[-1])}</a>" for f in r['facts'])
        + "</td></tr>" for r in records) if show_records else ""
    detail_section = ("<h2>보관한 재무 항목과 출처</h2><table><tr><th>종목</th><th>접수번호·사용 가능일</th><th>기간·기준</th><th>항목</th><th>원문 표기값</th><th>상태</th><th>근거</th></tr>"+details+"</table>") if show_records else ""
    status_counts = " · ".join(f"{state_names[k]} {v}개" for k, v in summary["record_status_counts"].items())
    status = {"trained": "재무 추가 모델의 연구용 학습·과거 비교 완료", "partial": "일부 구간 연구용 학습 완료", "blocked": "연구용 학습 조건 미충족"}[models["status"]]
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>근거 재무를 추가한 연구 모델</title><style>body{{font-family:system-ui;line-height:1.8;max-width:1300px;margin:30px auto;padding:20px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left;font-size:14px}}.note{{padding:18px;background:#fff2d5}}</style>
<h1>근거 재무를 추가한 연구 모델</h1><p>{status}</p><p>{esc(models.get('reason',''))}</p>
<p class="note">이미 확인한 과거 자료의 개발용 비교입니다. 연구 모델만 별도 저장했습니다. 기존 관찰 모델·예측을 바꾸거나 실제 주문·모델 자동 채택을 실행하지 않았습니다.</p>
<h2>자료 보완</h2><p>설명 대조로 모았던 {summary['narrative_records']}항목은 그대로 보존했습니다. 같은 접수번호의 재무 원문에서 직접 읽을 수 있는 당기 연간 항목은 {summary['available_current_records']}개입니다. 설명 대조가 없는 항목을 설명의 정답이나 새로 검증한 문장으로 바꾸지 않습니다. 두 건수는 서로 다른 수집 범위이며 정확도 개선율이 아닙니다.</p>
<table><tr><th>종목</th><th>대상 접수번호</th><th>매출·영업수익 있음</th><th>영업이익 있음</th><th>당기순이익 있음</th><th>확인 필요</th></tr>{''.join(coverage)}</table>
<p>연결·별도·정정본을 각각 셉니다. 항목 수는 독립 원문·독립 표본 수가 아닙니다. 당기·비교 전기 원문 상태: {esc(status_counts)}.</p>
<h2>같은 사례에서의 비교</h2><p>원래 {summary['source_rows']}사례 중 공개 시점·접수번호·명시적 연결/별도 기준을 맞춘 {summary['paired_rows']}사례를 양쪽 모델에 동일하게 사용했습니다. 평균 오차(%p)는 작을수록 좋습니다. 개선율은 거래 수익률이 아닙니다.</p>
<table><tr><th>관찰 구간</th><th>사례 / 날짜</th><th>수치 모델 오차</th><th>과거 평균 기준 오차</th><th>재무 추가 오차</th><th>수치 모델 대비 개선율</th><th>재무 모델 상승 일치 / 실제 상승 비율</th><th>판정</th></tr>{''.join(rows)}</table>
<p>양쪽 모두 같은 기간과 사례에서 다시 학습하므로 예전 점수와 다를 수 있습니다. 기준보다 나쁘면 탈락, 좋아도 새 자료 검증 전까지 보류합니다. 항상 상승이라고 답한 일치율은 실제 상승 비율과 같습니다.</p>
<h2>추가한 입력과 연결 범위</h2><p>기존 가격·거래량·공시·재무 비율 8개에 영업이익률(영업이익 ÷ 매출 × 100)과 영업이익률의 전년 대비 변화(%포인트)를 더합니다. 순이익률·매출 증가율은 기존 입력에도 있어 새 입력으로 중복 추가하지 않습니다. 부호가 명시된 원문 손실은 음수로 유지합니다.</p>
<p>전년 값은 선택한 동일 접수번호 원문에 실린 비교 전기 금액으로만 계산합니다. 과거의 최초 보고 금액과 같다는 뜻은 아니며, 현재 제출본이 공개된 다음 날부터만 사용합니다. 연결·별도, 기간·접수번호를 섞지 않습니다. 문장의 뜻·주가 상승 원인·뉴스는 학습하지 않았습니다.</p>
<p>영업이익률 값 있음: {summary['margin_rows']}사례 · 전년 대비 변화 값 있음: {summary['margin_change_rows']}사례. 미연결 사유: {esc(summary['exclusions'])}. 보완한 금액을 찾지 못하면 이전 제출본·다른 기준으로 대신하지 않습니다.</p>
<p>추가 입력의 일부 누락은 학습 구간의 중앙값으로만 보정하며 값 유무를 별도로 기록합니다. 학습 전체에서 없는 열은 기여하지 않습니다. 두 추가 열 모두 학습에서 없으면 학습을 보류합니다. 수집 범위 미완료 종목·같은 날 여러 최신 제출본은 보류합니다.</p>
<h2>원문의 반복 사용</h2>{''.join(repeated)}<p>연간 원문 하나가 여러 거래일에 반복됩니다. 같은 종목·날짜와 겹친 미래 구간은 독립 표본이 아니며 새 기업·새 원문·시장 전체의 성능이나 통계적 유의성을 입증하지 않습니다.</p>
<h2>재무 원문으로 본 기준일 자료 · {summary['reference_date']}</h2><table><tr><th>종목·범위</th><th>선택 접수번호</th><th>매출·영업수익</th><th>영업이익</th><th>당기순이익</th><th>상태</th></tr>{latest}</table>
<p>빈칸은 0이 아닙니다. 당시 공개된 최신 회계기간·최신 접수일을 먼저 고르고 항목이 없으면 비웁니다. 이후 옛 회계기간 정정이 새 기간을 대신하지 않습니다. 자료 기준일 뒤 새 공시의 완전성은 확인하지 못합니다.</p>
{detail_section}<p><a href='records.html'>전체 금액과 원문 근거 보기</a> · <a href='financial_records.csv' download>재무 항목 CSV</a> · <a href='feature_joins.csv' download>공개 시점별 입력 연결 CSV</a> · <a href='historical_comparisons.csv' download>과거 비교 CSV</a></p>
<h2>검증의 범위</h2><p>원래 대조 입력·본문·ZIP·기업 확인 기록·재무 항목을 재현한 뒤, 같은 학습 자료의 해시와 기존 평가 결과를 확인했습니다. 같은 회사 제출 자료의 개발용 연구이며 독립된 정답 검증이 아닙니다. 기간 경계를 넘는 미래 결과를 앞 기간 학습에서 제외하고, 보정·표준화·모델 강도 선택에 마지막 비교 기간을 쓰지 않았습니다. 이 기간은 이미 본 자료이므로 새 미사용 검증으로 주장하지 않습니다.</p>
<p>새 결과는 finance_learning 별도 폴더에 저장했습니다. 기존 모델·예측·원문·설명 대조 결과는 유지합니다. 실제 주문·자동 채택·비용과 세금·체결·모의매매·과거 당시 수정주가·거래소 달력·외부 시각 인증·생산 학습 승인은 검증하지 않았습니다.</p></html>"""


def run(source, learning_source, root, *, progress=print, minimums=(100, 30, 30), minimum_documents=6):
    progress("저장 원문·재무 대조·접수번호별 근거를 재현 확인 중...")
    parent, original, rows, coverage, _, finance, artifacts = verify_source(source)
    require(parent["source_learning"] and Path(parent["source_learning"]).resolve() == learning_source.resolve(),
            "재무 대조에 사용한 원래 학습 폴더와 현재 관찰 모델의 원본이 다릅니다.")
    require(content(source / "financial_summary.json") == content(learning_source / "summary.json")
            and content(source / "financial_audit.json") == content(learning_source / "financial_audit.json"),
            "재무 대조 때의 학습 자료 기록이 달라졌습니다. 기존 기록을 보존하고 확인해주세요.")
    narrative = build_records(rows, coverage)
    records = financial_records(finance, coverage, narrative, rows)
    codes = {s["code"] for s in original["stocks"]}
    complete = {s["code"]: s["read_documents"] == s["target_documents"] > 0 and s["selection_status"] == "available" for s in original["stocks"]}
    progress("같은 학습 사례에 공개 당시의 영업이익률을 연결하는 중...")
    source_inputs = {name: content(learning_source / name) for name in (
        "summary.json", "financial_audit.json", "models.json", "features_with_financials.csv", "future_outcomes.csv")}
    old_models, features, outcomes = load_source(learning_source)
    diagnosis, _ = diagnose(old_models, features, outcomes)
    require(parse_date(original["reference_date"]) >= max(parse_date(f["as_of_date"]) for f in features))
    paired, exclusions = attach_financials(features, records, coverage, complete)
    horizons = [h for h in ("5", "20") if diagnosis["models"].get(h, {}).get("disposition") == "hold"]
    progress(f"재무 연결 {len(paired)}/{len(features)}사례 · 같은 사례에서 두 연구 모델 학습·비교 중...")
    models = train_comparison(paired, outcomes, old_models["split"], horizons, minimums=minimums, minimum_documents=minimum_documents)
    current = [r for r in records if r["period_end"] == r["report_period_end"] and r["status"] == "available"]
    view_records = [{**r, "reference_value_krw": r["value_krw"], "id": hashlib.sha256(json.dumps([r[k] for k in
                    ("code", "receipt_no", "scope", "period_end", "metric")]).encode()).hexdigest()[:20]} for r in current]
    snapshots = snapshot_as_of(view_records, coverage, codes, original["reference_date"], completeness=complete)
    summary = {"schema_version": 1, "policy": POLICY, "created_at": datetime.now(SEOUL).isoformat(),
        "source_crosscheck": str(source.resolve()), "source_learning": str(learning_source.resolve()),
        "source_crosscheck_sha256": hashlib.sha256(content(source / "summary.json")).hexdigest(),
        "source_rows": len(features), "paired_rows": len(paired), "codes": sorted(codes), "reference_date": original["reference_date"],
        "narrative_records": len(narrative), "available_current_records": len(current), "exclusions": exclusions,
        "record_status_counts": dict(Counter(r["status"] for r in records)), "coverage_complete_by_code": complete,
        "margin_rows": sum(f[EXTRA_FEATURES[0]] is not None for f in paired), "margin_change_rows": sum(f[EXTRA_FEATURES[1]] is not None for f in paired),
        "source_complete": all(complete.values()) and parent["financial_read"] == parent["text_documents"],
        "status": models["status"], "research_training_executed": any(m["status"] == "trained" for m in models["models"].values()),
        "production_input_approved": False, "fresh_unseen_evaluation": False, "independent_accuracy_verified": False,
        "historical_point_in_time_certified": False, "existing_model_modified": False, "automatic_promotion": False,
        "live_trading_allowed": False, "costs_included": False, "semantic_accuracy_established": False}
    historical = []
    indexed = index_outcomes(outcomes)
    for horizon, comparison in models["models"].items():
        if comparison["status"] != "trained": continue
        parts, _ = partition_examples(paired, indexed, int(horizon), old_models["split"]["validation_start"], old_models["split"]["test_start"])
        test = parts["test"]
        base, extra = (predict_model(comparison[k], test) for k in ("numeric", "numeric_and_finance"))
        for (f, o), b, e in zip(test, base, extra):
            historical.append({"horizon": horizon, "code": f["code"], "as_of_date": f["as_of_date"], "entry_date": o["entry_date"],
                "exit_date": o["exit_date"], "source_financial_document_id": f["source_financial_document_id"],
                "actual_change_pct": o["price_return_pct"], "numeric_prediction_pct": float(b), "finance_prediction_pct": float(e)})
    parent_dir = root / "finance_learning"; parent_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent_dir))
    target = parent_dir / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    try:
        (staging / "financial_sources").mkdir(); (staging / "inputs").mkdir()
        for stem, (raw, metadata) in artifacts.items():
            folder = staging / "financial_sources" / stem; folder.mkdir()
            (folder / "original.zip").write_bytes(raw); (folder / "source.json").write_bytes(metadata)
            code, receipt = stem.split("_")
            (folder / "facts.html").write_text(financial_html(finance[code, receipt]["facts"], json.loads(metadata)), encoding="utf-8")
        input_hashes = {}
        for name, raw in source_inputs.items():
            require(content(learning_source / name) == raw, "비교 중 원래 학습 자료가 바뀌어 새 결과를 저장하지 않았습니다.")
            (staging / "inputs" / name).write_bytes(raw); input_hashes[name] = hashlib.sha256(raw).hexdigest()
        for name in ("summary.json", "documents.json", "comparisons.jsonl", "baseline_comparisons.jsonl"):
            raw = content(source / name)
            (staging / "inputs" / ("crosscheck_"+name)).write_bytes(raw)
            input_hashes["crosscheck_"+name] = hashlib.sha256(raw).hexdigest()
        summary["input_sha256"] = input_hashes
        for name, obj in (("financial_records.json", records), ("coverage.json", coverage), ("as_of.json", snapshots),
                          ("feature_joins.json", paired), ("models.json", models)):
            (staging / name).write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        write_csv(staging / "financial_records.csv", ["code", "receipt_no", "receipt_date", "available_from", "report_period_end", "period_start", "period_end",
            "scope", "metric", "status", "value_krw", "quantum", "source_sha256", "source_kind", "narrative_record_id"], records)
        write_csv(staging / "feature_joins.csv", ["code", "as_of_date", "source_financial_document_id", "source_financial_scope", "source_financial_available_from",
            "source_financial_sha256", "source_current_period_end", "source_comparative_period_end", *EXTRA_FEATURES, "source_margin_missing", "source_margin_change_missing"], paired)
        write_csv(staging / "historical_comparisons.csv", ["horizon", "code", "as_of_date", "entry_date", "exit_date", "source_financial_document_id",
            "actual_change_pct", "numeric_prediction_pct", "finance_prediction_pct"], historical)
        summary["output_sha256"] = {n: hashlib.sha256(content(staging / n)).hexdigest() for n in (
            "financial_records.json", "coverage.json", "as_of.json", "feature_joins.json", "models.json", "historical_comparisons.csv", "financial_records.csv", "feature_joins.csv")}
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        (staging / "report.html").write_text(render_report(summary, records, snapshots, models), encoding="utf-8")
        (staging / "records.html").write_text(render_report(summary, records, snapshots, models, show_records=True), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return target, summary, models


def main(argv=None):
    default = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="저장 재무 원문의 보완과 별도 연구 모델 비교")
    parser.add_argument("--root", type=Path, default=default)
    parser.add_argument("--input", type=Path, help="기존 CROSS_CHECK 결과; 기본은 최신 결과")
    parser.add_argument("--learning", type=Path, help="기존 수치 학습 결과; 기본은 관찰 중인 모델의 원본")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, summary, _ = run(args.input or latest_crosscheck(args.root / "financial_crosschecks"), args.learning or select_learning(args.root), args.root)
        print(f"\n재무 원문에서 직접 확인한 당기 항목: {summary['available_current_records']}개 · 연결 사례 {summary['paired_rows']}/{summary['source_rows']}")
        print("재무 추가 모델의 연구용 학습·과거 비교 완료" if summary["status"] == "trained" else "일부 구간 학습 또는 조건 미충족 · 보고서를 확인해주세요.")
        print(f"결과 보고서: {target / 'report.html'}")
        print("기존 모델·예측은 유지합니다. 실제 주문과 자동 채택은 실행하지 않았습니다.")
        if sys.platform == "win32" and not args.no_open: webbrowser.open((target / "report.html").resolve().as_uri())
        return 0 if summary["status"] == "trained" and summary["source_complete"] else 2
    except CollectionError as exc: print(f"재무 학습 확인 필요: {exc}", file=sys.stderr)
    except FileNotFoundError: print("기존 학습·재무 대조 자료가 없습니다. 원래 저장 폴더를 확인해주세요. 인증키는 필요하지 않습니다.", file=sys.stderr)
    except KeyboardInterrupt: return 130
    except Exception: print("재무 추가 연구를 완료하지 못했습니다. 마지막 메시지를 알려주세요. 기존 파일은 바꾸지 않았습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__": sys.exit(main())
