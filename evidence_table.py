"""Reproducible, source-linked research tables; no model or order integration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

from business_text import source_html
from check_facts import content, decoded, load_facts, number, require
from cross_check import financial_html, row_html
from facts import display_value
from financial_crosscheck import STATES, TEXT_METRICS, compare_amount, parse_facts
from financial_history import report_period
from lyupang import CollectionError, SEOUL, parse_date
from report_context import ContextIndex, POLICY as CONTEXT_POLICY

METRICS = {"revenue": "매출·영업수익", "operating_income": "영업이익", "net_income": "당기순이익"}
SCOPES = {"CFS": "연결", "OFS": "별도"}
POLICY = "receipt_evidence_table_v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_lines(path):
    return [decoded(line) for line in content(path).splitlines() if line.strip()]


def latest_crosscheck(root):
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
        and re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{8}", p.name)) if root.exists() else []
    require(bool(folders), "저장된 재무 대조 결과가 없습니다. 기존 CROSS_CHECK 결과 폴더를 확인해주세요.")
    return folders[-1]


def verify_source(source):
    raw_summary = content(source / "summary.json")
    summary = decoded(raw_summary)
    require(summary["schema_version"] == 1 and summary.get("context_policy") == CONTEXT_POLICY,
            "현재 문맥 근거를 포함한 CROSS_CHECK 결과가 필요합니다.")
    flags = ("learning_ready", "independent_accuracy_verified", "fresh_unseen_evaluation", "model_inputs_modified", "training_executed", "live_trading_allowed")
    require(all(summary.get(key) is False for key in flags))
    original, claims, docs, text_files, facts_summary_raw, facts_claims_raw = load_facts(Path(summary["source_facts"]))
    require(digest(facts_summary_raw) == summary["source_summary_sha256"] and digest(facts_claims_raw) == summary["source_claims_sha256"])
    require(content(source / "input_summary.json") == facts_summary_raw and content(source / "input_claims.jsonl") == facts_claims_raw)
    require(len(docs) == summary["text_documents"] and original["target_documents"] == summary["text_target_documents"])
    for stem, raw in text_files.items(): require(content(source / "text_sources" / (stem+".json"), 64*1024*1024) == raw)
    saved_documents = decoded(content(source / "documents.json"))
    require(len(saved_documents) == len(docs))
    finance = {(d["code"], d["receipt_no"]): d for d in saved_documents}
    require(len(finance) == len(docs) and set(finance) == set(docs))
    artifacts = {}
    raw_audit = raw_financial_summary = None
    if any(d["status"] == "read" for d in saved_documents):
        raw_audit = content(source / "financial_audit.json")
        raw_financial_summary = content(source / "financial_summary.json")
        require(digest(raw_audit) == summary["financial_audit_sha256"])
        audit = decoded(raw_audit)
        research = Path(decoded(raw_financial_summary)["source_research"])
        total_bytes = 0
        for key, d in finance.items():
            if d["status"] != "read":
                require(d["status"] == "unavailable" and not d["facts"])
                continue
            code, receipt = key
            matches = [e for e in audit.get(code, []) if e["receipt_no"] == receipt]
            require(len(matches) == 1 and matches[0]["status"] == "verified")
            event = matches[0]
            meta = docs[key]["metadata"]
            require(event["receipt_date"] == meta["receipt_date"] and event["period_end"] == meta["period_end"]
                    and report_period(event["report_name"]).isoformat() == meta["period_end"])
            folder = source / "financial_sources" / (code+"_"+receipt)
            raw = content(folder / "original.zip")
            raw_meta = content(folder / "source.json", 64*1024); saved_meta = decoded(raw_meta)
            require((saved_meta["provider"], saved_meta["endpoint"], saved_meta["code"], saved_meta["receipt_no"]) == ("OpenDART", "fnlttXbrl.xml", code, receipt))
            require(digest(raw) == event["sha256"] == d["source_sha256"] == saved_meta["sha256"])
            company_raw = content(research / "dart" / (code+".json"))
            require(digest(company_raw) == saved_meta["company_record_sha256"])
            corp = decoded(company_raw)["company"]["corp_code"]
            require(corp == d["corp_code"] and re.fullmatch(r"[0-9]{8}", corp))
            facts, identity = parse_facts(raw, parse_date(meta["period_end"]), event["metrics"], corp, code)
            require(facts == d["facts"] and all(d[k] == v and saved_meta[k] == v for k, v in identity.items()))
            total_bytes += len(raw); require(total_bytes <= 128*1024*1024)
            artifacts[code+"_"+receipt] = (raw, raw_meta)
    else:
        require(all(d["status"] == "unavailable" and not d["facts"] for d in saved_documents))
    require(sum(d["status"] == "read" for d in saved_documents) == summary["financial_read"])
    indexes = {key: ContextIndex(d["blocks"], d["company_name"], code=key[0], receipt=key[1]) for key, d in docs.items()}
    pairs = [(c, q) for c in claims for q in c["measurements"] if q["metric"] in TEXT_METRICS]
    rows = [compare_amount(c, q, finance[c["code"], c["receipt_no"]], context=indexes[c["code"], c["receipt_no"]].resolve(c, q)) for c, q in pairs]
    baseline = [compare_amount(c, q, finance[c["code"], c["receipt_no"]]) for c, q in pairs]
    require(rows == json_lines(source / "comparisons.jsonl") and baseline == json_lines(source / "baseline_comparisons.jsonl"),
            "저장된 대조 결과를 원문에서 재현하지 못했습니다. 원래 결과를 확인해주세요.")
    require(summary["comparison_counts"] == {s: sum(r["status"] == s for r in rows) for s in STATES})
    require(summary["baseline_counts"] == {s: sum(r["status"] == s for r in baseline) for s in STATES})
    require(len(rows) == summary["financial_quantities"])
    require(summary["baseline_compared"] == sum(r["status"] != "not_comparable" for r in baseline))
    require(summary["context_compared"] == sum(r["status"] != "not_comparable" for r in rows))
    require(summary["context_evidence_compared"] == sum(r["status"] != "not_comparable" and bool(r["context_resolution"]["evidence"]) for r in rows))
    coverage = [{"code": key[0], "company_name": doc["company_name"], **doc["metadata"],
                 "financial_status": finance[key]["status"]} for key, doc in docs.items()]
    return summary, original, rows, coverage, text_files, finance, artifacts


def build_records(rows, coverage):
    documents = {(d["code"], d["receipt_no"]): d for d in coverage}
    groups = {}
    for row in rows:
        if row["status"] not in {"exact_match", "rounding_overlap"}: continue
        ctx, refs = row["context_resolution"], row["financial_facts"]
        require(ctx["year"] is not None and ctx["scope"] in SCOPES and refs)
        metric = TEXT_METRICS[row["quantity"]["metric"]]
        doc = documents[row["code"], row["receipt_no"]]
        require(doc["financial_status"] == "read" and not row["model_input_approved"])
        for f in refs:
            require((f["metric"], f["scope"], f["period_end"]) == (metric, ctx["scope"], doc["period_end"]))
        values = {number(f["value"]) for f in refs}; require(len(values) == 1)
        key = (row["code"], row["receipt_no"], ctx["scope"], refs[0]["period_start"], doc["period_end"], metric)
        value = format(next(iter(values)), "f")
        if key not in groups:
            groups[key] = {"id": hashlib.sha256("|".join(key).encode()).hexdigest()[:20], "code": key[0], "company_name": doc["company_name"],
                "receipt_no": key[1], "scope": key[2], "period_start": key[3], "period_end": key[4], "metric": metric,
                "receipt_date": doc["receipt_date"], "available_from": doc["available_from"], "currency": "KRW",
                "reference_value_krw": value, "comparison_kinds": [], "evidence": [], "reference_facts": [],
                "candidate_only": True, "model_input_approved": False}
        record = groups[key]
        require(number(record["reference_value_krw"]) == number(value), "같은 항목·접수번호의 재무 표기값이 충돌합니다.")
        if row["status"] not in record["comparison_kinds"]: record["comparison_kinds"].append(row["status"])
        record["evidence"].append(row)
        for f in refs:
            if f not in record["reference_facts"]: record["reference_facts"].append(f)
    return sorted(groups.values(), key=lambda r: (r["code"], r["period_end"], r["receipt_date"], r["receipt_no"], r["scope"], r["metric"]))


def snapshot_as_of(records, coverage, codes, as_of, *, completeness=None):
    as_of = parse_date(as_of)
    result = []
    for code in sorted(codes):
        complete = completeness is None or completeness.get(code) is True
        eligible = [d for d in coverage if d["code"] == code and parse_date(d["available_from"][:10]) <= as_of] if complete else []
        chosen = None
        state = "저장 자료 중 공개된 대상 없음" if complete else "자료 수집 범위 미완료로 보류"
        if eligible:
            period = max(d["period_end"] for d in eligible)
            fiscal = [d for d in eligible if d["period_end"] == period]
            received = max(d["receipt_date"] for d in fiscal)
            latest = [d for d in fiscal if d["receipt_date"] == received]
            if len(latest) == 1: chosen = latest[0]
            else: state = "같은 날 여러 제출본으로 보류"
        for scope in SCOPES:
            row = {"as_of": as_of.isoformat(), "code": code, "scope": scope,
                "selected_receipt_no": chosen["receipt_no"] if chosen else None,
                "period_end": chosen["period_end"] if chosen else None, "available_from": chosen["available_from"] if chosen else None,
                "state": state, "values": {m: None for m in METRICS}, "record_ids": {}, "missing_metrics": list(METRICS),
                "model_input_approved": False}
            if chosen:
                available = [r for r in records if r["code"] == code and r["receipt_no"] == chosen["receipt_no"] and r["scope"] == scope]
                for r in available:
                    require(row["values"][r["metric"]] is None, "시점별 항목이 중복돼 선택하지 않았습니다.")
                    row["values"][r["metric"]] = r["reference_value_krw"]
                    row["record_ids"][r["metric"]] = r["id"]
                row["missing_metrics"] = [m for m in METRICS if row["values"][m] is None]
                row["state"] = "최신 제출본에서 기록된 항목 없음" if not available else (
                    "일부 항목 있음·검토용" if row["missing_metrics"] else "항목 있음·검토용")
            result.append(row)
    return result


def money(value):
    return display_value({"normalized_value": value, "unit": "KRW"}) if value is not None else "확인 자료 없음"


def render_report(summary, records, snapshots, *, details=False):
    esc = lambda x: html.escape(str(x))
    listing = "".join(f"<tr><td>{esc(r['company_name'])}<br>{r['code']}</td><td>{esc(r['period_end'])}<br>{SCOPES[r['scope']]}</td>"
        f"<td>{METRICS[r['metric']]}</td><td>{esc(' / '.join(dict.fromkeys(e['quantity']['text'] for e in r['evidence'])))}</td>"
        f"<td>{esc(money(r['reference_value_krw']))}</td><td>{esc(r['available_from'][:10])}</td>"
        f"<td>{esc(' / '.join(STATES[k] for k in r['comparison_kinds']))}<br><a href='details.html#{r['id']}'>근거 보기</a></td></tr>" for r in records)
    as_of = "".join(f"<tr><td>{r['code']} · {SCOPES[r['scope']]}</td><td>{esc(r['selected_receipt_no'] or '없음')}</td>"
        f"<td>{esc(money(r['values']['revenue']))}</td><td>{esc(money(r['values']['operating_income']))}</td>"
        f"<td>{esc(money(r['values']['net_income']))}</td><td>{esc(r['state'])}</td></tr>" for r in snapshots)
    sections = "".join(f"<section id='{r['id']}'><h2>{esc(r['company_name'])} · {METRICS[r['metric']]} · {SCOPES[r['scope']]}</h2>"
        f"<p>기간 {r['period_start']} ~ {r['period_end']} · 접수 {r['receipt_date']} · 사용 가능 {r['available_from']}<br>재무 원문 표기값: {esc(money(r['reference_value_krw']))}</p>"
        +"".join(row_html(e) for e in r['evidence'])+"</section>" for r in records) if details else ""
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>회사별 근거 자료표</title><style>body{{font-family:system-ui;line-height:1.8;max-width:1300px;margin:30px auto;padding:20px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left;font-size:14px}}blockquote{{white-space:pre-wrap;padding:15px;background:#f2f6fb}}.note{{background:#fff2d5;padding:18px}}section{{margin-top:30px}}</style>
<h1>회사별 근거 자료표</h1><p class="note">대조 근거가 있는 항목을 다시 볼 수 있게 모은 연구용 후보 표입니다. 학습 입력 승인·수익성 검증·실제 주문은 실행하지 않았습니다.</p>
<h2>정리 요약</h2><p>저장된 대조 {summary['source_rows']}수치를 원문과 재현 확인했습니다. 그중 조건 내 표기값 같음 또는 표시 범위 겹침 {summary['included_evidence']}개를 {summary['records']}항목으로 모았습니다. 종목 {summary['companies_with_records']}/{summary['source_companies']}개 · 고유 접수번호 {summary['receipts_with_records']}건.</p>
<p>같은 항목의 반복 설명은 근거를 모두 남기고 한 항목으로 모았습니다. 서로 다른 연결/별도 기준·접수번호는 합치지 않습니다. 항목 수는 독립 표본 수나 정확도 점수가 아닙니다. 근거가 없는 종목도 아래 시점별 표에서 확인할 수 있습니다.</p>
<h2>전체 제출 이력의 근거 금액</h2><p>위에서 확인한 전체 제출 이력을 아래 표에 모았습니다. 선택 날짜에 사용할 수 있었던 값은 이어지는 공개 시점별 표에 표시합니다.</p><table><tr><th>회사</th><th>기간·범위</th><th>항목</th><th>설명에 적힌 수치</th><th>재무 원문 표기값</th><th>사용 가능일</th><th>대조 결과</th></tr>{listing}</table>
<p>설명의 반올림 금액을 재무 값으로 덮어쓰지 않습니다. 두 표기를 나란히 보관합니다. 재무 원문 표기값에도 원문의 표시 정밀도가 있으며 외부에서 확인한 실제 값이라는 보증은 아닙니다.</p>
<h2>공개 시점별 자료 보기 · {summary['as_of']}</h2><table><tr><th>종목·범위</th><th>선택한 접수번호</th><th>매출·영업수익</th><th>영업이익</th><th>당기순이익</th><th>상태</th></tr>{as_of}</table>
<p>자료 기준일 {summary['reference_date']}까지 저장된 제출본만 고려합니다. 접수일 다음 날부터 사용 가능하다고 보수적으로 기록했습니다. 선택한 날짜에 공개된 최신 회계기간·최신 접수일을 먼저 고릅니다. 최신 제출본에 확인한 항목이 없으면 이전 값으로 대신 채우지 않고 비워둡니다. 같은 날 여러 대상 제출본이 있으면 보류합니다. 빈칸은 0이 아닙니다.</p>
<p><a href="evidence_table.csv" download>금액 표 CSV</a> · <a href="as_of.csv" download>시점별 표 CSV</a> · <a href="details.html">모든 원문 근거</a></p>
{sections}<h2>이번 자료의 범위</h2><p>이미 본 소수 기업·연간 원문의 개발용 정리입니다. 두 자료는 같은 회사 제출물이며 독립된 정답·새 문서 정확도·모든 항목의 완전성·과거 당시 자료의 완전한 재현·외부 시각 인증을 입증하지 않습니다. 새로 학습하거나 고정 모델·예측 기록을 바꾸지 않았습니다. 학습 사용 대기를 유지합니다.</p></html>"""


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def run(source, root, as_of=None, *, progress=print):
    progress("저장된 대조·원문·재무 표기값을 재현 확인 중...")
    parent_summary, original, rows, coverage, text_files, finance, artifacts = verify_source(source)
    records = build_records(rows, coverage)
    codes = {s["code"] for s in original["stocks"]}
    as_of = (parse_date(as_of) if as_of else parse_date(original["reference_date"])).isoformat()
    completeness = {s["code"]: s["read_documents"] == s["target_documents"] > 0 and s["selection_status"] == "available" for s in original["stocks"]}
    snapshots = snapshot_as_of(records, coverage, codes, as_of, completeness=completeness)
    summary = {"schema_version": 1, "policy": POLICY, "created_at": datetime.now(SEOUL).isoformat(), "source_crosscheck": str(source.resolve()),
        "source_summary_sha256": digest(content(source / "summary.json")), "source_comparisons_sha256": digest(content(source / "comparisons.jsonl")),
        "source_rows": len(rows), "included_evidence": sum(len(r["evidence"]) for r in records), "records": len(records),
        "source_companies": len(codes), "companies_with_records": len({r['code'] for r in records}),
        "receipts_with_records": len({(r['code'], r['receipt_no']) for r in records}), "reference_date": original["reference_date"], "as_of": as_of,
        "coverage_complete_by_code": completeness,
        "learning_ready": False, "independent_accuracy_verified": False, "historical_point_in_time_certified": False,
        "model_inputs_modified": False, "training_executed": False, "live_trading_allowed": False,
        "status": "candidate_table_created" if records else "waiting_evidence"}
    parent = root / "evidence_tables"; parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    target = parent / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    try:
        (staging / "text_sources").mkdir(); (staging / "financial_sources").mkdir()
        for stem, raw in text_files.items():
            (staging / "text_sources" / (stem+".json")).write_bytes(raw)
            (staging / "text_sources" / (stem+".html")).write_text(source_html(decoded(raw), stem), encoding="utf-8")
        for stem, (raw, metadata) in artifacts.items():
            folder = staging / "financial_sources" / stem; folder.mkdir()
            (folder / "original.zip").write_bytes(raw); (folder / "source.json").write_bytes(metadata)
            code, receipt = stem.split("_")
            (folder / "facts.html").write_text(financial_html(finance[code, receipt]["facts"], decoded(metadata)), encoding="utf-8")
        for name in ("summary.json", "documents.json", "comparisons.jsonl", "baseline_comparisons.jsonl", "input_summary.json", "input_claims.jsonl", "financial_audit.json", "financial_summary.json"):
            if (source / name).exists(): (staging / ("source_"+name)).write_bytes(content(source / name))
        for name, data in (("summary.json", summary), ("records.json", records), ("coverage.json", coverage), ("as_of.json", snapshots)):
            (staging / name).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        fields = ["code", "scope", "metric", "period_start", "period_end", "receipt_no", "receipt_date", "available_from", "reference_value_krw", "id"]
        write_csv(staging / "evidence_table.csv", fields, records)
        flat = [{**r, **r["values"], "missing_metrics": "/".join(r["missing_metrics"])} for r in snapshots]
        write_csv(staging / "as_of.csv", ["as_of", "code", "scope", "selected_receipt_no", "period_end", "available_from", *METRICS, "state", "missing_metrics"], flat)
        (staging / "report.html").write_text(render_report(summary, records, snapshots), encoding="utf-8")
        (staging / "details.html").write_text(render_report(summary, records, snapshots, details=True), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return target, summary


def main(argv=None):
    default = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="대조 근거가 있는 금액과 공개 시점별 후보 자료표")
    parser.add_argument("--root", type=Path, default=default)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, summary = run(args.input or latest_crosscheck(args.root / "financial_crosschecks"), args.root, args.as_of)
        print(f"\n근거 자료표: {summary['records']}항목 · {summary['companies_with_records']}/{summary['source_companies']}종목 · 고유 접수번호 {summary['receipts_with_records']}건")
        print(f"공개 시점별 보기: {summary['as_of']} · 학습 사용 대기 · 기존 모델·예측 유지")
        print(f"자료표 보고서: {target / 'report.html'}")
        if sys.platform == "win32" and not args.no_open: webbrowser.open((target / "report.html").resolve().as_uri())
        return 0 if summary["status"] == "candidate_table_created" else 2
    except CollectionError as exc: print(f"자료표 확인 필요: {exc}", file=sys.stderr)
    except FileNotFoundError: print("기존 대조 결과 또는 원문 파일이 없습니다. 원래 자료 폴더를 확인해주세요. 인증키는 필요하지 않습니다.", file=sys.stderr)
    except KeyboardInterrupt: return 130
    except Exception: print("원문과 저장 기록을 재현하지 못했습니다. 기존 자료는 바꾸지 않았습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__": sys.exit(main())
