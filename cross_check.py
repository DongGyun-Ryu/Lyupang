"""Compare saved report amounts with the same receipts' cached financial originals."""
from __future__ import annotations

import argparse
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
from collections import Counter
from datetime import datetime
from pathlib import Path

from business_text import source_html
from check_facts import content, decoded, latest_facts, load_facts, require
from facts import METRIC_NAMES, display_value
from financial_crosscheck import TEXT_METRICS, STATES, REASONS, cached_xbrl, compare_amount, parse_facts
from financial_history import report_period
from lyupang import CollectionError, SEOUL, parse_date
from report_context import ContextIndex, POLICY as CONTEXT_POLICY


def learning_source(root):
    pinned = root / "forward/experiment/experiment.json"
    if pinned.exists():
        record = decoded(content(pinned))
        canonical = json.dumps(record["payload"], sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        require(hashlib.sha256(canonical).hexdigest() == record["sha256"], "고정 모델의 원래 자료 경로 기록 해시가 다릅니다.")
        require(record["payload"].get("policy") == "forward_journal_v1")
        return Path(record["payload"]["source_learning"])
    parent = root / "learning"
    folders = sorted(p for p in parent.iterdir() if p.is_dir() and not p.is_symlink()
                     and re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{8}", p.name)) if parent.exists() else []
    require(bool(folders), "저장된 과거 재무 원문 검증 기록을 찾지 못했습니다.")
    return folders[-1]


def load_audit(source):
    raw_summary = content(source / "summary.json")
    summary = decoded(raw_summary)
    require(summary["schema_version"] == 1 and isinstance(summary["source_research"], str))
    raw = content(source / "financial_audit.json")
    audit = decoded(raw)
    require(isinstance(audit, dict) and len(audit) <= 20)
    for code, events in audit.items():
        require(re.fullmatch(r"[0-9]{6}", code) and isinstance(events, list) and len(events) <= 30)
        receipts = [e["receipt_no"] for e in events]
        require(len(receipts) == len(set(receipts)) and all(re.fullmatch(r"[0-9]{14}", r) for r in receipts))
    return summary, audit, raw_summary, raw


def safe_reason(exc):
    return str(exc) if isinstance(exc, CollectionError) else "저장된 재무 원문의 형식·기간·단위를 확인해야 합니다."


def financial_html(facts, metadata):
    esc = lambda x: html.escape(str(x))
    rows = "".join(f"<tr id='{f['id']}'><td>{esc(f['metric'])}</td><td>{esc(f['value'])}</td><td>{esc(f['scope'])}</td>"
        f"<td>{esc(f['period_start'])} ~ {esc(f['period_end'])}</td><td>{esc(f['member'])}<br>{esc(f['concept'])}<br>context: {esc(f['context_ref'])}<br>decimals: {esc(f['decimals'])}</td></tr>" for f in facts)
    return f"<!doctype html><html lang='ko'><meta charset='utf-8'><title>접수번호별 재무 항목</title><h1>{esc(metadata['receipt_no'])}</h1><p>원문 ZIP SHA256: {esc(metadata['sha256'])}</p><table><tr><th>항목</th><th>원 단위 값</th><th>범위</th><th>기간</th><th>원문 위치</th></tr>{rows}</table></html>"


def row_html(row):
    esc = lambda x: html.escape(str(x))
    q = row["quantity"]
    value = display_value(q)
    references = " / ".join(sorted({f["value"]+" 원 ("+f["scope"]+")" for f in row["financial_facts"]})) or "조건 미충족 또는 항목 없음"
    notes = " / ".join(REASONS[r] for r in row["reasons"])
    if row["status"] == "rounding_overlap": notes = "반올림을 가정한 표시 범위만 겹칩니다. 금액이 같거나 해석이 옳다는 확정이 아닙니다."
    source = f"text_sources/{row['code']}_{row['receipt_no']}.html#{row['block_id']}"
    links = " · ".join(f"<a href='financial_sources/{row['code']}_{row['receipt_no']}/facts.html#{f['id']}'>재무 위치 {i+1}</a>" for i, f in enumerate(row["financial_facts"]))
    context = row.get("context_resolution")
    context_html = ""
    if context:
        names = {"sentence_explicit": "문장에 직접 표시", "clause_explicit": "같은 문장의 범위별 구분", "source_context": "원문 문맥에서 연결", "unresolved": "미확정"}
        scope = {"CFS": "연결", "OFS": "별도"}.get(context["scope"], "미확정")
        context_html = f"<p>대조 기간: {esc(context['year'] or '미확정')} ({names[context['year_method']]}) · 기준: {scope} ({names[context['basis_method']]})</p>"
        for evidence in context["evidence"]:
            link = f"text_sources/{evidence['code']}_{evidence['receipt_no']}.html#{evidence['block_id']}"
            context_html += f"<blockquote>{esc(evidence['text'])}</blockquote><p><a href='{link}'>기간·기준 근거 원문</a> · 문단 {esc(evidence['block_id'])} · 문자 {evidence['start']}~{evidence['end']}</p>"
        if context["conflicts"]: context_html += "<p>적용 범위가 섞이거나 모호한 문맥은 가져오지 않았습니다.</p>"
    return (f"<article><h3>{esc(STATES[row['status']])} · {esc(row['code'])} · {esc(METRIC_NAMES.get(q['metric'], q['metric']))}</h3>"
        f"<p>접수번호 {esc(row['receipt_no'])}</p><blockquote>{esc(row['text'])}</blockquote>"
        f"<p>설명 수치: {esc(q['text'])} → {esc(value)}<br>재무 원문: {esc(references)}<br>차이(설명−재무): {esc(row['difference_krw'] if row['difference_krw'] is not None else '미대조')} 원</p>"
        f"<p>{esc(notes)}</p>{context_html}<p><a href='{source}'>설명 원문</a> · {links}</p></article>")


def render_report(summary, documents, rows, *, all_rows=False):
    esc = lambda x: html.escape(str(x))
    counts = summary["comparison_counts"]
    document_rows = "".join(f"<tr><td>{esc(d['code'])}</td><td>{esc(d['receipt_no'])}</td><td>{'읽음' if d['status']=='read' else '확인 필요'}</td><td>{esc(d.get('reason',''))}</td></tr>" for d in documents)
    reasons = "".join(f"<tr><td>{esc(REASONS[key])}</td><td>{count}</td></tr>" for key, count in summary["reason_counts"].items())
    shown = rows if all_rows else [r for r in rows if r["status"] != "not_comparable"]
    examples = []
    if not all_rows:
        for reason in summary["reason_counts"]:
            found = next((r for r in rows if reason in r["reasons"]), None)
            if found is not None and found not in examples: examples.append(found)
    sections = "".join(row_html(r) for r in shown)
    extra = "<h2>조건을 확인할 예시</h2>"+"".join(row_html(r) for r in examples) if examples else ""
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>설명 금액과 재무 원문 대조</title><style>body{{font-family:system-ui;line-height:1.8;max-width:1200px;margin:30px auto;padding:20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}blockquote{{white-space:pre-wrap;background:#f2f6fb;padding:15px;margin:12px 0}}article{{border-top:1px solid #ddd;margin-top:25px}}.note{{background:#fff2d5;padding:16px}}</style>
<h1>설명 금액과 재무 원문 대조</h1><p class="note">같은 회사가 제출한 설명과 재무 자료의 대조입니다. 독립된 전문가의 정답·새 문서 정확도·주가 원인·매매 성능을 검증한 결과가 아닙니다. 학습 사용과 주문은 연결하지 않습니다.</p>
<h2>대조 요약</h2><p>설명 원문 읽기 {summary['text_documents']}/{summary['text_target_documents']}건 · 재무 원문 읽기 {summary['financial_read']}/{summary['text_documents']}건.<br>
매출·영업수익·이익/손실에 연결한 수치 {summary['financial_quantities']}개 중 원화 금액 {summary['krw_amounts']}개.</p>
<table><tr><th>조건 내 표기값 같음</th><th>표시 정밀도 범위 겹침</th><th>조건 내 금액 다름</th><th>조건 확인 필요</th></tr><tr><td>{counts['exact_match']}</td><td>{counts['rounding_overlap']}</td><td>{counts['value_mismatch']}</td><td>{counts['not_comparable']}</td></tr></table>
<p>정확도 비율이 아닙니다. 조건 확인 필요는 통과·불일치로 채우지 않습니다. 대조 0건이면 검증 완료가 아닙니다. 범위 겹침은 표시 단위로 반올림했다고 가정했을 때의 후보이며 같은 값이라는 확정이 아닙니다.</p>
<h2>문맥 근거 연결</h2><p>같은 수치를 문장만으로 대조하면 {summary['baseline_compared']}건, 원문 문맥을 포함하면 {summary['context_compared']}건입니다. 문맥 근거가 연결된 대조는 {summary['context_evidence_compared']}건입니다. 이 증가가 정확도 향상을 입증하지 않습니다. 원래 수치·문장·모델을 바꾸지 않습니다.</p>
<p>같은 문장의 연결/별도 구분, 같은 문단의 단일 기준, 현재 제목, 적용 방향이 명시된 인접 선언, 해당 장을 명시한 작성 기준과 명시적 회계기수-연도 쌍만 제한적으로 사용합니다. 근거 표현의 문단·문자 위치·해시를 보관합니다. 다른 회사·혼합 범위·여러 연도·전망·참조 문맥은 보류합니다. 보고서 연도나 재무 금액에 맞춰 기준을 고르지 않습니다. 이 규칙이 모든 문맥의 적용 범위를 정확히 이해한다는 보장은 없습니다.</p>
<p>같은 접수번호·기간·명시적 연결/별도 기준과 표준 항목이 맞을 때만 대조합니다. 사업 부문·전망·증감률·여러 금액 등 해석이 남은 문장은 확정하지 않습니다. 범위가 없는 XBRL 문맥을 연결로 추정하지 않습니다. 매출·영업수익은 표준 Revenue에 제한해 대응하며 모든 업종의 회계 의미가 같다고 인증하지 않습니다. 표준 항목 일부만 지원합니다.</p>
<h2>재무 원문 상태</h2><table><tr><th>종목</th><th>접수번호</th><th>읽기</th><th>확인할 내용</th></tr>{document_rows}</table>
<p>{esc(' / '.join(summary['setup_errors']))}</p><h2>대조하지 못한 이유</h2><table>{reasons}</table><p>한 수치에 여러 사유가 있을 수 있습니다.</p>
<h2>{'전체 대조 기록' if all_rows else '조건을 충족한 대조 전체'}</h2>{sections or '<p>조건을 모두 충족한 금액 대조가 없습니다.</p>'}{extra}
<p><a href="comparisons.html">미대조를 포함한 전체 수치 보기</a> · 전체 자료는 comparisons.jsonl에 보관합니다. 요약 화면의 미대조 예시는 사유당 최대 1개이며 전체 기록에서 생략 없이 확인할 수 있습니다.</p>
<h2>확인의 범위</h2><p>재무 원문은 과거 기업 확인을 마친 검증 기록과 같은 ZIP 해시인지 확인하고, 단일 기업 식별자와 이전 재무 금액·기간·범위의 재현을 확인한 뒤 읽었습니다. 과거 저장 기록에 의존하며 이번에 외부 기관에 기업 신원을 새로 확인하지 않았습니다. 모든 식별자·태그·context·기간·단위·표시 정밀도와 해시를 결과에 보관합니다. 두 자료 모두 같은 회사가 제출한 내용이며 외부 사실 인증은 아닙니다.</p>
<p>후속 정정이나 다른 접수번호로 대신 채우지 않습니다. 이전 재무 검증 기록은 독립된 정답이 아니고 전체 분류체계·뉴스 의미·시장 성능 검증은 남아 있습니다. 기존 모델·전향 예측·원문을 바꾸거나 재학습하지 않았습니다. 학습 사용 대기 상태를 유지합니다.</p><p>처리 시각: {esc(summary['created_at'])}</p></html>"""


def run(source, root, learning=None, cache=None, *, progress=print):
    original, claims, text_docs, text_files, raw_summary, raw_claims = load_facts(source)
    setup_errors, finance, artifacts = [], {}, {}
    audit_summary = audit_data = audit_summary_raw = audit_raw = None
    source_learning = None
    try:
        source_learning = learning or learning_source(root)
        audit_summary, audit_data, audit_summary_raw, audit_raw = load_audit(source_learning)
    except Exception as exc:
        setup_errors.append(safe_reason(exc))
    cache = cache or root / "financial_documents"
    total_bytes = 0
    for (code, receipt), text_doc in text_docs.items():
        progress(f"종목 {code} · 접수번호 {receipt} · 저장된 재무 원문 대조 준비...")
        doc = {"code": code, "receipt_no": receipt, "status": "unavailable", "facts": []}
        try:
            require(audit_data is not None, "저장된 재무 원문 검증 기록 미확인")
            records = [e for e in audit_data.get(code, []) if e["receipt_no"] == receipt]
            require(len(records) == 1 and records[0]["status"] == "verified", "같은 접수번호의 과거 재무 검증 기록이 없습니다.")
            event = records[0]
            for key in ("receipt_date", "period_end"):
                require(event[key] == text_doc["metadata"][key], "설명과 재무 검증 기록의 기간·접수일이 다릅니다.")
            require(report_period(event["report_name"]).isoformat() == event["period_end"])
            research = Path(audit_summary["source_research"])
            company_raw = content(research / "dart" / (code+".json"))
            corp = decoded(company_raw)["company"]["corp_code"]
            require(re.fullmatch(r"[0-9]{8}", corp))
            raw, metadata = cached_xbrl(cache, code, receipt, event["sha256"])
            total_bytes += len(raw); require(total_bytes <= 128 * 1024 * 1024)
            facts, identity = parse_facts(raw, parse_date(event["period_end"]), event["metrics"], corp, code)
            doc.update(status="read", facts=facts, source_sha256=event["sha256"], corp_code=corp, **identity)
            artifacts[code+"_"+receipt] = (raw, {**metadata, **identity, "company_record_sha256": hashlib.sha256(company_raw).hexdigest()})
        except Exception as exc:
            doc["reason"] = safe_reason(exc)
        finance[code, receipt] = doc
    indexes = {key: ContextIndex(doc["blocks"], doc["company_name"], code=key[0], receipt=key[1]) for key, doc in text_docs.items()}
    pairs = [(c, q) for c in claims for q in c["measurements"] if q["metric"] in TEXT_METRICS]
    baseline = [compare_amount(c, q, finance[c["code"], c["receipt_no"]]) for c, q in pairs]
    rows = [compare_amount(c, q, finance[c["code"], c["receipt_no"]],
                context=indexes[c["code"], c["receipt_no"]].resolve(c, q)) for c, q in pairs]
    counts = {state: sum(r["status"] == state for r in rows) for state in STATES}
    reasons = Counter(reason for row in rows for reason in row["reasons"])
    summary = {"schema_version": 1, "created_at": datetime.now(SEOUL).isoformat(), "source_facts": str(source.resolve()),
        "source_learning": str(source_learning.resolve()) if source_learning else None,
        "source_summary_sha256": hashlib.sha256(raw_summary).hexdigest(), "source_claims_sha256": hashlib.sha256(raw_claims).hexdigest(),
        "financial_audit_sha256": hashlib.sha256(audit_raw).hexdigest() if audit_raw else None,
        "text_documents": len(text_docs), "text_target_documents": original["target_documents"], "text_source_status": original["status"],
        "financial_read": sum(d["status"] == "read" for d in finance.values()),
        "financial_quantities": len(rows), "krw_amounts": sum(r["quantity"]["unit"] == "KRW" for r in rows),
        "comparison_counts": counts, "reason_counts": dict(reasons), "setup_errors": setup_errors,
        "context_policy": CONTEXT_POLICY, "baseline_counts": {s: sum(r['status'] == s for r in baseline) for s in STATES},
        "baseline_compared": sum(r['status'] != 'not_comparable' for r in baseline),
        "context_compared": sum(r['status'] != 'not_comparable' for r in rows),
        "context_evidence_compared": sum(r['status'] != 'not_comparable' and bool(r['context_resolution']['evidence']) for r in rows),
        "learning_ready": False, "independent_accuracy_verified": False, "fresh_unseen_evaluation": False,
        "model_inputs_modified": False, "training_executed": False, "live_trading_allowed": False,
        "status": "review_required"}
    compared = sum(counts[k] for k in ("exact_match", "rounding_overlap", "value_mismatch"))
    if (compared and not counts["value_mismatch"] and not reasons.get("fact_conflict") and not setup_errors
            and original["status"] == "structured" and summary["financial_read"] == len(text_docs)):
        summary["status"] = "comparisons_available"
    parent = root / "financial_crosschecks"; parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    target = parent / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    try:
        (staging / "text_sources").mkdir(); (staging / "financial_sources").mkdir()
        for stem, raw in text_files.items():
            (staging / "text_sources" / (stem+".json")).write_bytes(raw)
            (staging / "text_sources" / (stem+".html")).write_text(source_html(decoded(raw), stem), encoding="utf-8")
        for stem, (raw, metadata) in artifacts.items():
            folder = staging / "financial_sources" / stem; folder.mkdir()
            (folder / "original.zip").write_bytes(raw)
            (folder / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            code, receipt = stem.split("_")
            (folder / "facts.html").write_text(financial_html(finance[code, receipt]["facts"], metadata), encoding="utf-8")
        for name, raw in (("input_summary.json", raw_summary), ("input_claims.jsonl", raw_claims), ("financial_audit.json", audit_raw), ("financial_summary.json", audit_summary_raw)):
            if raw is not None: (staging / name).write_bytes(raw)
        for name, data in (("summary.json", summary), ("documents.json", list(finance.values()))):
            (staging / name).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        with (staging / "comparisons.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        with (staging / "baseline_comparisons.jsonl").open("w", encoding="utf-8") as stream:
            for row in baseline: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        (staging / "report.html").write_text(render_report(summary, list(finance.values()), rows), encoding="utf-8")
        (staging / "comparisons.html").write_text(render_report(summary, list(finance.values()), rows, all_rows=True), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return target, summary


def main(argv=None):
    default = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="저장된 설명 금액과 같은 접수번호의 재무 원문 대조")
    parser.add_argument("--root", type=Path, default=default)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--learning", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, summary = run(args.input or latest_facts(args.root / "facts"), args.root, args.learning, args.cache)
        counts = summary["comparison_counts"]
        print(f"\n재무 원문 읽기: {summary['financial_read']}/{summary['text_documents']}건")
        print(f"표기값 같음 {counts['exact_match']} · 표시 범위 겹침 {counts['rounding_overlap']} · 금액 다름 {counts['value_mismatch']} · 조건 확인 필요 {counts['not_comparable']}")
        print("학습 사용 대기 · 기존 모델·예측 유지 · 실제 주문 없음")
        print(f"대조 보고서: {target / 'report.html'}")
        if sys.platform == "win32" and not args.no_open: webbrowser.open((target / "report.html").resolve().as_uri())
        return 0 if summary["status"] == "comparisons_available" else 2
    except CollectionError as exc:
        print(f"대조 확인 필요: {exc}", file=sys.stderr)
    except FileNotFoundError:
        print("저장된 설명 파일을 찾지 못했습니다. 원래 자료 폴더를 확인해주세요. 인증키는 필요하지 않습니다.", file=sys.stderr)
    except KeyboardInterrupt:
        return 130
    except Exception:
        print("대조 자료의 형식 또는 저장 상태를 확인해야 합니다. 기존 파일은 바꾸지 않았습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
