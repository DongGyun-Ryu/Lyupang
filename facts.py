"""Offline, source-preserving business statement and measurement report."""
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
from datetime import datetime
from pathlib import Path

from business_text import evidence_as_of, source_html
from contextualize import latest_business, read_json, verified_blocks
from financial_history import report_period
from lyupang import CollectionError, SEOUL, parse_date
from report_claims import POLICY, TOPIC_NAMES, extract_claims, mark_possible_repeats

SCOPES = {"company": "회사에 관한 표현", "industry": "산업에 관한 표현", "related_or_segment": "다른 회사·계열사·부문 표현",
          "mixed": "범위 혼합", "unclear": "주체 불명확"}
TIMES = {"reported": "과거·현재 보고 표현", "outlook": "전망·계획", "mixed": "보고와 전망 혼합", "unclear": "시점 불명확"}
REASONS = {"scope_requires_review": "회사 전체에 관한 수치인지 확인 필요", "not_unambiguous_reported_statement": "전망·혼합·시점 확인 필요",
    "period_not_single_explicit_year": "문장 안의 기준 기간 확인 필요", "measurements_require_review": "수치·항목 연결·범위 확인 필요",
    "base_context_requires_review": "본문 문맥 확인 필요", "multiple_reporting_bases": "연결·별도 기준 혼합",
    "numeric_expression_not_supported": "지원하지 않는 수치 표현", "multiple_amounts_for_same_metric": "같은 항목에 여러 금액이 있어 역할 확인 필요",
    "adjustment_or_netting_context": "차감·자산화·총액·순액 등 조정 문맥 확인 필요", "conditional_context": "가정·조건에 따른 설명인지 확인 필요"}
METRIC_NAMES = {"revenue": "매출", "operating_revenue": "영업수익", "gross_profit": "매출총이익", "gross_margin": "매출총이익률",
    "operating_profit": "영업이익", "operating_loss": "영업손실",
    "net_profit": "순이익", "net_loss": "순손실", "operating_margin": "영업이익률", "net_margin": "순이익률",
    "cost_ratio": "원가율", "revenue_share": "매출 비중", "cost": "비용·원가", "impairment": "손상차손",
    "sales_volume": "판매량", "order_backlog": "수주잔고", "orders": "수주", "contract": "공급계약·체결",
    "investment": "투자", "research_cost": "연구개발비", "operating_cost": "영업비용", "selling_cost": "판매관리비",
    "material_cost": "원재료비", "labor_cost": "인건비"}


def tally(claims):
    return {"claims": len(claims), "reported_candidates": sum(c["status"] == "reported_candidate" for c in claims),
            "outlook_included": sum(c["time_orientation"] in {"outlook", "mixed"} for c in claims),
            "other_or_unclear_scope": sum(c["scope"] != "company" for c in claims),
            "needs_review": sum(c["status"] == "needs_review" for c in claims),
            "possible_repeated_claims": sum(c.get("possible_repeat_count", 1) > 1 for c in claims),
            "supported_measurements": sum(q["status"] == "supported" for c in claims for q in c["measurements"]),
            "bound_measurements": sum(q["status"] == "supported" and q["metric"] is not None for c in claims for q in c["measurements"])}


def source_reason(exc):
    if isinstance(exc, CollectionError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return "사업보고서의 본문 파일을 찾지 못했습니다. 원래 저장 폴더를 확인해주세요."
    if isinstance(exc, PermissionError):
        return "사업보고서 폴더를 읽을 권한을 확인해주세요."
    return "사업보고서 본문 형식 또는 저장 상태를 확인해야 합니다."


def display_value(quantity):
    value = quantity["normalized_value"]
    if value is None:
        return "미확정"
    integer, _, fraction = value.partition(".")
    negative_zero = integer == "-0"
    value = ("-" if negative_zero else "") + format(int(integer), ",") + (("."+fraction.rstrip("0")) if fraction.rstrip("0") else "")
    return value + " " + {"KRW": "원", "USD": "달러", "EUR": "유로", "JPY": "엔", "percent": "%", "percentage_points": "%p"}.get(quantity["unit"], quantity["unit"] or "단위 미확정")


def render_report(summary, claims):
    esc = lambda value: html.escape(str(value))
    table, sections = [], []
    for stock in summary["stocks"]:
        counts = stock["counts"]
        table.append(f"<tr><td>{esc(stock['code'])}</td><td>{stock['read_documents']}/{stock['target_documents']}</td>"
            f"<td>{counts['claims']}</td><td>{counts['reported_candidates']}</td><td>{counts['outlook_included']}</td>"
            f"<td>{counts['other_or_unclear_scope']}</td><td>{counts['needs_review']}</td><td>{esc(' / '.join(stock['errors']))}</td></tr>")
        reports = []
        for doc in stock["documents"]:
            own = [c for c in claims if c["code"] == stock["code"] and c["receipt_no"] == doc["receipt_no"]]
            groups = []
            for topic, name in TOPIC_NAMES.items():
                candidates = [c for c in own if topic in c["topics"]]
                candidates.sort(key=lambda c: (c["status"] != "reported_candidate", not any(q["metric"] for q in c["measurements"])))
                examples = []
                for claim in candidates[:3]:
                    quantities = []
                    for q in claim["measurements"][:8]:
                        binding = METRIC_NAMES.get(q["metric"], "항목 미확정")
                        quantities.append(f"<tr><td>{esc(q['text'])}</td><td>{esc(display_value(q))}</td>"
                            f"<td>{esc(binding)}</td><td>{'단위 읽음' if q['status']=='supported' else '확인 필요'}"
                            f"{' · 근사·범위 등 확인' if q['qualifiers'] else ''}</td></tr>")
                    period = " / ".join(p["text"] for p in claim["period_expressions"]) or "문장에 명시되지 않음"
                    basis = " / ".join(p["text"] for p in claim["basis_expressions"]) or "문장에 명시되지 않음"
                    background = " / ".join(p["text"] for p in claim["attribution_expressions"]) or "자동으로 찾은 표현 없음"
                    reasons = " / ".join(REASONS[key] for key in claim["review_reasons"])
                    repeated = (f" · 같은 접수번호에서 동일 항목·수치 조합 {claim['possible_repeat_count']}회 반복 가능"
                                if claim.get("possible_repeat_count", 1) > 1 else "")
                    source = f"sources/{stock['code']}_{doc['receipt_no']}.html#{claim['block_id']}"
                    examples.append(f"<article><p>{esc(SCOPES[claim['scope']])} · {esc(TIMES[claim['time_orientation']])} · "
                        f"{'보고 표현 후보' if claim['status']=='reported_candidate' else '확인 필요'}</p>"
                        f"<blockquote>{esc(claim['text'])}</blockquote><p>문장 속 기간 표현: {esc(period)}<br>문장 속 연결·별도 표현: {esc(basis)}<br>"
                        f"회사가 제시한 배경의 단서: {esc(background)} (원인 입증 아님)</p>"
                        f"<table><tr><th>원문 수치</th><th>단위 정리</th><th>연결 항목 후보</th><th>상태</th></tr>{''.join(quantities)}</table>"
                        f"<p>{esc(reasons+repeated)} · <a href='{esc(source)}'>앞뒤 원문 문단 보기</a></p></article>")
                groups.append(f"<h4>{name} · {len(candidates)}후보 중 최대 3개</h4>{''.join(examples)}")
            opened = " open" if doc["receipt_no"] == stock.get("selected_receipt_no") else ""
            reports.append(f"<details{opened}><summary>{esc(doc['report_name'])} · 접수일 {esc(doc['receipt_date'])}</summary>"
                f"<p>사용 가능: {esc(doc['available_from'])} · 보고서 회계 기준: {esc(doc['period_end'])}<br>"
                "회계 기준을 모든 수치의 기간으로 자동 대입하지 않습니다.</p>"+"".join(groups)+"</details>")
        active = "" if stock.get("selected_receipt_no") else "<p>기준일의 최신 원문을 확인하지 못했습니다. 이전 보고서로 대신 채우지 않았습니다.</p>"
        sections.append(f"<section><h2>{esc(stock['code'])} · {esc(stock['company_name'])}</h2>{active}{''.join(reports)}</section>")
    counts = summary["counts"]
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>사업 설명의 근거와 수치</title><style>body{{font-family:system-ui;line-height:1.8;max-width:1200px;margin:30px auto;padding:20px;color:#17263c}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left;font-size:14px}}blockquote{{white-space:pre-wrap;margin:12px 0;padding:15px;background:#f2f6fb}}details{{border:1px solid #ddd;border-radius:8px;padding:15px;margin:18px 0}}article{{padding:12px 0;border-top:1px solid #eee}}.note{{background:#fff3d5;padding:18px}}</style>
<h1>사업 설명의 근거와 수치</h1><p class="note">사업보고서에서 매출·이익, 수주, 투자, 비용, 위험요인에 관한 표현을 규칙으로 정리한 후보입니다. 회사가 적은 설명의 사실 여부나 주가 상승 원인을 확인한 결과는 아닙니다. 해석이 불명확하면 확인 대상으로 남깁니다.</p>
<p>자료 기준일: {esc(summary['reference_date'])} · 처리 시각: {esc(summary['created_at'])}</p>
<h2>구조화 상태</h2><table><tr><th>종목</th><th>원문 읽기</th><th>주장 후보</th><th>회사 보고 표현 후보</th><th>전망 포함</th><th>다른 범위·불명확</th><th>확인 필요</th><th>확인할 내용</th></tr>{''.join(table)}</table>
<p>열은 중복될 수 있습니다. 단위를 읽은 수치 {counts['supported_measurements']}개 중 항목에 연결한 후보 {counts['bound_measurements']}개입니다. 이는 추출 정확도나 전체 항목의 완전성을 검증한 숫자가 아닙니다. 후보와 숫자는 원문 접수번호·문단·문자 범위·해시로 추적합니다.</p>
<p>동일 접수번호에서 항목·수치 조합이 반복될 가능성을 표시한 후보 {counts['possible_repeated_claims']}개가 있습니다. 표현이 반복됐다고 독립적인 근거가 늘어나는 것은 아니며, 같은 숫자가 반드시 같은 뜻이라는 의미도 아닙니다. 다른 보고서 사이의 모든 중복을 검증하지는 않았습니다.</p>
<p>돈은 명시된 통화의 단위로만 정리하고 환산하지 않습니다. 퍼센트와 퍼센트포인트는 다르게 보관합니다. 근사치·범위·단위 누락·여러 항목이나 기간이 섞인 숫자는 확정하지 않습니다. 부호는 원문 표기를 따르며 괄호나 ‘손실’이라는 단어만으로 음수를 만들지 않습니다.</p>
<p>연결·별도, 회사·다른 회사·계열사·부문의 구분은 표현에 따른 후보이며 모든 기업 관계를 확인하지 않았습니다. 표의 병합셀·주석 단위를 문장에 가져와 숫자를 추정하지 않습니다. 보지 않은 원문과 사람의 정답으로 추출 성능을 검증하는 작업은 남아 있습니다.</p>
{''.join(sections)}
<h2>이번 기능의 범위</h2><p>표 {summary['skipped']['tables_not_interpreted']}개는 문장으로 해석하지 않았고 긴 문장 {summary['skipped']['long_sentences_not_structured']}개는 자동 구조화하지 않았습니다. 화면에는 항목별 최대 3문장, 문장마다 최대 8개 수치를 표시합니다. 전체 후보와 세부 이유는 claims.jsonl에 보관합니다. 숫자를 금액 수준·변화액·기간·연결 범위로 모두 정확히 이해한 결과는 아닙니다.</p>
<p>공개 시점과 원문 이력은 보관하지만 가격 결과로 후보를 고르지 않습니다. 기존 0.9 모델·OBSERVE 고정 모델·예측 기록은 수정하지 않고, 이번 후보를 예측 입력으로 넣거나 재학습·모의매매·실제 주문하지 않았습니다. 이미 본 자료의 개발용 추출이며 새 미사용 검증이 아닙니다.</p></html>"""


def run(source, root, *, progress=print):
    evidence = read_json(source / "evidence.json")
    stocks = evidence.get("stocks", [])
    if evidence.get("schema_version") != 1 or not 1 <= len(stocks) <= 20:
        raise CollectionError("기존 BUSINESS.cmd의 사업보고서 결과가 필요합니다.")
    reference = parse_date(evidence["reference_date"])
    summary = {"schema_version": 1, "created_at": datetime.now(SEOUL).isoformat(), "reference_date": reference.isoformat(),
        "source_business": str(source.resolve()), "source_evidence_sha256": hashlib.sha256((source / "evidence.json").read_bytes()).hexdigest(),
        "policy": POLICY, "status": "partial", "stocks": [], "externally_verified": False,
        "model_inputs_modified": False, "training_executed": False, "live_trading_allowed": False,
        "fresh_unseen_evidence": False, "skipped": {"tables_not_interpreted": 0, "long_sentences_not_structured": 0}}
    codes = [s["code"] for s in stocks]
    if len(set(codes)) != len(codes) or any(not re.fullmatch(r"[0-9]{6}", code) for code in codes):
        raise CollectionError("사업보고서 종목 목록이 올바르지 않거나 중복됐습니다.")
    parent = root / "facts"; parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    target = parent / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    all_claims = []
    try:
        (staging / "sources").mkdir()
        for stock in stocks:
            code = stock["code"]
            progress(f"종목 {code} · 근거 문장과 수치 정리 중...")
            item = {"code": code, "company_name": stock["company_name"], "target_documents": len(stock["events"]),
                    "read_documents": 0, "documents": [], "errors": []}
            receipts = [e["receipt_no"] for e in stock["events"]]
            if len(receipts) > 30 or len(set(receipts)) != len(receipts):
                raise CollectionError("원문 접수번호가 중복됐거나 수 한도를 넘었습니다.")
            events, own = [], []
            for original in stock["events"]:
                event = dict(original)
                try:
                    if (not re.fullmatch(r"[0-9]{14}", event["receipt_no"]) or event["status"] != "read"
                            or event.get("company_identity_verified") is not True or event.get("document_kind_verified") is not True
                            or report_period(event["report_name"]).isoformat() != event["period_end"]
                            or event["period_end"] >= event["receipt_date"]):
                        raise CollectionError("기업·보고서·회계기간을 확인한 원문이 필요합니다.")
                    raw, blocks = verified_blocks(source, code, event)
                    claims, skipped = extract_claims(blocks, event, code, stock["company_name"])
                    own.extend(claims)
                    for key, value in skipped.items():
                        summary["skipped"][key] += value
                    key = code+"_"+event["receipt_no"]
                    (staging / "sources" / (key+".json")).write_bytes(raw)
                    (staging / "sources" / (key+".html")).write_text(source_html(blocks, event["report_name"]), encoding="utf-8")
                    item["documents"].append({k: event[k] for k in ("receipt_no", "receipt_date", "available_from", "period_end", "report_name", "source_blocks_sha256")})
                    item["read_documents"] += 1
                except Exception as exc:
                    event["status"] = "unavailable"
                    item["errors"].append(f"{event.get('receipt_no','원문')}: "+source_reason(exc))
                events.append(event)
            selected = evidence_as_of(events, reference)
            item["selected_receipt_no"] = selected.get("receipt_no") if selected["status"] == "available" else None
            item["selection_status"] = selected["status"]
            mark_possible_repeats(own)
            item["counts"] = tally(own)
            all_claims.extend(own)
            summary["stocks"].append(item)
        summary["counts"] = tally(all_claims)
        summary["read_documents"] = sum(s["read_documents"] for s in summary["stocks"])
        summary["target_documents"] = sum(s["target_documents"] for s in summary["stocks"])
        if all(s["read_documents"] == s["target_documents"] > 0 and not s["errors"] and s["selection_status"] == "available" for s in summary["stocks"]):
            summary["status"] = "structured"
        with (staging / "claims.jsonl").open("w", encoding="utf-8") as stream:
            for claim in all_claims:
                stream.write(json.dumps(claim, ensure_ascii=False, allow_nan=False)+"\n")
        summary["claims_sha256"] = hashlib.sha256((staging / "claims.jsonl").read_bytes()).hexdigest()
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        (staging / "report.html").write_text(render_report(summary, all_claims), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, summary


def main(argv=None):
    default = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="사업 설명의 근거와 수치 정리")
    parser.add_argument("--root", type=Path, default=default)
    parser.add_argument("--input", type=Path, help="기존 BUSINESS 결과; 기본은 최신 결과")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, summary = run(args.input or latest_business(args.root / "business"), args.root)
        print(f"\n원문 구조화: {summary['read_documents']}/{summary['target_documents']}건 · 주장 후보 {summary['counts']['claims']}개")
        print(f"회사 보고 표현 후보 {summary['counts']['reported_candidates']}개 · 확인 필요 {summary['counts']['needs_review']}개")
        print(f"결과 보고서: {target / 'report.html'}")
        print("기존 모델·예측은 유지합니다. 사실 검증·재학습·실제 주문은 실행하지 않았습니다.")
        if sys.platform == "win32" and not args.no_open:
            webbrowser.open((target / "report.html").resolve().as_uri())
        return 0 if summary["status"] == "structured" else 2
    except CollectionError as exc:
        print(f"구조화 확인 필요: {exc}", file=sys.stderr)
    except FileNotFoundError:
        print("기존 사업보고서 파일을 찾지 못했습니다. 원래 저장 폴더를 확인해주세요. 인증키는 필요하지 않습니다.", file=sys.stderr)
    except KeyboardInterrupt:
        return 130
    except Exception:
        print("근거 문장과 수치 정리를 완료하지 못했습니다. 마지막 메시지를 알려주세요. 기존 자료는 바꾸지 않았습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
