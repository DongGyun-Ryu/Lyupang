"""Collect source-linked business-report evidence alongside the frozen forecast journal."""
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
from datetime import datetime, timedelta
from pathlib import Path

from business_text import (CATEGORIES, cached_document, evidence_as_of, extract_blocks, extract_evidence,
                           main_document, source_html)
from dart_client import DartClient, DartError, is_annual_report, stored_key
from financial_history import report_period
from lyupang import CollectionError, SEOUL, parse_date
from research import safe_reason
from train import latest_research


def render_report(summary):
    esc = lambda v: html.escape(str(v))
    rows, sections = [], []
    reasons = {"no_previous_annual_report": "기준일 전에 공개된 사업보고서가 없습니다.",
               "ambiguous_same_day_reports": "같은 날 여러 보고서가 있어 하나로 특정하지 않았습니다.",
               "latest_report_unreadable": "최신 대상 원문을 읽지 못했습니다. 이전 원문으로 대신 채우지 않았습니다."}
    for stock in summary["stocks"]:
        selected = stock["selected"]
        event = selected.get("event")
        counts = [event["evidence"][key]["shown_count"] if event else 0 for key in CATEGORIES]
        warning = stock.get("error") or reasons.get(selected.get("reason"), "")
        if selected.get("reason") == "latest_report_unreadable":
            failed = next((e for e in stock["events"] if e["receipt_no"] == selected.get("receipt_no")), {})
            warning += " " + failed.get("error", "")
        if event and any(count == 0 for count in counts):
            warning = "자동으로 찾지 못한 항목이 있습니다. 해당 원문에서 확인해주세요."
        rows.append(f"<tr><td>{esc(stock['code'])}</td><td>{stock['read_reports']} / {stock['target_reports']}</td>"
                    + "".join(f"<td>{n}</td>" for n in counts) + f"<td>{esc(warning)}</td></tr>")
        excerpts = []
        if event:
            for category, label in CATEGORIES.items():
                evidence = event["evidence"][category]
                quotes = []
                for quote in evidence["quotes"]:
                    flags = []
                    if quote["forward_looking_language"]:
                        flags.append("전망·계획 표현 포함")
                    if quote["is_partial_block"]:
                        flags.append("문단·표의 일부")
                    if quote["kind"] == "table":
                        flags.append("표 평문 — 병합셀·단위 자동 해석 없음")
                    link = event["source_html"] + "#" + quote["block_id"]
                    quotes.append(f"<article><p class='path'>{esc(' / '.join(quote['headings']))}</p>"
                                  f"<blockquote>{esc(quote['quote'])}</blockquote><p>{esc(' · '.join(flags))} "
                                  f"<a href='{esc(link)}'>추출 본문의 해당 문단 보기</a></p></article>")
                message = "자동으로 찾은 설명이 없습니다. 원문에 해당 내용이 없다는 뜻은 아닙니다." if not quotes else ""
                excerpts.append(f"<h3>{esc(label)}</h3><p>후보 {evidence['candidate_count']}개 중 원문 순서로 "
                                f"최대 5개 표시 · 표시 {evidence['shown_count']}개</p><p>{esc(message)}</p>{''.join(quotes)}")
            introduction = (f"<p>{esc(event['report_name'])} · 접수일 {esc(event['receipt_date'])} · "
                            f"사용 가능일 {esc(event['available_from'])}<br>접수번호 {esc(event['receipt_no'])} · "
                            f"<a href='{esc(event['url'])}'>DART 공시 원문 열기</a></p>")
        else:
            introduction = f"<p>{esc(warning)}</p>"
        timeline = "".join(f"<tr><td>{esc(e['receipt_date'])}</td><td>{esc(e['report_name'])}</td>"
                           f"<td>{'읽기 성공' if e['status'] == 'read' else '확인 필요'}</td><td>{esc(e.get('error', ''))}</td></tr>"
                           for e in stock["events"])
        sections.append(f"<section><h2>{esc(stock['code'])} · {esc(stock['company_name'])}</h2>{introduction}{''.join(excerpts)}"
                        f"<details><summary>과거 원문과 정정 기록</summary><table><tr><th>접수일</th><th>보고서</th><th>읽기</th><th>확인할 내용</th></tr>{timeline}</table>"
                        "<p>각 접수번호의 전체 추출 본문과 후보 설명은 sources 폴더 및 evidence.json에 보관합니다.</p></details></section>")
    return f"""<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Lyupang 사업보고서 근거 모음</title><style>body{{max-width:1100px;margin:35px auto;padding:0 20px;font-family:system-ui,sans-serif;line-height:1.8;color:#17263c}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:10px;text-align:left;font-size:14px}}section{{margin:35px 0}}blockquote{{white-space:pre-wrap;margin:10px 0;padding:15px;background:#f4f6f9;border-left:3px solid #8aa0bc}}.note{{background:#fff2d5;padding:18px;border-radius:10px}}.path{{font-size:14px;color:#53647b}}article{{margin:20px 0}}</style>
<h1>사업보고서 근거 모음</h1><p class='note'>회사가 사업보고서에서 밝힌 설명을 출처와 함께 모았습니다. 제목·키워드로 고른 원문 발췌이며, 설명의 사실 여부나 주가 상승 원인을 입증한 결과는 아닙니다.</p>
<p>자료 기준일: {esc(summary['reference_date'])} · 수집 시각: {esc(summary['created_at'])}</p>
<h2>자료 수집 상태</h2><p>원문 읽기 성공: {summary['read_reports']} / {summary['target_reports']}건</p>
<table><tr><th>종목</th><th>읽기 성공 원문/대상</th><th>사업 개요</th><th>제품·서비스</th><th>실적 설명</th><th>위험요인</th><th>확인할 내용</th></tr>{''.join(rows)}</table>
<p>각 설명의 숫자는 기준일에 선택한 보고서에서 표시한 발췌 수입니다. 회계기간은 DART 목록의 사업보고서 제목을 기준으로 구분합니다. 접수번호별 본문의 기업번호·문서 종류를 확인하며, 접수일 다음 날 이후에만 사용할 수 있게 기록합니다.</p>
{''.join(sections)}
<h2>이번 기능의 범위</h2><p>실적 변화에 관한 회사 설명과 전망·계획이 섞여 있을 수 있습니다. 자동 발췌는 모든 설명을 찾거나 의미를 정확히 이해한다고 보장하지 않습니다. 표 전체의 병합셀·단위·범위를 해석해 수치를 재계산하지 않습니다.</p>
<p>정정은 그 접수일 이후에만 선택하며 최신 원문을 못 읽거나 같은 날 여러 대상이 있으면 이전 설명으로 대신 채우지 않습니다. 읽지 못한 문서는 해당 이유를 표시합니다. 보관된 ZIP·XML 해시와 추출 문단 위치로 원문을 대조할 수 있습니다.</p>
<p>기존 고정 모델·예측 기록은 바꾸지 않습니다. 이번 본문을 예측 입력에 추가하거나 재학습·뉴스 수집·모의매매·실제 주문을 실행하지 않았습니다. 추후 본문 기반 모델은 이미 확인한 과거 평가 기간을 새 검증으로 재사용할 수 없습니다.</p></html>"""


def run(source, output_root, cache, client, *, progress=print):
    batch_bytes = (source / "batch.json").read_bytes()
    batch = json.loads(batch_bytes)
    stocks = batch.get("stocks", [])
    if batch.get("schema_version") != 1 or not 1 <= len(stocks) <= 20:
        raise CollectionError("기존 RESEARCH.cmd의 종목 목록을 확인하지 못했습니다.")
    codes = [s["code"] for s in stocks]
    if len(set(codes)) != len(codes) or any(not re.fullmatch(r"[0-9]{6}", c) for c in codes):
        raise CollectionError("기존 연구 결과의 종목코드가 올바르지 않거나 중복됐습니다.")
    start, end = parse_date(batch["requested_start"]), parse_date(batch["requested_end"])
    if start > end or end >= datetime.now(SEOUL).date():
        raise CollectionError("기존 연구 결과의 기간이 올바르지 않습니다.")
    summary = {"schema_version": 1, "created_at": datetime.now(SEOUL).isoformat(), "source_research": str(source.resolve()),
               "source_batch_sha256": hashlib.sha256(batch_bytes).hexdigest(), "reference_date": end.isoformat(),
               "status": "partial", "stocks": [], "read_reports": 0, "target_reports": 0,
               "extraction_method": "heading_and_keyword_exact_substrings_v1", "causal_claim_verified": False,
               "model_inputs_modified": False, "training_executed": False, "live_trading_allowed": False}
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=output_root))
    target = output_root / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    try:
        (staging / "sources").mkdir()
        companies, setup_error = {}, None
        try:
            companies = client.companies(codes)
        except Exception as exc:
            setup_error = safe_reason(exc)
        for code in codes:
            item = {"code": code, "company_name": code, "events": [], "read_reports": 0, "target_reports": 0,
                    "selected": {"status": "unavailable", "reason": "no_previous_annual_report"}}
            progress(f"종목 {code} · 사업보고서 설명 수집 중...")
            try:
                if setup_error:
                    raise CollectionError(setup_error)
                company = companies.get(code)
                if not company:
                    raise CollectionError("DART 기업번호를 찾지 못했습니다.")
                corp = company["corp_code"]
                item["company_name"] = company["corp_name"]
                client.company_identifiers(corp, code)  # Verify live corporation-to-stock identity.
                filings = client.disclosures(corp, start - timedelta(days=550), end, annual_only=True)
                filings = [f for f in filings if is_annual_report(f.get("report_name"))]
                if len(filings) > 30:
                    raise CollectionError("종목당 사업보고서 본문 30건 한도를 넘었습니다.")
                for filing in filings:
                    period, received = report_period(filing["report_name"]), parse_date(filing["receipt_date"])
                    if not period < received or not start - timedelta(days=550) <= received <= end:
                        raise CollectionError("사업보고서의 회계기간·접수일이 올바르지 않습니다.")
                    receipt = filing["receipt_no"]
                    if not re.fullmatch(r"[0-9]{14}", receipt):
                        raise CollectionError("사업보고서 접수번호가 올바르지 않습니다.")
                    item["events"].append({"receipt_no": receipt, "receipt_date": received.isoformat(),
                        "report_name": filing["report_name"], "period_end": period.isoformat(), "period_source": "DART_report_title",
                        "available_from": (received + timedelta(days=1)).isoformat() + "T00:00:00+09:00",
                        "url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + receipt, "status": "unavailable"})
                item["target_reports"] = len(item["events"])
                stop_error = None
                for index, event in enumerate(item["events"], 1):
                    if stop_error:
                        event["error"] = "앞선 연결·인증 오류로 조회 대기: " + stop_error
                        continue
                    progress(f"  {code} 본문 {index}/{item['target_reports']} · 접수일 {event['receipt_date']}")
                    try:
                        raw, meta, folder = cached_document(cache, code, corp, event["receipt_no"], client)
                        member, xml, document = main_document(raw, event["receipt_no"], corp)
                        blocks = extract_blocks(document)
                        evidence = extract_evidence(blocks)
                        stem = f"{code}_{event['receipt_no']}"
                        blocks_content = json.dumps(blocks, ensure_ascii=False, indent=2).encode("utf-8")
                        (staging / "sources" / (stem + ".json")).write_bytes(blocks_content)
                        (staging / "sources" / (stem + ".html")).write_text(source_html(blocks, event["report_name"]), encoding="utf-8")
                        event.update(status="read", evidence=evidence, company_identity_verified=True, document_kind_verified=True,
                            archive_sha256=meta["sha256"], archive_path=str(folder.resolve()), retrieved_at=meta["retrieved_at"],
                            xml_member=member, xml_sha256=hashlib.sha256(xml).hexdigest(), block_count=len(blocks),
                            source_blocks=f"sources/{stem}.json", source_blocks_sha256=hashlib.sha256(blocks_content).hexdigest(),
                            source_html=f"sources/{stem}.html")
                        item["read_reports"] += 1
                    except Exception as exc:
                        event["error"] = safe_reason(exc)
                        progress(f"  본문 확인 필요: {event['error']}")
                        if isinstance(exc, DartError) and (exc.code in {"010", "011", "012", "020", "800", "901", "network"}
                                                          or str(exc.code).startswith("HTTP_")):
                            stop_error = event["error"]
                item["selected"] = evidence_as_of(item["events"], end)
            except Exception as exc:
                item["error"] = safe_reason(exc)
                progress(f"  {code} 확인 필요: {item['error']}")
            summary["stocks"].append(item)
            progress(f"  본문 읽기 성공 {item['read_reports']}/{item['target_reports']}건")
        summary["read_reports"] = sum(s["read_reports"] for s in summary["stocks"])
        summary["target_reports"] = sum(s["target_reports"] for s in summary["stocks"])
        if all(s["target_reports"] > 0 and s["read_reports"] == s["target_reports"] and s["selected"]["status"] == "available"
               and all(s["selected"]["event"]["evidence"][k]["shown_count"] > 0 for k in CATEGORIES)
               and not s.get("error") for s in summary["stocks"]):
            summary["status"] = "collected"
        (staging / "evidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (staging / "report.html").write_text(render_report(summary), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, summary


def main(argv=None):
    root = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="사업보고서의 사업·실적 설명을 원문 근거와 함께 수집")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--input", type=Path, help="기존 research 실행 폴더; 생략하면 최신 결과")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = args.input or latest_research(args.root / "research")
        key = stored_key()
        if not key:
            raise CollectionError("기존 DART 키를 찾지 못했습니다. DART_SETUP.cmd로 확인해주세요.")
        target, summary = run(source, args.root / "business", args.root / "business_documents", DartClient(key))
        print(f"\n사업보고서 본문 읽기: {summary['read_reports']}/{summary['target_reports']}건")
        print("사업·실적 설명 수집 완료" if summary["status"] == "collected" else "일부 설명 수집 · 확인할 항목이 있습니다.")
        print(f"결과 보고서: {target / 'report.html'}")
        print("기존 모델·예측 기록은 유지합니다. 재학습과 실제 주문은 실행하지 않았습니다.")
        if sys.platform == "win32" and not args.no_open:
            webbrowser.open((target / "report.html").resolve().as_uri())
        return 0 if summary["status"] == "collected" else 2
    except CollectionError as exc:
        print(f"사업보고서 확인 필요: {exc}", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print("수집을 취소했습니다. 기존 자료는 유지합니다.", file=sys.stderr)
        return 130
    except Exception:
        print("사업보고서 수집을 완료하지 못했습니다. 실행 창의 마지막 메시지를 알려주세요.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
