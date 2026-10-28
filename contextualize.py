"""Read-only statement context and post-publication price comparison."""
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
from evidence_context import POLICY, annotate, counts, mark_overlaps, release_outcomes
from financial_history import report_period
from lyupang import CollectionError, SEOUL, parse_date
from research_data import load_snapshot


def safe_reason(exc):
    if isinstance(exc, CollectionError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return "기존 연구 결과 또는 가격 파일을 찾지 못했습니다. 저장된 폴더 위치를 확인해주세요."
    if isinstance(exc, PermissionError):
        return "저장된 자료를 읽을 권한을 확인해주세요."
    return "자료의 형식 또는 저장 상태를 확인해야 합니다."


def read_json(path):
    def invalid(_):
        raise CollectionError("자료에 유효하지 않은 수치가 있습니다.")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def latest_business(root):
    candidates = []
    if root.exists():
        for folder in root.iterdir():
            if not folder.is_dir() or folder.is_symlink() or folder.name.startswith("."):
                continue
            try:
                captured = datetime.fromisoformat(read_json(folder / "evidence.json")["created_at"])
                if captured.tzinfo is not None:
                    candidates.append((captured, folder))
            except (OSError, ValueError, KeyError, TypeError, CollectionError):
                continue
    if not candidates:
        raise CollectionError("먼저 BUSINESS.cmd를 실행해주세요.")
    return max(candidates, key=lambda pair: pair[0])[1]


def verified_blocks(source, code, event):
    expected = f"sources/{code}_{event['receipt_no']}.json"
    if event.get("source_blocks") != expected:
        raise CollectionError("본문 출처의 파일 위치가 지원 형식과 다릅니다.")
    content = (source / expected).read_bytes()
    if len(content) > 64 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != event.get("source_blocks_sha256"):
        raise CollectionError("사업보고서 추출 본문의 크기 또는 해시가 일치하지 않습니다.")
    blocks = json.loads(content)
    if not isinstance(blocks, list) or len(blocks) != event.get("block_count") or len(blocks) > 100_000:
        raise CollectionError("본문 문단 수가 저장 기록과 다릅니다.")
    seen = set()
    for block in blocks:
        if (not re.fullmatch(r"b[0-9]{6}", block["id"]) or block["id"] in seen
                or block["kind"] not in {"paragraph", "heading", "table"}
                or not isinstance(block["text"], str) or not isinstance(block["headings"], list)
                or any(not isinstance(h, str) for h in block["headings"])
                or hashlib.sha256(block["text"].encode()).hexdigest() != block["text_sha256"]):
            raise CollectionError("본문 문단의 식별번호·형식·해시가 올바르지 않습니다.")
        seen.add(block["id"])
    return content, blocks


def render_report(summary, statements, outcomes):
    esc = lambda v: html.escape(str(v))
    status_names = {"pending": "이후 자료 대기", "outside_coverage": "가격 기간 밖",
                    "quality_issue": "가격 품질 확인", "source_error": "가격 파일 확인"}
    subject_names = {"company": "회사", "industry": "산업", "mixed": "혼합", "unclear": "판단 보류"}
    time_names = {"reported": "과거·현재 보고 표현", "outlook": "전망·계획", "mixed": "혼합", "unclear": "판단 보류"}
    term_names = {"accounting_term": "회계 용어", "management_term": "관리·제도 용어", "organization_term": "조직·체계 용어"}
    table, sections = [], []
    for stock in summary["stocks"]:
        stats = stock["counts"]
        stock_prices = [r for r in outcomes if r["code"] == stock["code"]]
        observed = sum(r["status"] == "observed" for r in stock_prices)
        table.append(f"<tr><td>{esc(stock['code'])}</td><td>{stock['read_documents']} / {stock['target_documents']}</td>"
                     f"<td>{stats['sentences']}</td><td>{stats['company_reported_performance']}</td><td>{stats['industry']}</td>"
                     f"<td>{stats['outlook_included']}</td><td>{stats['needs_review']}</td><td>{stats['technical_term_review']}</td><td>{observed} / {len(stock_prices)}</td>"
                     f"<td>{esc(' / '.join(stock['errors']))}</td></tr>")
        documents = []
        for doc in stock["documents"]:
            prices = [r for r in stock_prices if r["receipt_no"] == doc["receipt_no"]]
            rows = []
            for price in prices:
                result = f"{price['price_change_pct']:+.2f}%" if price["status"] == "observed" else status_names[price["status"]]
                flags = " · 같은 날 여러 보고서" if len(price.get("same_day_receipts", [])) > 1 else ""
                if price.get("overlaps_other_receipt"):
                    flags += " · 다른 보고서와 구간 겹침"
                rows.append(f"<tr><td>{price['horizon']}일봉</td><td>{esc(price.get('entry_date') or '—')}</td>"
                            f"<td>{esc(price.get('exit_date') or '—')}</td><td>{esc(result + flags)}</td></tr>")
            selected = [s for s in statements if s["release_id"] == doc["release_id"]]
            groups = (
                ("회사 실적 · 보고 표현 후보", lambda c: c["subject"] == "company" and c["time_orientation"] == "reported" and "performance" in c["topics"] and c["status"] == "rule_based_candidate"),
                ("산업 동향 후보", lambda c: c["subject"] == "industry"),
                ("전망·계획 포함", lambda c: c["time_orientation"] in {"outlook", "mixed"}),
                ("회계·관리 용어 확인", lambda c: bool(c["excluded_expressions"])),
                ("혼합·판단 보류 예시", lambda c: c["status"] == "needs_review"),
            )
            examples = []
            for label, predicate in groups:
                matches = [s for s in selected if predicate(s["classification"])]
                quotes = []
                for statement in matches[:4]:
                    classification = statement["classification"]
                    cue_texts = sorted({m["text"] for matches in classification["matched_expressions"].values() for m in matches})
                    reasons = " / ".join(cue_texts[:10]) or "명확한 표현 부족"
                    if classification["company_heading_fallback"]:
                        reasons += " / 회사 실적 제목 문맥"
                    if classification.get("financial_heading_fallback"):
                        reasons += " / 경영진단·재무 제목과 당기·전기 인식액 문맥"
                    excluded = " / ".join(sorted({f"{m['context_text']} ({term_names[m['reason']]})"
                                                     for m in classification["excluded_expressions"]}))
                    caution = (f"<br>전망 근거에서 제외: {esc(excluded)}. 다른 전망 표현은 별도로 확인하며 이 문장은 판단을 보류합니다."
                               if excluded else "")
                    source_link = doc["source_html"] + "#" + statement["block_id"]
                    quotes.append(f"<article><p class='path'>{esc(' / '.join(statement['headings']))}</p>"
                                  f"<blockquote>{esc(statement['text'][:1500])}{' … (문장 일부)' if len(statement['text']) > 1500 else ''}</blockquote>"
                                  f"<p>{esc(subject_names[classification['subject']])} · {esc(time_names[classification['time_orientation']])}<br>"
                                  f"분류에 사용한 표현: {esc(reasons)}{caution} · <a href='{esc(source_link)}'>원문 문단 보기</a></p></article>")
                examples.append(f"<h4>{esc(label)}</h4><p>{len(matches)}문장 중 원문 순서로 최대 4개 표시</p>{''.join(quotes)}")
            opened = " open" if doc["receipt_no"] == stock.get("selected_receipt_no") else ""
            documents.append(f"<details id='{esc(doc['release_id'])}'{opened}><summary>{esc(doc['report_name'])} · 접수일 {esc(doc['receipt_date'])}</summary>"
                f"<p>사용 가능: {esc(doc['available_from'])} · 회계기간: {esc(doc['period_end'])} · "
                f"<a href='https://dart.fss.or.kr/dsaf001/main.do?rcpNo={esc(doc['receipt_no'])}'>DART 원문</a></p>"
                f"<table><tr><th>관찰 구간</th><th>첫 관찰 일봉</th><th>종료 일봉</th><th>비용 전 가격 변화</th></tr>{''.join(rows)}</table>"
                + "".join(examples) + "</details>")
        sections.append(f"<section><h2>{esc(stock['code'])} · {esc(stock['company_name'])}</h2>{''.join(documents)}</section>")
    tally = summary["outcome_counts"]
    return f"""<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Lyupang 설명 구분과 공개 이후 가격</title><style>body{{max-width:1200px;margin:35px auto;padding:0 20px;font-family:system-ui,sans-serif;line-height:1.8;color:#17263c}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left;font-size:14px}}blockquote{{white-space:pre-wrap;margin:10px 0;padding:15px;background:#f2f5f9}}details{{border:1px solid #d8dfe8;border-radius:8px;padding:15px;margin:20px 0}}summary{{cursor:pointer;font-weight:bold}}.note{{background:#fff2d5;padding:18px;border-radius:10px}}.path{{font-size:13px;color:#52647b}}</style>
<h1>설명 구분과 공개 이후 가격</h1><p class='note'>회사 실적·산업 동향·전망을 표현 규칙으로 구분한 후보입니다. 뜻이 섞이거나 불명확하면 보류합니다. 보고서 공개 뒤의 가격을 연결해 관찰하며 주가 상승 원인을 입증하거나 매매 성적을 계산한 결과는 아닙니다.</p>
<p>자료 기준일: {esc(summary['reference_date'])} · 처리 시각: {esc(summary['created_at'])}</p>
<h2>문장 분류·가격 연결 상태</h2><table><tr><th>종목</th><th>원문 처리/대상</th><th>문장</th><th>회사 실적 후보</th><th>산업 후보</th><th>전망 포함</th><th>혼합·보류</th><th>용어 확인</th><th>가격 확인/대상</th><th>확인할 내용</th></tr>{''.join(table)}</table>
<p>분류 열은 중복될 수 있습니다. 보고 표현은 실제 성과를 외부에서 검증했다는 뜻이 아닙니다. 본문의 표는 병합셀·단위를 자동 해석하지 않아 문장 분류에서 제외합니다.</p>
<h2>용어 혼동 점검</h2><p>회계·관리 용어가 포함되어 따로 확인할 문장: {summary['counts']['technical_term_review']}개. '기대신용손실' 등의 용어 안에 있는 기대·추정·계획·예측만으로 미래 전망을 판정하지 않습니다. 같은 문장에 별도의 전망·계획이 있으면 그 표현은 유지하고 문장 전체는 보류합니다. 회사의 판매 지역이나 시장 진출 표현만으로 산업 전체 동향을 판정하지 않습니다.</p>
<p>아직 사람이 전체 문장의 정답을 확인한 결과가 아닙니다. 보류가 줄거나 특정 분류 수가 바뀌었다고 정확도가 높아졌다고 판단하지 않습니다.</p>
<h2>가격 자료 상태</h2><p>확인 {tally.get('observed',0)}건 · 이후 자료 대기 {tally.get('pending',0)}건 · 가격 기간 밖 {tally.get('outside_coverage',0)}건 · 품질 확인 {tally.get('quality_issue',0)}건 · 파일 확인 {tally.get('source_error',0)}건</p>
<p>가격 비교 한 건은 보고서 접수번호와 관찰 구간의 조합입니다. 문장마다 같은 수익률을 복제하지 않습니다. 수집 시작 전 공시는 뒤늦은 첫 가격에 연결하지 않고 '가격 기간 밖'으로 표시합니다.</p>
{''.join(sections)}
<h2>연결 기준과 남은 확인</h2><p>접수일 당일은 제외하고 다음 날 00:00 이후 첫 관찰 일봉의 시가부터 5·20·60개 일봉 마지막 종가까지 비교합니다. 정정 본문은 그 정정이 공개된 이후에만 연결하며 이전 문서의 설명을 덮어쓰지 않습니다. 같은 날 여러 보고서나 겹치는 기간은 독립 표본이 아닙니다.</p>
<p>다른 뉴스·시장 전체 영향·이미 알려진 정보와의 차이는 분리하지 않았습니다. 시장 대비 초과수익·인과관계·통계적 유의성을 검증하지 않았습니다. 거래소 달력 누락·특별 개장 시각·거래정지·과거 수정주가 시점은 확인되지 않았으며 비교에는 저장된 조회 시점의 수정주가를 사용합니다. 수수료·세금·슬리피지·체결을 반영한 전략 수익률이 아닙니다.</p>
<p>분류에는 주가 결과를 사용하지 않았습니다. 이번 과거 비교를 보고 수정한 모델은 같은 기간을 새 미사용 검증으로 주장할 수 없습니다. 기존 모델·예측 기록은 그대로 두며 재학습·모의매매·실제 주문은 실행하지 않았습니다.</p></html>"""


def run(source, root, *, progress=print):
    evidence_path = source / "evidence.json"
    business = read_json(evidence_path)
    stocks = business.get("stocks", [])
    if business.get("schema_version") != 1 or not 1 <= len(stocks) <= 20:
        raise CollectionError("지원하는 BUSINESS.cmd 결과가 필요합니다.")
    codes = [s["code"] for s in stocks]
    if len(set(codes)) != len(codes) or any(not re.fullmatch(r"[0-9]{6}", c) for c in codes):
        raise CollectionError("사업보고서 결과의 종목이 잘못됐거나 중복됐습니다.")
    parse_date(business["reference_date"])
    summary = {"schema_version": 1, "created_at": datetime.now(SEOUL).isoformat(), "reference_date": business["reference_date"],
        "source_business": str(source.resolve()), "source_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "classification_policy": POLICY, "status": "partial", "stocks": [], "live_trading_allowed": False,
        "model_inputs_modified": False, "training_executed": False, "historical_exploration": True,
        "fresh_unseen_evidence": False, "causal_claim_verified": False, "statistical_significance_established": False}
    price_sources, research_error = {}, None
    try:
        research = Path(business["source_research"])
        batch_bytes = (research / "batch.json").read_bytes()
        if hashlib.sha256(batch_bytes).hexdigest() != business["source_batch_sha256"]:
            raise CollectionError("사업보고서와 가격 연구의 출처 기록이 일치하지 않습니다.")
        batch = json.loads(batch_bytes)
        if batch["requested_end"] != business["reference_date"]:
            raise CollectionError("사업보고서와 가격 연구의 기준일이 다릅니다.")
        price_sources = {s["code"]: s for s in batch["stocks"]}
    except Exception as exc:
        research_error = safe_reason(exc)
    output_root = root / "context"
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=output_root))
    target = output_root / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    statements, outcomes = [], []
    try:
        (staging / "sources").mkdir()
        for stock in stocks:
            code = stock["code"]
            progress(f"종목 {code} · 문장 구분과 공개 이후 가격 연결 중...")
            item = {"code": code, "company_name": stock["company_name"], "documents": [], "errors": [],
                    "target_documents": len(stock["events"]), "read_documents": 0,
                    "selected_receipt_no": stock.get("selected", {}).get("receipt_no")}
            if len(stock["events"]) > 30:
                raise CollectionError("종목당 문서 수 한도를 넘었습니다.")
            receipts = [event["receipt_no"] for event in stock["events"]]
            if len(receipts) != len(set(receipts)):
                raise CollectionError("같은 종목의 접수번호가 중복됐습니다.")
            metadata, bars, price_error = None, [], research_error
            if not price_error:
                try:
                    snapshot = Path(price_sources[code]["source_snapshot"])
                    metadata, bars = load_snapshot(snapshot)
                    if metadata["requested_start"] != batch["requested_start"] or metadata["requested_end"] != batch["requested_end"]:
                        raise CollectionError("가격 스냅샷의 요청 기간이 연구 기록과 다릅니다.")
                    if (metadata["code"] != code or metadata.get("adjusted") is not True
                            or metadata.get("adjustment_base_date") != metadata["requested_end"]):
                        raise CollectionError("가격 스냅샷의 종목 또는 수정주가 기준이 다릅니다.")
                    item["price_source"] = {"snapshot": str(snapshot.resolve()), "csv_sha256": metadata["csv_sha256"]}
                except Exception as exc:
                    price_error = safe_reason(exc)
            if price_error:
                item["errors"].append("가격 연결: " + price_error)
            own_statements = []
            for event in stock["events"]:
                try:
                    if (not re.fullmatch(r"[0-9]{14}", event["receipt_no"]) or event["status"] != "read"
                            or event.get("company_identity_verified") is not True or event.get("document_kind_verified") is not True
                            or report_period(event["report_name"]).isoformat() != event["period_end"]
                            or event["period_end"] >= event["receipt_date"]):
                        raise CollectionError("기업·보고서 종류·회계기간을 확인한 원문이 필요합니다.")
                    content, blocks = verified_blocks(source, code, event)
                    annotated, skipped = annotate(blocks, event, code, stock["company_name"])
                    own_statements.extend(annotated)
                    stem = f"{code}_{event['receipt_no']}"
                    (staging / "sources" / (stem + ".json")).write_bytes(content)
                    (staging / "sources" / (stem + ".html")).write_text(source_html(blocks, event["report_name"]), encoding="utf-8")
                    doc = {key: event[key] for key in ("receipt_no", "receipt_date", "available_from", "period_end", "report_name", "source_blocks_sha256", "xml_sha256")}
                    doc.update(release_id=stem, source_html=f"sources/{stem}.html", counts=counts(annotated), skipped_tables=skipped)
                    item["documents"].append(doc)
                    item["read_documents"] += 1
                    if price_error:
                        for horizon in (5, 20, 60):
                            outcomes.append({"release_id": stem, "code": code, "receipt_no": event["receipt_no"],
                                "receipt_date": event["receipt_date"], "available_from": event["available_from"], "horizon": horizon,
                                "status": "source_error", "reason": price_error, "price_change_pct": None})
                    else:
                        linked = release_outcomes(event, code, metadata, bars, business["reference_date"])
                        for row in linked:
                            row["price_csv_sha256"] = metadata["csv_sha256"]
                        outcomes.extend(linked)
                except Exception as exc:
                    item["errors"].append(f"{event.get('receipt_no', '원문')}: " + safe_reason(exc))
            item["counts"] = counts(own_statements)
            statements.extend(own_statements)
            summary["stocks"].append(item)
        mark_overlaps(outcomes)
        summary["outcome_counts"] = dict(Counter(r["status"] for r in outcomes))
        summary["counts"] = counts(statements)
        summary["read_documents"] = sum(s["read_documents"] for s in summary["stocks"])
        summary["target_documents"] = sum(s["target_documents"] for s in summary["stocks"])
        if all(s["read_documents"] == s["target_documents"] > 0 and not s["errors"] for s in summary["stocks"]):
            summary["status"] = "linked"
        with (staging / "statements.jsonl").open("w", encoding="utf-8") as stream:
            for statement in statements:
                stream.write(json.dumps(statement, ensure_ascii=False, allow_nan=False) + "\n")
        (staging / "release_outcomes.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        for name, key in (("statements.jsonl", "statement_sha256"), ("release_outcomes.json", "outcome_sha256")):
            summary[key] = hashlib.sha256((staging / name).read_bytes()).hexdigest()
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (staging / "report.html").write_text(render_report(summary, statements, outcomes), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, summary


def main(argv=None):
    root = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="사업보고서 설명 구분과 공개 이후 가격 연결")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--input", type=Path, help="기존 business 실행 폴더; 생략하면 최신 결과")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, summary = run(args.input or latest_business(args.root / "business"), args.root)
        print(f"\n문장 구분 원문: {summary['read_documents']}/{summary['target_documents']}건 · {summary['counts']['sentences']}문장")
        print(f"가격 자료 상태: {summary['outcome_counts']}")
        print("설명 구분·가격 연결 완료" if summary["status"] == "linked" else "일부 처리 · 보고서의 확인할 내용을 봐주세요.")
        print(f"결과 보고서: {target / 'report.html'}")
        print("기존 모델·예측 기록은 유지합니다. 재학습과 실제 주문은 실행하지 않았습니다.")
        if sys.platform == "win32" and not args.no_open:
            webbrowser.open((target / "report.html").resolve().as_uri())
        return 0 if summary["status"] == "linked" else 2
    except CollectionError as exc:
        print(f"자료 연결 확인 필요: {exc}", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print("작업을 취소했습니다.", file=sys.stderr)
        return 130
    except Exception:
        print("설명 구분과 가격 연결을 완료하지 못했습니다. 실행 창의 마지막 메시지를 알려주세요.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
