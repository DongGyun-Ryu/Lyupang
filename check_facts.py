"""Offline audit of stored claims against sources and explicit development criteria."""
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
from decimal import Decimal, InvalidOperation
from pathlib import Path

from business_text import source_html
from facts import METRIC_NAMES, SCOPES, TIMES, display_value, tally, run as refresh_facts
from report_claims import POLICY
from financial_history import report_period
from lyupang import CollectionError, SEOUL, parse_date

REFERENCE_SHA256 = "31756ae8bc2c5cc75572b7705b932d116c163185d359e39ac8ff445e52a8274d"
REFERENCE = Path(__file__).with_name("claim_audit_reference.json")
STATUSES = {"matched": "기준 부합", "mismatch": "불일치", "missing": "자료 없음",
            "source_mismatch": "기준 원문과 다름", "ambiguous": "복수 후보"}
SPAN_FIELDS = ("measurements", "metrics", "period_expressions", "change_expressions", "basis_expressions",
               "adjustment_expressions", "conditional_expressions", "attribution_expressions",
               "additional_plan_expressions", "related_scope_expressions")


def require(condition, message="저장된 근거 자료의 형식·해시·원문 위치가 일치하지 않습니다."):
    if not condition:
        raise CollectionError(message)


def content(path, maximum=32 * 1024 * 1024):
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= maximum,
            "검증할 파일이 없거나 지원 크기·저장 형식과 다릅니다.")
    return path.read_bytes()


def decoded(raw):
    def invalid(_):
        raise CollectionError("검증 자료에 유효하지 않은 숫자가 있습니다.")
    return json.loads(raw, parse_constant=invalid)


def latest_facts(root):
    # Do not silently fall back to an older successful report after a bad latest run.
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                     and re.fullmatch(r"[0-9]{8}_[0-9]{6}_[a-f0-9]{8}", p.name)) if root.exists() else []
    require(bool(folders), "기존 FACTS.cmd 결과를 찾지 못했습니다. 원래 자료 폴더를 확인해주세요.")
    return folders[-1]


def valid_span(text, item):
    start, end = item["start"], item["end"]
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(text))
    require(text[start:end] == item["text"])


def number(value):
    require(isinstance(value, str) and len(value) <= 100)
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise CollectionError("저장된 수치의 형식이 올바르지 않습니다.") from None
    require(result.is_finite() and abs(result) <= Decimal("1e80"))
    return result


def load_facts(source):
    raw_summary = content(source / "summary.json")
    summary = decoded(raw_summary)
    require(summary["schema_version"] == 1 and summary["policy"] in {
        "source_statement_measurements_v1", "source_statement_measurements_v2", "source_statement_measurements_v3"})
    require(summary.get("training_executed") is False and summary.get("model_inputs_modified") is False
            and summary.get("live_trading_allowed") is False)
    raw_claims = content(source / "claims.jsonl")
    require(hashlib.sha256(raw_claims).hexdigest() == summary["claims_sha256"])
    claims = [decoded(line) for line in raw_claims.splitlines() if line.strip()]
    require(len(claims) <= 100_000 and 1 <= len(summary["stocks"]) <= 20)
    documents, source_files, codes = {}, {}, set()
    total_bytes = 0
    for stock in summary["stocks"]:
        code = stock["code"]
        require(re.fullmatch(r"[0-9]{6}", code) and code not in codes)
        codes.add(code)
        require(len(stock["documents"]) == stock["read_documents"] <= stock["target_documents"] <= 30)
        for doc in stock["documents"]:
            receipt = doc["receipt_no"]
            require(re.fullmatch(r"[0-9]{14}", receipt) and (code, receipt) not in documents)
            require(report_period(doc["report_name"]).isoformat() == doc["period_end"])
            require(doc["period_end"] < doc["receipt_date"])
            expected = (parse_date(doc["receipt_date"])+timedelta(days=1)).isoformat()+"T00:00:00+09:00"
            require(doc["available_from"] == expected)
            stem = code+"_"+receipt
            raw = content(source / "sources" / (stem+".json"), 64 * 1024 * 1024)
            total_bytes += len(raw)
            require(total_bytes <= 128 * 1024 * 1024)
            require(hashlib.sha256(raw).hexdigest() == doc["source_blocks_sha256"])
            blocks = decoded(raw)
            require(isinstance(blocks, list) and len(blocks) <= 100_000)
            by_id = {}
            for block in blocks:
                require(re.fullmatch(r"b[0-9]{6}", block["id"]) and block["id"] not in by_id)
                require(block["kind"] in {"heading", "paragraph", "table"} and isinstance(block["text"], str))
                require(isinstance(block["headings"], list) and all(isinstance(h, str) for h in block["headings"]))
                require(hashlib.sha256(block["text"].encode()).hexdigest() == block["text_sha256"])
                by_id[block["id"]] = block
            documents[code, receipt] = {"metadata": doc, "blocks": by_id, "company_name": stock["company_name"]}
            source_files[stem] = raw
    require(len(documents) == summary["read_documents"])
    require(sum(s["target_documents"] for s in summary["stocks"]) == summary["target_documents"])
    seen = set()
    for claim in claims:
        require(claim["id"] not in seen)
        seen.add(claim["id"])
        doc = documents[claim["code"], claim["receipt_no"]]
        for key in ("receipt_date", "available_from"):
            require(claim[key] == doc["metadata"][key])
        require(claim["report_period_end"] == doc["metadata"]["period_end"])
        block = doc["blocks"][claim["block_id"]]
        require(block["kind"] == "paragraph" and block["headings"] == claim["headings"])
        require(block["text_sha256"] == claim["block_sha256"])
        valid_span(block["text"], claim)
        require(hashlib.sha256(claim["text"].encode()).hexdigest() == claim["text_sha256"])
        require(claim["classification_policy"] == summary["policy"])
        require(all(claim.get(k) is False for k in ("externally_verified", "causal_claim_verified", "model_input_approved")))
        for field in SPAN_FIELDS:
            for item in claim[field]:
                valid_span(claim["text"], item)
                for blocker in item.get("binding_blockers", []):
                    valid_span(claim["text"], blocker)
                for expression in item.get("sign_expressions", []):
                    valid_span(claim["text"], expression)
        for quantity in claim["measurements"]:
            if quantity["normalized_value"] is not None:
                number(quantity["normalized_value"])
    require(tally(claims) == summary["counts"])
    for stock in summary["stocks"]:
        require(tally([c for c in claims if c["code"] == stock["code"]]) == stock["counts"])
    return summary, claims, documents, source_files, raw_summary, raw_claims


def prepare_input(source, root, refresh):
    if not refresh:
        return source
    original, *_ = load_facts(source)
    if original["policy"] == POLICY:
        return source
    business_source = Path(original["source_business"])
    raw = content(business_source / "evidence.json")
    require(hashlib.sha256(raw).hexdigest() == original["source_evidence_sha256"],
            "원래 사업보고서 결과가 바뀌어 자동 재정리하지 않았습니다. 기존 파일을 확인해주세요.")
    print("저장된 같은 사업보고서로 수치를 새 규칙에 따라 정리합니다. 이전 결과는 보존합니다.")
    target, _ = refresh_facts(business_source, root)
    return target


def load_reference():
    raw = content(REFERENCE)
    require(hashlib.sha256(raw).hexdigest() == REFERENCE_SHA256, "개발 기준 파일의 해시가 배포본과 다릅니다.")
    reference = decoded(raw)
    require(reference["schema_version"] == 1 and reference["authorship"] == "assistant_source_review"
            and reference["independent_human_reviewed"] is False and reference["fresh_unseen_evaluation"] is False)
    require(len(reference["cases"]) == 12 and len({c["id"] for c in reference["cases"]}) == 12)
    return reference, raw


def compare_case(case, claims, documents):
    result = {"reference": case, "status": "missing", "quote": None, "claim_id": None,
              "quantity_checks": [], "context_checks": []}
    doc = documents.get((case["code"], case["receipt_no"]))
    if doc is None:
        return result
    if doc["metadata"]["source_blocks_sha256"] != case["source_blocks_sha256"]:
        result["status"] = "source_mismatch"
        return result
    matches = [c for c in claims if (c["code"], c["receipt_no"], c["block_id"], c["text_sha256"]) ==
               (case["code"], case["receipt_no"], case["block_id"], case["quote_sha256"])]
    if len(matches) != 1:
        result["status"] = "ambiguous" if matches else "missing"
        return result
    claim = matches[0]
    result.update(quote=claim["text"], claim_id=claim["id"])
    for expected in case["quantities"]:
        found = [q for q in claim["measurements"] if q["text"] == expected["text"]]
        actual = found[0] if len(found) == 1 else None
        number_match = actual is not None and actual["unit"] == expected["unit"]
        if number_match:
            a, b = actual["normalized_value"], expected["normalized_value"]
            number_match = a is None and b is None if a is None or b is None else number(a) == number(b)
        binding_checked = "metric_equals" in expected or "metric_forbidden" in expected
        binding_match = actual is not None
        if binding_checked and actual is not None:
            binding_match = ("metric_equals" not in expected or actual["metric"] == expected["metric_equals"]) and actual["metric"] not in expected.get("metric_forbidden", [])
        result["quantity_checks"].append({"expected": expected, "actual": actual,
            "number_match": bool(number_match), "binding_checked": binding_checked,
            "binding_match": bool(binding_match) if binding_checked else None})
    for key, value in case["context"].items():
        result["context_checks"].append({"field": key, "expected": value, "actual": claim[key], "matched": claim[key] == value})
    passed = all(q["number_match"] and q["binding_match"] is not False for q in result["quantity_checks"])
    passed = passed and all(c["matched"] for c in result["context_checks"])
    result["status"] = "matched" if passed else "mismatch"
    return result


def render_report(summary, results):
    esc = lambda x: html.escape(str(x))
    articles = []
    for result in results:
        case = result["reference"]
        rows = []
        for item in result["quantity_checks"]:
            expected, actual = item["expected"], item["actual"]
            wanted = (METRIC_NAMES.get(expected["metric_equals"], expected["metric_equals"]) if "metric_equals" in expected else
                      " / ".join(METRIC_NAMES.get(m, m) for m in expected.get("metric_forbidden", []))+"로 연결 금지" if "metric_forbidden" in expected else "항목 연결은 이번 기준 밖")
            actual_metric = METRIC_NAMES.get(actual["metric"], actual["metric"] or "연결 보류") if actual else "수치 없음·복수"
            rows.append(f"<tr><td>{esc(expected['text'])}</td><td>{esc(display_value(expected))}</td>"
                        f"<td>{esc(display_value(actual) if actual else '수치 없음·복수')}</td><td>{'부합' if item['number_match'] else '불일치'}</td>"
                        f"<td>{esc(wanted)}</td><td>{esc(actual_metric)}</td><td>{'기준 밖' if item['binding_match'] is None else '부합' if item['binding_match'] else '불일치'}</td></tr>")
        context_names = {"scope": "설명의 범위", "time_orientation": "설명의 시점", "status": "확인 상태"}
        value_names = {**SCOPES, **TIMES, "needs_review": "확인 필요", "reported_candidate": "보고 표현 후보"}
        contexts = " / ".join(f"{context_names.get(c['field'], c['field'])}: 기준 {value_names.get(c['expected'], c['expected'])}, 결과 {value_names.get(c['actual'], c['actual'])} ({'부합' if c['matched'] else '불일치'})" for c in result["context_checks"])
        link = f"sources/{case['code']}_{case['receipt_no']}.html#{case['block_id']}"
        articles.append(f"<section><h2>{esc(case['title'])} — {STATUSES[result['status']]}</h2>"
            f"<p>{esc(case['code'])} · 접수번호 {esc(case['receipt_no'])}</p><blockquote>{esc(result['quote'] or '기준과 같은 원문·문장·수치를 확인하지 못했습니다. 통과로 채우지 않습니다.')}</blockquote>"
            f"<p>{esc(case['rationale'])}</p><table><tr><th>원문 수치</th><th>개발 기준 값</th><th>프로그램 값</th><th>단위·숫자</th><th>항목 기준</th><th>프로그램 항목</th><th>항목 대조</th></tr>{''.join(rows)}</table>"
            f"<p>{esc(contexts)}</p>"+(f"<p><a href='{esc(link)}'>앞뒤 원문 보기</a></p>" if result["quote"] else "")+"</section>")
    counts = summary["case_counts"]
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>수치와 항목의 검증</title><style>body{{font-family:system-ui;line-height:1.8;max-width:1200px;margin:30px auto;padding:20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}blockquote{{white-space:pre-wrap;background:#f2f6fb;padding:15px;margin:12px 0}}section{{border-top:1px solid #ddd;margin-top:30px}}.note{{background:#fff2d5;padding:16px}}</style>
<h1>수치와 항목의 검증</h1><p class="note">학습 사용 대기. 이 결과만으로 학습 입력을 승인하지 않습니다. 실제 주문·재학습·기존 예측 변경은 실행하지 않았습니다.</p>
<h2>검증 요약</h2><p>출처 대조: 원문 {summary['documents']}건 · 저장 문장 {summary['claims']}개 · 수치 {summary['quantities']}개. 원문 해시·문장 및 수치 위치·공개 시점·저장 집계를 확인했습니다. 이것은 뜻이나 사실 여부의 검증과 다릅니다.</p>
<p>개발 기준 {len(results)}사례: 부합 {counts['matched']} · 불일치 {counts['mismatch']} · 자료 없음 {counts['missing']} · 기준 원문과 다름 {counts['source_mismatch']} · 복수 후보 {counts['ambiguous']}.</p>
<p>12사례는 원문 {summary['reference_documents']}건에서 선택했습니다. 이 기준은 개발 과정에서 AI가 원문을 읽고 작성한 기준입니다. 독립된 사람이 정답을 확인하거나 보지 않은 문서에서 전체 정확도를 평가한 결과가 아닙니다. 기존 오류를 포함해 고른 사례이며 비율을 전체 추출 정확도로 제시하지 않습니다. 기준 자체도 검토가 필요합니다. 자료가 없거나 기준과 다르면 통과로 계산하지 않습니다.</p>
<p>단위·숫자와 항목 연결을 따로 대조합니다. 항목 연결 기준이 없는 수치는 항목 대조를 '기준 밖'으로 남깁니다. 동일 회사·연간 원문에서 나온 사례는 독립 표본으로 취급하지 않습니다. 외부 재무 원문 대조·표 전체·새 문서와 독립 검토는 남아 있습니다.</p>
<p>처리 시각: {esc(summary['created_at'])} · 저장된 추출 정책: {esc(summary['source_policy'])}</p>
{''.join(articles)}<p>12사례를 생략 없이 표시했습니다. 전체 문장과 출처 복사본, 기준·대조 결과는 이번 결과 폴더에 보관합니다. 수치가 잘 읽혀도 투자 판단의 타당성·주가 원인·수익성을 입증하지 않습니다.</p></html>"""


def run(source, root, *, progress=print):
    progress("저장된 문장·수치와 원문 위치 확인 중...")
    original, claims, documents, source_files, raw_summary, raw_claims = load_facts(source)
    reference, raw_reference = load_reference()
    progress("원문에서 작성한 개발 기준 12사례와 대조 중...")
    results = [compare_case(case, claims, documents) for case in reference["cases"]]
    counts = {key: sum(r["status"] == key for r in results) for key in STATUSES}
    summary = {"schema_version": 1, "created_at": datetime.now(SEOUL).isoformat(), "source_facts": str(source.resolve()),
        "source_summary_sha256": hashlib.sha256(raw_summary).hexdigest(), "source_claims_sha256": hashlib.sha256(raw_claims).hexdigest(),
        "reference_sha256": REFERENCE_SHA256, "reference_version": reference["reference_version"], "source_policy": original["policy"],
        "reference_documents": len({(c['code'], c['receipt_no']) for c in reference['cases']}),
        "documents": len(documents), "claims": len(claims), "quantities": sum(len(c["measurements"]) for c in claims),
        "case_counts": counts, "source_status": original["status"], "learning_ready": False,
        "independent_accuracy_verified": False, "fresh_unseen_evaluation": False, "model_inputs_modified": False,
        "training_executed": False, "live_trading_allowed": False,
        "status": "development_checks_matched" if counts["matched"] == len(results) and original["status"] == "structured" else "review_required"}
    parent = root / "fact_checks"; parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    target = parent / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    try:
        (staging / "sources").mkdir()
        for stem, raw in source_files.items():
            (staging / "sources" / (stem+".json")).write_bytes(raw)
            (staging / "sources" / (stem+".html")).write_text(source_html(decoded(raw), stem), encoding="utf-8")
        for name, raw in (("input_summary.json", raw_summary), ("input_claims.jsonl", raw_claims), ("reference.json", raw_reference)):
            (staging / name).write_bytes(raw)
        for name, data in (("summary.json", summary), ("comparisons.json", results)):
            (staging / name).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        (staging / "report.html").write_text(render_report(summary, results), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, summary


def main(argv=None):
    default = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="저장된 수치·항목과 개발 기준 대조")
    parser.add_argument("--root", type=Path, default=default)
    parser.add_argument("--input", type=Path, help="기존 FACTS 결과; 기본은 최신 실행")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="추출 규칙이 바뀌었으면 저장된 같은 원문으로 재정리 후 검증")
    args = parser.parse_args(argv)
    try:
        source = prepare_input(args.input or latest_facts(args.root / "facts"), args.root, args.refresh)
        target, summary = run(source, args.root)
        counts = summary["case_counts"]
        print(f"\n출처 대조: {summary['documents']}건 · 저장 문장 {summary['claims']}개")
        print(f"개발 기준 12사례: 부합 {counts['matched']} · 불일치 {counts['mismatch']} · 미확인 {sum(counts[k] for k in ('missing','source_mismatch','ambiguous'))}")
        print("학습 사용 대기 · 독립된 정확도 검증은 아직 없습니다. 기존 모델·예측을 유지합니다.")
        print(f"검증 보고서: {target / 'report.html'}")
        if sys.platform == "win32" and not args.no_open:
            webbrowser.open((target / "report.html").resolve().as_uri())
        return 0 if summary["status"] == "development_checks_matched" else 2
    except CollectionError as exc:
        print(f"검증 확인 필요: {exc}", file=sys.stderr)
    except FileNotFoundError:
        print("저장된 FACTS 파일을 찾지 못했습니다. 원래 자료 폴더를 확인해주세요. 인증키는 필요하지 않습니다.", file=sys.stderr)
    except KeyboardInterrupt:
        return 130
    except Exception:
        print("검증 자료의 형식 또는 저장 상태를 확인해야 합니다. 기존 자료는 바꾸지 않았습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
