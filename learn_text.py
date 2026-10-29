"""Research-only paired comparison using existing numeric data and report text."""
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
from contextualize import latest_business, read_json, verified_blocks
from financial_history import report_period
from lyupang import CollectionError, SEOUL, parse_date
from observe import digest, read_record
from review import diagnose, latest_learning, load_source
from learning import index_outcomes, partition_examples
from text_learning import attach_text, fragments, predict_candidate, train_comparison


def select_learning(root):
    experiment = root / "forward" / "experiment"
    if experiment.exists():
        manifest = read_record(experiment / "experiment.json")
        source = Path(manifest["source_learning"])
        if digest(read_json(source / "models.json")) != manifest["model_sha256"]:
            raise CollectionError("기존 학습 결과와 관찰 중인 고정 모델이 다릅니다. 기존 파일을 보존하고 확인해주세요.")
        return source
    return latest_learning(root / "learning")


def load_documents(source, codes):
    metadata = read_json(source / "evidence.json")
    if metadata.get("schema_version") != 1 or not 1 <= len(metadata.get("stocks", [])) <= 20:
        raise CollectionError("기존 BUSINESS.cmd의 사업보고서 결과가 필요합니다.")
    histories, documents, audit = {}, {}, []
    seen, total_chars, count = set(), 0, 0
    for stock in metadata["stocks"]:
        code = stock["code"]
        if not re.fullmatch(r"[0-9]{6}", code) or code in seen:
            raise CollectionError("사업보고서 종목 목록이 올바르지 않거나 중복됐습니다.")
        seen.add(code)
        if code not in codes:
            continue
        if len(stock["events"]) > 30:
            raise CollectionError("종목당 원문 수가 지원 범위를 넘었습니다.")
        events, receipts = [], set()
        for original in stock["events"]:
            event = dict(original)
            receipt = event["receipt_no"]
            if not re.fullmatch(r"[0-9]{14}", receipt) or receipt in receipts:
                raise CollectionError("사업보고서 접수번호가 올바르지 않거나 중복됐습니다.")
            receipts.add(receipt)
            key = code + "_" + receipt
            count += 1
            if count > 120:
                raise CollectionError("설명 학습의 전체 원문 120건 지원 한도를 넘었습니다.")
            record = {"document_id": key, "code": code, "receipt_no": receipt, "status": "unavailable"}
            try:
                if (event["status"] != "read" or event.get("company_identity_verified") is not True
                        or event.get("document_kind_verified") is not True
                        or report_period(event["report_name"]).isoformat() != event["period_end"]
                        or event["period_end"] >= event["receipt_date"]):
                    raise CollectionError("기업·문서 종류·회계기간을 확인한 원문이 필요합니다.")
                _, blocks = verified_blocks(source, code, event)
                selected = []
                for block in blocks:
                    headings = re.sub(r"\s+", "", "/".join(block["headings"]))
                    if block["kind"] == "paragraph" and any(word in headings for word in
                            ("사업의내용", "사업개요", "사업의개요", "경영진단", "영업실적", "경영성과")):
                        selected.append(block["text"])
                text = "\n".join(selected)
                if not text or len(text) > 500_000 or total_chars + len(text) > 8_000_000:
                    raise CollectionError("원문 설명이 없거나 텍스트 지원 크기 한도를 넘었습니다.")
                total_chars += len(text)
                documents[key] = {"text": text, "terms": fragments(text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "receipt_date": event["receipt_date"], "blocks": blocks, "title": event["report_name"],
                    "source_blocks_sha256": event["source_blocks_sha256"]}
                record.update(status="read", text_sha256=documents[key]["sha256"], characters=len(text),
                              receipt_date=event["receipt_date"], available_from=event["available_from"],
                              source_blocks_sha256=event["source_blocks_sha256"])
            except Exception as exc:
                event["status"] = "unavailable"
                record["reason"] = str(exc) if isinstance(exc, CollectionError) else "원문 파일의 형식·저장 상태를 확인해야 합니다."
            events.append(event)
            audit.append(record)
        histories[code] = events
    return histories, documents, audit, metadata


def render_report(summary, models):
    esc = lambda value: html.escape(str(value))
    rows, details = [], []
    for horizon, model in models["models"].items():
        if model["status"] != "trained":
            rows.append(f"<tr><td>{esc(horizon)}일봉</td><td colspan='7'>학습 대기: {esc(model['reason'])}</td></tr>")
            continue
        base, text = (model[name]["test_metrics"] for name in ("numeric", "numeric_and_text"))
        gain = model["relative_error_reduction_pct"]
        rows.append(f"<tr><td>{esc(horizon)}일봉</td><td>{text['rows']} / {text['unique_dates']}</td>"
            f"<td>{base['mae_pp']:.3f}</td><td>{base['constant_baseline_mae_pp']:.3f}</td><td>{text['mae_pp']:.3f}</td><td>{'—' if gain is None else f'{gain:.2f}%'}</td>"
            f"<td>{text['direction_accuracy']*100:.1f}% / {text['actual_up_fraction']*100:.1f}%</td>"
            f"<td>{'탈락 — 수치 모델 또는 과거 평균 기준보다 오차가 크거나 같음' if model['disposition']=='rejected' else '보류 — 새 자료 검증 필요'}</td></tr>")
        vocab = model["numeric_and_text"]["text_vocabulary"]
        details.append(f"<p>{esc(horizon)}일봉: 최종 학습에 사용한 원문 {model['training_documents']}건 · 비교 구간의 원문 {model['test_documents']}건 · "
            f"학습에서 읽지 않은 비교 원문 {model['unseen_test_documents']}건 · 이미 읽은 원문을 사용하는 비교 사례 {model['rows_using_seen_documents']}개 · "
            f"사용한 글자 조각 {len(vocab['vocabulary'])}개</p>")
    examples = []
    for example in summary["text_examples"]:
        examples.append(f"<article><p>{example['horizon']}일봉 · 글자 조각 ‘{esc(example['term'])}’ · "
            f"계산식의 계수 {example['coefficient']:+.4f}</p><blockquote>{esc(example['excerpt'])}</blockquote>"
            f"<a href='sources/{esc(example['document_id'])}.html'>학습 원문 보기</a></article>")
    audit_rows = "".join(f"<tr><td>{esc(row['code'])}</td><td>{esc(row['receipt_no'])}</td>"
        f"<td>{'읽음' if row['status']=='read' else '제외'}</td><td>{esc(row.get('reason',''))}</td></tr>" for row in summary["document_audit"])
    status = {"trained": "설명 추가 모델의 연구용 학습·과거 비교 완료", "partial": "일부 구간 연구용 학습 완료", "blocked": "설명 학습 조건 미충족"}[models["status"]]
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>회사 설명을 추가한 연구 모델</title><style>body{{font-family:system-ui;line-height:1.8;max-width:1200px;margin:30px auto;padding:20px;color:#17263c}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:18px;background:#fff3d5}}blockquote{{white-space:pre-wrap;background:#f3f6fa;padding:12px}}</style>
<h1>회사 설명을 추가한 연구 모델</h1><p>{esc(status)}</p>
<p>{esc(models.get('reason', ''))}</p>
<p class="note">이미 확인한 과거 자료를 다시 비교한 개발용 연구입니다. 기존 관찰 모델과 예측은 바꾸지 않았습니다. 실제 주문·모델 자동 채택·본문 의미 이해나 수익성 검증은 수행하지 않았습니다.</p>
<p>설명 연결: {summary['paired_rows']}/{summary['source_rows']}사례 · 사용 가능 원문 {summary['read_documents']}건 · 종목 {summary['symbols']}개</p>
<p>사업보고서 목록이 없는 종목: {esc(', '.join(summary['missing_business_codes']) or '없음')}</p>
<h2>같은 사례에서의 비교</h2><p>평균 오차(%p)는 작을수록 좋습니다. 설명이 있는 동일한 종목·날짜·미래 결과만 양쪽 모델에 사용합니다. 수치 모델도 이 같은 사례에서 다시 학습하므로 예전 보고서의 점수와 다를 수 있습니다. 개선율은 설명 모델과 수치 모델의 오차 차이이며 거래 수익률이 아닙니다.</p>
<table><tr><th>관찰 구간</th><th>사례 / 날짜</th><th>수치 모델 오차</th><th>과거 평균 기준 오차</th><th>설명 추가 오차</th><th>수치 모델 대비 개선율</th><th>설명 모델 상승 일치 / 실제 상승 비율</th><th>자동 판정</th></tr>{''.join(rows)}</table>
<p>과거 평균 기준은 학습·검증 구간의 평균 가격 변화만 답하는 방식입니다. 설명 모델이 수치 모델과 이 평균 기준 모두보다 오차가 작아도 새 자료 검증 전에는 보류합니다.</p>
<h2>원문이 반복되는 범위</h2>{''.join(details)}
<p>하나의 연간 보고서가 여러 거래일에 반복 사용됩니다. 사례 수가 많아도 서로 독립된 원문 수가 많다는 뜻은 아닙니다. 회사 이름이나 반복 문체를 외웠을 수 있으며, 새로운 기업·새 원문·시장 전체에서의 성능을 입증하지 않습니다.</p>
<h2>학습에 사용한 글자 조각과 원문 예시</h2><p>계수 크기가 큰 조각 중 원문 위치를 확인한 예시입니다. 계수는 모델 계산식의 값이며 주가 상승·하락의 원인이나 종목 추천을 뜻하지 않습니다. 짧은 조각은 독립된 단어조차 아닐 수 있습니다.</p>{''.join(examples)}
<h2>이번 학습의 방식</h2><p>과거 가격 변화·거래량·공시 건수·검증한 재무 비율에, 사업·경영진단 문단의 한글·영문 2~3글자 조각 빈도를 추가합니다. 숫자·표는 텍스트 조각 입력에서 제외합니다. 뜻을 정확히 이해하거나 설명의 진실성을 확인하는 모델은 아닙니다. 사람이 검토한 2개 답안을 정답으로 확대하거나 문장 분류 규칙의 출력을 사람 정답으로 취급하지 않습니다.</p>
<p>접수일 다음 날부터 이용 가능한 최신 회계기간·접수번호만 연결합니다. 최신 원문을 못 읽거나 같은 날 여러 제출본이 있으면 이전 값으로 채우지 않습니다. 원문 해시와 정확한 기업·문서 확인 기록을 검사했습니다.</p>
<p>단어 조각 선택·역문서빈도·결측값 보정·표준화·모델 강도 선택은 학습·검증 구간에서만 수행합니다. 정정·새 보고서는 공개 이후에만 반영합니다. 기간 경계 뒤에 끝나는 미래 결과는 앞 기간에서 제외합니다. 비교 기간을 모델 선택에 사용하지 않았지만 이미 열람한 과거 기간이므로 새 미사용 검증은 아닙니다.</p>
<p>이번 실험은 5·20일봉의 비교 관찰 구간에 한정하며 보유 기간의 자동 선택이 아닙니다. 60일봉은 이번 비교 대상에 넣지 않습니다. 비용·세금·체결·시장 대비 효과·뉴스·상장폐지 종목·거래소 달력·과거 당시 수정주가·통계적 유의성은 검증하지 않았습니다.</p>
<h2>원문 확인 상태</h2><table><tr><th>종목</th><th>접수번호</th><th>처리</th><th>확인할 내용</th></tr>{audit_rows}</table>
<p>제외 사유별 사례 수: {esc(summary['text_exclusions'])}</p>
<p>결과는 별도 text_learning 폴더에 저장합니다. 이 후보 모델은 OBSERVE에 자동 연결되지 않습니다. 현재 관찰 중인 6개 등 기존 예측 기록과 고정 모델은 그대로 이어갑니다.</p></html>"""


def run(learning_source, business_source, root, *, progress=print, minimums=(100, 30, 30), minimum_documents=6):
    progress("기존 학습 자료의 해시·기간 분리·평가 결과를 확인하는 중...")
    original, features, outcomes = load_source(learning_source)
    diagnosis, _ = diagnose(original, features, outcomes)
    horizons = [h for h in ("5", "20") if diagnosis["models"].get(h, {}).get("disposition") == "hold"]
    codes = {feature["code"] for feature in features}
    histories, documents, audit, business_meta = load_documents(business_source, codes)
    if parse_date(business_meta["reference_date"]) < max(parse_date(f["as_of_date"]) for f in features):
        raise CollectionError("사업보고서 기준일이 학습 사례의 마지막 관찰일보다 이전입니다.")
    paired, exclusions = attach_text(features, histories, documents)
    progress(f"회사 설명 연결 {len(paired)}/{len(features)}사례 · 읽은 원문 {len(documents)}건")
    progress("같은 기간·사례에서 수치 모델과 설명 추가 모델을 학습·비교하는 중...")
    models = train_comparison(paired, outcomes, documents, original["split"], horizons,
                              minimums=minimums, minimum_documents=minimum_documents)
    summary = {"schema_version": 1, "created_at": datetime.now(SEOUL).isoformat(), "status": models["status"],
        "source_learning": str(learning_source.resolve()), "source_business": str(business_source.resolve()),
        "source_learning_model_sha256": digest(original),
        "source_feature_sha256": hashlib.sha256((learning_source / "features_with_financials.csv").read_bytes()).hexdigest(),
        "source_outcome_sha256": hashlib.sha256((learning_source / "future_outcomes.csv").read_bytes()).hexdigest(),
        "source_business_sha256": hashlib.sha256((business_source / "evidence.json").read_bytes()).hexdigest(),
        "source_rows": len(features), "paired_rows": len(paired), "symbols": len(codes), "read_documents": len(documents),
        "document_audit": audit, "text_exclusions": exclusions, "fresh_unseen_evidence": False,
        "automatic_promotion": False, "live_trading_allowed": False, "frozen_observer_modified": False,
        "horizons": horizons, "split": original["split"], "semantic_accuracy_established": False,
        "point_in_time_verified": False, "statistical_significance_established": False, "costs_included": False,
        "source_complete": all(histories.get(code) for code in codes) and all(row["status"] == "read" for row in audit),
        "missing_business_codes": sorted(codes-set(histories)), "text_examples": []}
    historical = []
    indexed = index_outcomes(outcomes)
    for horizon, comparison in models["models"].items():
        if comparison["status"] != "trained":
            continue
        parts, _ = partition_examples(paired, indexed, int(horizon), original["split"]["validation_start"], original["split"]["test_start"])
        test = parts["test"]
        base_values = predict_candidate(comparison["numeric"], test, documents)
        text_values = predict_candidate(comparison["numeric_and_text"], test, documents)
        for (feature, outcome), base_value, text_value in zip(test, base_values, text_values):
            historical.append({"horizon": horizon, "code": feature["code"], "as_of_date": feature["as_of_date"],
                "entry_date": outcome["entry_date"], "exit_date": outcome["exit_date"], "text_document_id": feature["text_document_id"],
                "actual_change_pct": outcome["price_return_pct"], "numeric_prediction_pct": float(base_value), "text_prediction_pct": float(text_value)})
        text_model = comparison["numeric_and_text"]
        vocabulary = text_model["text_vocabulary"]
        ranked = sorted(zip(vocabulary["vocabulary"], text_model["coefficients"][len(text_model["numeric_features"]):]), key=lambda pair: -abs(pair[1]))
        shown = 0
        for term, coefficient in ranked:
            for key in vocabulary["training_document_ids"]:
                match = re.search(re.escape(term), documents[key]["text"], re.IGNORECASE)
                if match:
                    start, end = max(0, match.start()-80), min(len(documents[key]["text"]), match.end()+80)
                    summary["text_examples"].append({"horizon": int(horizon), "term": term, "coefficient": coefficient,
                        "document_id": key, "text_sha256": documents[key]["sha256"], "start": start, "end": end,
                        "excerpt": documents[key]["text"][start:end]})
                    shown += 1
                    break
            if shown >= 6:
                break
    parent = root / "text_learning"
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    target = parent / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    try:
        (staging / "sources").mkdir()
        index = {}
        for key, doc in documents.items():
            index[key] = {k: doc[k] for k in ("sha256", "receipt_date", "source_blocks_sha256", "title")}
            (staging / "sources" / (key + ".html")).write_text(source_html(doc["blocks"], doc["title"]), encoding="utf-8")
            (staging / "sources" / (key + ".txt")).write_bytes(doc["text"].encode("utf-8"))
        columns = ("code", "as_of_date", "text_document_id", "text_receipt_date", "text_available_from", "text_sha256")
        with (staging / "text_joins.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader(); writer.writerows(paired)
        with (staging / "historical_comparisons.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("horizon", "code", "as_of_date", "entry_date", "exit_date", "text_document_id",
                                                       "actual_change_pct", "numeric_prediction_pct", "text_prediction_pct"))
            writer.writeheader(); writer.writerows(historical)
        for name, value in (("models.json", models), ("document_index.json", index)):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        summary["models_sha256"] = hashlib.sha256((staging / "models.json").read_bytes()).hexdigest()
        summary["joins_sha256"] = hashlib.sha256((staging / "text_joins.csv").read_bytes()).hexdigest()
        summary["comparison_sha256"] = hashlib.sha256((staging / "historical_comparisons.csv").read_bytes()).hexdigest()
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        (staging / "report.html").write_text(render_report(summary, models), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, summary, models


def main(argv=None):
    default = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="사업보고서 설명 추가 모델의 연구용 학습·과거 비교")
    parser.add_argument("--root", type=Path, default=default)
    parser.add_argument("--learning", type=Path, help="기존 수치 학습 결과; 기본은 관찰 중인 모델의 원본")
    parser.add_argument("--business", type=Path, help="기존 사업보고서 결과; 기본은 최신 BUSINESS 결과")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        target, summary, models = run(args.learning or select_learning(args.root),
                                     args.business or latest_business(args.root / "business"), args.root)
        print(f"\n회사 설명 연결: {summary['paired_rows']}/{summary['source_rows']}사례")
        print("설명 추가 모델의 연구용 학습·과거 비교 완료" if models["status"] == "trained" else "일부 구간 학습 또는 조건 미충족 · 보고서를 확인해주세요.")
        print(f"결과 보고서: {target / 'report.html'}")
        print("기존 고정 모델·예측 기록은 유지합니다. 실제 주문과 자동 채택은 실행하지 않았습니다.")
        if not summary["source_complete"]:
            print("읽지 못해 제외한 원문이 있습니다. 보고서의 원문 확인 상태를 봐주세요.")
        if sys.platform == "win32" and not args.no_open:
            webbrowser.open((target / "report.html").resolve().as_uri())
        return 0 if models["status"] == "trained" and summary["source_complete"] else 2
    except CollectionError as exc:
        print(f"설명 학습 확인 필요: {exc}", file=sys.stderr)
    except FileNotFoundError:
        print("기존 학습 또는 사업보고서 파일을 찾지 못했습니다. 원래 저장 폴더를 확인해주세요. 인증키는 필요하지 않습니다.", file=sys.stderr)
    except KeyboardInterrupt:
        return 130
    except Exception:
        print("설명 모델의 연구용 비교를 완료하지 못했습니다. 마지막 메시지를 알려주세요. 기존 자료는 바꾸지 않았습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
