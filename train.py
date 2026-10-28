"""User-operated receipt audit and research model training; never places orders."""
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
from datetime import datetime
from pathlib import Path

from dart_client import DartClient, stored_key
from financial_history import FINANCIAL_COLUMNS, audit_reports, financial_as_of
from learning import train_models
from lyupang import CollectionError, SEOUL, parse_date
from research import safe_reason
from research_data import FEATURE_COLUMNS, OUTCOME_COLUMNS, build_cases, load_snapshot, write_table


def latest_research(root: Path) -> Path:
    candidates = []
    if root.exists():
        for folder in root.iterdir():
            if not folder.is_dir() or folder.is_symlink() or folder.name.startswith("."):
                continue
            try:
                batch = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
                captured = datetime.fromisoformat(batch["created_at"])
                if captured.tzinfo is not None:
                    candidates.append((captured, folder))
            except (OSError, ValueError, KeyError, TypeError):
                continue
    if not candidates:
        raise CollectionError("먼저 RESEARCH.cmd를 실행해주세요.")
    return max(candidates, key=lambda item: item[0])[1]


def render_report(summary: dict, models: dict) -> str:
    esc = lambda value: html.escape(str(value))
    stocks = "".join(f"<tr><td>{esc(r['code'])}</td><td>{r.get('verified_reports', 0)}/{r.get('reports', 0)}</td>"
                     f"<td>{r.get('usable_rows', 0)}/{r.get('rows', 0)}</td><td>{esc(r.get('error') or r.get('warning', ''))}</td></tr>"
                     for r in summary["stocks"])
    rows = []
    for horizon, model in models["models"].items():
        if model["status"] != "trained":
            rows.append(f"<tr><td>{esc(horizon)} 거래일</td><td colspan='4'>표본 부족으로 학습 대기 · {esc(model.get('counts'))}</td></tr>")
            continue
        test = model["test_metrics"]
        rows.append(f"<tr><td>{esc(horizon)} 거래일</td><td>{test['rows']}</td>"
                    f"<td>{test['mae_percentage_points']:.3f}</td><td>{test['baseline_mae_percentage_points']:.3f}</td>"
                    f"<td>{test['direction_accuracy'] * 100:.1f}%</td></tr>")
    status = {"trained": "연구용 모델 학습·평가 완료", "partial": "일부 관찰 구간 학습 완료", "blocked": "학습 조건 미충족"}[models["status"]]
    observed = summary.get("financial_field_observed_rows", {})
    coverage = " · ".join(f"{label}: {observed.get(key, 0)}사례" for key, label in
                           (("debt_ratio_pct", "부채/자산 비율"), ("net_margin_pct", "순이익률"), ("revenue_yoy_pct", "매출 증가율")))
    return f"""<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Lyupang 원문 검증과 첫 학습</title><style>body{{max-width:1050px;margin:40px auto;padding:0 20px;font-family:system-ui,sans-serif;color:#182335;line-height:1.8}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}section{{margin:30px 0}}.note{{padding:18px;background:#fff4d6;border-radius:10px}}</style>
<h1>원문 검증과 첫 학습</h1><p>{esc(status)}</p><p class='note'>연구용 결과입니다. 실제 주문과 자동매매는 연결하지 않았습니다. 이 결과는 수익 보장이나 종목 추천이 아닙니다.</p>
<section><h2>접수번호별 재무 원문</h2><table><tr><th>종목</th><th>읽기 성공 원문/대상</th><th>재무 연결 사례/전체</th><th>확인할 내용</th></tr>{stocks}</table>
<p>접수일 다음 날 이후의 사례에만 사용합니다. 정정 보고서는 그 정정이 공개된 이후에만 반영합니다. 같은 날 여러 보고서가 있거나 최신 원문을 읽지 못하면 이전 값으로 대신 채우지 않습니다.</p></section>
<section><h2>최종 평가 기간 결과</h2><p>학습·모델 선택에 쓰지 않은 마지막 기간에서 평가했습니다. 평균 오차는 미래 가격 변화 예측과 실제 변화의 차이(%포인트)이며, 작을수록 좋습니다. 비교 기준은 과거 학습 구간의 평균 변화만 예측하는 방식입니다.</p>
<table><tr><th>비교 관찰 구간</th><th>평가 사례 수</th><th>모델 평균 오차</th><th>비교 기준 오차</th><th>상승 여부 일치</th></tr>{''.join(rows)}</table>
<p>{esc(models.get('reason', ''))}</p><p>5·20·60거래일은 모델 비교용 관찰 구간입니다. 실제 보유 기간이나 매매 규칙을 확정한 결과가 아닙니다. 수수료·세금·슬리피지·체결 실패는 평가하지 않았습니다.</p></section>
<section><h2>이번 모델의 입력</h2><p>과거 가격 변화, 거래량 비율, 이전 공시 건수, 원문에서 확인한 부채/자산 비율·순이익률·매출 증가율을 사용합니다. 재무 항목이 없는 일부 열은 학습 구간의 중앙값으로 보정하며, 평가 자료로 보정값을 계산하지 않습니다.</p>
<p>원문에서 읽어 사례에 연결한 항목: {esc(coverage)}. 학습 자료 전체에서 값이 없는 열은 모델에 기여하지 않습니다. 원문에 없는 재무 수치를 실제로 확인한 값으로 취급하지 않습니다.</p>
<p>입력 수치 보정과 모델 선택에 최종 평가 자료를 사용하지 않았습니다. 기간 경계 뒤에 끝나는 미래 결과는 앞 기간 학습에서 제외합니다. 선택한 종목의 나중 기간을 평가한 것이며, 새로운 종목이나 시장 전체에서의 성능을 검증한 결과가 아닙니다. 통계적 관계를 학습하며 상승 원인을 입증하지는 않습니다.</p></section>
<section><h2>남아 있는 확인</h2><p>사업·뉴스 본문 의미 분석, 상장폐지·시장 전체 표본, 거래일 달력 대조, 과거 수정주가 시점, 종목·구간이 겹치는 표본의 통계적 불확실성은 해결되지 않았습니다. 재무 원문의 접수번호와 시점을 확인한 범위 외에 전체 데이터가 과거 당시와 동일하다고 인증하지 않습니다.</p>
<p>수집은 원본 파일 그대로 보관하지만 전체 XBRL 분류체계 검증기는 아닙니다. 표준 IFRS 항목·원화·명확한 기업/기간/범위만 읽고, 지원하지 않는 형식이나 항목은 추정하지 않습니다.</p>
<p>실제 매매에 사용할 수 있는 모델로 승인하지 않았습니다. 모델 학습이 됐어도 종목 추천이나 주문을 실행하지 않습니다.</p></section></html>"""


def run(input_folder: Path, output_root: Path, document_cache: Path, client, *, progress=print, minimums=(100, 30, 30)):
    batch = json.loads((input_folder / "batch.json").read_text(encoding="utf-8"))
    if batch.get("schema_version") != 1 or batch.get("dart_state") != "enabled" or not batch.get("stocks"):
        raise CollectionError("DART를 연결한 RESEARCH.cmd 결과가 필요합니다.")
    if len(batch["stocks"]) > 20:
        raise CollectionError("종목 수 한도를 초과했습니다.")
    summary = {"schema_version": 1, "created_at": datetime.now(SEOUL).isoformat(), "stocks": [],
               "source_research": str(input_folder.resolve()), "point_in_time_verified": False,
               "production_training_ready": False, "trading_enabled": False, "research_only": True}
    features, outcomes, audits = [], [], {}
    for stock in batch["stocks"]:
        code = stock["code"]
        validate_code = code.isascii() and len(code) == 6 and code.isdigit()
        if not validate_code:
            raise CollectionError("연구 결과의 종목코드가 올바르지 않습니다.")
        item = {"code": code, "rows": stock.get("feature_rows", 0), "usable_rows": 0, "reports": 0, "verified_reports": 0}
        try:
            if stock.get("price_status") != "collected":
                raise CollectionError("이 종목의 가격 수집을 먼저 완료해주세요.")
            metadata, bars = load_snapshot(Path(stock["source_snapshot"]))
            if metadata["code"] != code:
                raise CollectionError("가격 파일의 종목코드가 요청과 다릅니다.")
            payload = json.loads((input_folder / "dart" / f"{code}.json").read_text(encoding="utf-8"))
            filings = payload.get("disclosures")
            if not isinstance(filings, list):
                raise CollectionError("이 종목의 공시 목록이 없습니다.")
            progress(f"종목 {code} · 접수번호별 재무 원문 확인 중...")
            events = audit_reports(code, payload["company"]["corp_code"], parse_date(metadata["requested_start"]),
                                   parse_date(metadata["requested_end"]), client, document_cache, progress=progress)
            audits[code] = events
            base_features, labels = build_cases(bars, filings)
            enriched = [{**row, **financial_as_of(events, parse_date(row["as_of_date"]))} for row in base_features]
            features.extend(enriched)
            outcomes.extend(labels)
            item.update(rows=len(enriched), usable_rows=sum(f["financial_available"] for f in enriched),
                        reports=len(events), verified_reports=sum(e["status"] == "verified" for e in events))
            unavailable = [e.get("reason", "원문 미확인") for e in events if e["status"] != "verified"]
            if unavailable:
                item["warning"] = "일부 원문 제외: " + " / ".join(dict.fromkeys(unavailable))[:500]
            progress(f"  원문 읽기 성공 {item['verified_reports']}/{item['reports']}건 · 재무 연결 {item['usable_rows']}/{item['rows']}사례")
        except Exception as exc:
            item["error"] = safe_reason(exc)
            progress(f"  {code} 확인 필요: {item['error']}")
        summary["stocks"].append(item)
    progress("공개 시점 연결 결과로 학습·검증·최종 평가 구간을 분리하는 중...")
    models = train_models(features, outcomes, minimums=minimums)
    if models["status"] == "blocked" and not features and any(s.get("error") for s in summary["stocks"]):
        models["reason"] = "재무 원문 연결 단계가 중단되어 학습 입력을 만들지 못했습니다. 종목별 확인할 내용을 봐주세요."
    summary["model_status"] = models["status"]
    summary["rows"] = len(features)
    summary["source_rows"] = sum(s["rows"] for s in summary["stocks"])
    summary["financial_field_observed_rows"] = {key: sum(f.get(key) is not None for f in features)
                                               for key in ("debt_ratio_pct", "net_margin_pct", "revenue_yoy_pct")}
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=output_root))
    target = output_root / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    try:
        write_table(staging / "features_with_financials.csv", FEATURE_COLUMNS + FINANCIAL_COLUMNS, features)
        write_table(staging / "future_outcomes.csv", OUTCOME_COLUMNS, outcomes)
        for name, value in (("summary.json", summary), ("models.json", models), ("financial_audit.json", audits)):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        summary["feature_sha256"] = hashlib.sha256((staging / "features_with_financials.csv").read_bytes()).hexdigest()
        summary["outcome_sha256"] = hashlib.sha256((staging / "future_outcomes.csv").read_bytes()).hexdigest()
        (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (staging / "report.html").write_text(render_report(summary, models), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return target, summary, models


def main(argv=None):
    root = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="공개 시점 연결과 첫 연구용 모델 학습")
    parser.add_argument("--input", type=Path, help="기존 research 실행 결과 폴더; 생략하면 최신 결과")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        key = stored_key()
        if not key:
            raise CollectionError("기존 DART 키를 찾지 못했습니다. DART_SETUP.cmd로 확인해주세요.")
        source = args.input or latest_research(args.root / "research")
        target, summary, models = run(source, args.root / "learning", args.root / "financial_documents", DartClient(key))
        report_to_open = target / "report.html"
        print(f"\n재무 원문 연결: {sum(s['usable_rows'] for s in summary['stocks'])}/{summary['source_rows']}사례")
        labels = {"trained": "연구용 모델 학습·평가 완료", "partial": "일부 모델 학습 완료", "blocked": "조건 미충족으로 학습 대기"}
        print(labels[models["status"]])
        if any(s.get("error") or s.get("warning") for s in summary["stocks"]):
            print("읽지 못한 원문 또는 제외된 종목이 있습니다. 보고서의 확인할 내용을 봐주세요.")
        print(f"결과 보고서: {target / 'report.html'}")
        if any(m.get("status") == "trained" for m in models["models"].values()):
            from review import run as review_run
            try:
                review_path, diagnosis = review_run(target, args.root / "diagnostics", args.root / "evaluation_registry")
            except Exception:
                print("학습 결과는 저장됐지만 성능 진단을 완료하지 못했습니다. REVIEW.cmd로 확인해주세요.", file=sys.stderr)
                return 2
            for horizon, checked in diagnosis["models"].items():
                state = "탈락" if checked["disposition"] == "rejected" else "보류"
                print(f"성능 진단 {horizon}거래일: {state}")
            print("자동 채택 모델: 0개 · 새 기간 검증 전에는 채택하지 않습니다.")
            print(f"진단 보고서: {review_path / 'report.html'}")
            report_to_open = review_path / "report.html"
        print("실제 주문 기능은 연결하지 않았습니다.")
        if sys.platform == "win32" and not args.no_open:
            webbrowser.open(report_to_open.resolve().as_uri())
        return 0 if models["status"] == "trained" and not any(s.get("error") for s in summary["stocks"]) else 2
    except CollectionError as exc:
        print(f"학습 준비 실패: {exc}", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print("작업을 취소했습니다.", file=sys.stderr)
        return 130
    except Exception:
        print("학습 준비를 완료하지 못했습니다. 실행 창의 마지막 메시지를 알려주세요.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
