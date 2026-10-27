"""Multi-stock research preparation, not a trained model or a trading engine."""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path

from dart_client import DartClient, DartError, stored_key
from lyupang import CollectionError, KiwoomSource, SEOUL, collect, default_period, parse_date, save_collection, validate_request
from research_data import FEATURE_COLUMNS, HORIZONS, OUTCOME_COLUMNS, build_cases, find_cached, write_table

DEFAULT_CODES = ("005930", "005380", "035420")  # Connection examples, not recommendations or a representative universe.


def safe_reason(exc: Exception) -> str:
    return str(exc) if isinstance(exc, CollectionError) else "조회 또는 저장을 완료하지 못했습니다. 연결 진단이 필요합니다."


def render_report(batch: dict, outcomes: list[dict]) -> str:
    esc = lambda value: html.escape(str(value))
    statuses = {"collected": "수집 완료", "failed": "실패", "waiting_key": "DART 키 등록 대기",
                "enabled": "연결 준비", "error": "오류", "not_requested": "가격만 요청",
                "company_not_found": "기업번호 확인 필요", "disclosures_collected": "공시 목록 수집",
                "annual_data_unavailable": "연간 재무자료 없음"}
    stocks = "".join(f"<tr><td>{esc(r['code'])}</td><td>{esc(r.get('row_count', 0))}</td>"
                     f"<td>{'기존 자료 재사용' if r.get('reused') else '새 조회'}</td>"
                     f"<td>{esc(statuses.get(r['price_status'], r['price_status']))}</td><td>{esc(statuses.get(r['dart_status'], r['dart_status']))}</td>"
                     f"<td>{esc(r.get('error', ''))}</td></tr>" for r in batch["stocks"])
    counts = []
    for horizon in HORIZONS:
        selected = [row for row in outcomes if row["observed_sessions"] == horizon]
        tally = Counter(row["direction"] for row in selected if row["status"] == "observed")
        counts.append(f"<tr><td>{horizon}거래일</td><td>{tally['up']}</td><td>{tally['down']}</td>"
                      f"<td>{tally['flat']}</td><td>{sum(r['status'] == 'pending' for r in selected)}</td>"
                      f"<td>{sum(r['status'] == 'quality_issue' for r in selected)}</td></tr>")
    state = "요청한 자료 준비 완료" if batch["status"] == "collected" else "일부 자료 준비 · 대기 항목 있음"
    dart = {"waiting_key": "DART 인증키 미등록 — DART_SETUP.cmd에서 등록하면 재무·공시를 연결할 수 있습니다.",
            "error": "DART 조회 오류 — 아래 종목별 상태를 확인해주세요.",
            "enabled": "DART 연결 사용", "not_requested": "가격 자료만 준비하도록 선택했습니다."}[batch["dart_state"]]
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Lyupang 사례 준비 결과</title><style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;color:#182335;line-height:1.7;background:#f8fafc}}h1{{font-size:28px}}section{{background:white;border:1px solid #dbe2ea;border-radius:12px;padding:20px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #e2e8f0;padding:10px;font-size:14px}}.note{{background:#fff8e5;padding:16px;border-radius:10px}}code{{background:#eef2f7;padding:2px 5px}}</style>
<h1>Lyupang · 학습 사례 준비</h1><p>{esc(batch['requested_start'])} ~ {esc(batch['requested_end'])} · {esc(state)}</p>
<p class="note">현재 단계는 데이터와 비교 사례 준비입니다. 모델 학습·종목 추천·자동 주문은 수행하지 않았습니다.</p>
<section><h2>자료 수집 상태</h2><p>{esc(dart)}</p><p>{esc(batch.get('dart_error', ''))}</p><table><tr><th>종목코드</th><th>거래일 수</th><th>자료 사용</th><th>가격</th><th>재무·공시</th><th>확인할 내용</th></tr>{stocks}</table></section>
<section><h2>이후 가격 변화의 비교 사례</h2><p>관찰 다음 거래일 시가부터 각 구간 마지막 종가까지의 가격 변화입니다. 수수료·세금·슬리피지는 포함하지 않았습니다. 5·20·60거래일은 비교용 관찰 구간이며, 정해진 매매 보유 기간이 아닙니다.</p>
<table><tr><th>관찰 구간</th><th>상승</th><th>하락</th><th>보합</th><th>이후 자료 부족</th><th>품질 확인 필요</th></tr>{''.join(counts)}</table></section>
<section><h2>다음 학습에 사용할 파일</h2><p><code>features.csv</code>: 각 관찰일까지의 가격·거래량 계산과, 연결됐을 때의 이전 공시 건수입니다.</p>
<p><code>future_outcomes.csv</code>: 이후 가격 변화입니다. 미래 결과이므로 예측 입력으로 사용하면 안 됩니다.</p>
<p><code>dart/</code>: 공시 접수일·원문 링크와 최근 연간 재무제표 조회 기록입니다. 같은 날 공시는 계산에서 제외했고, 재무 수치는 과거 당시의 수치인지 확인하기 전까지 예측 입력에 넣지 않습니다.</p></section>
<section><h2>아직 확인하지 않은 것</h2><p>상승 이유를 자동으로 설명하거나 인과관계를 입증한 결과가 아닙니다. 수정주가의 권리 반영 시점, 거래일 달력 누락, 상장폐지 종목, 현재 예시 종목의 선택 편향이 남아 있습니다. 겹치는 관찰 구간은 독립 표본이 아니며 학습·검증 분리 때 추가 처리가 필요합니다.</p>
<p>사업 분석·뉴스 본문 분석·모델 학습·모의매매·실전 주문은 이후 단계입니다.</p></section></html>"""


def run_batch(codes, start, end, data_root: Path, report_root: Path, *, source_factory=KiwoomSource,
              dart=None, use_dart=True, dart_setup_error=None, sleep=time.sleep, progress=print) -> tuple[Path, dict]:
    codes = list(dict.fromkeys(codes))
    if not 1 <= len(codes) <= 20:
        raise CollectionError("한 번에 1~20개 종목을 지정해주세요.")
    for code in codes:
        validate_request(code, start, end)
    now = datetime.now(SEOUL)
    batch = {"schema_version": 1, "created_at": now.isoformat(), "requested_start": start.isoformat(),
             "requested_end": end.isoformat(), "status": "partial", "stocks": [],
             "dart_state": "not_requested" if not use_dart else "waiting_key" if dart is None else "enabled",
             "training_ready": False, "trading_enabled": False, "horizons_are_observation_windows": True,
             "point_in_time_verified": False, "universe_is_representative": False}
    companies = {}
    if use_dart and dart_setup_error:
        batch["dart_state"], batch["dart_error"] = "error", dart_setup_error
    if use_dart and dart is not None:
        try:
            companies = dart.companies(codes)
        except Exception as exc:
            batch["dart_state"] = "error"
            batch["dart_error"] = safe_reason(exc)
    source, features, outcomes, dart_payloads = None, [], [], {}
    for code in codes:
        progress(f"종목 {code} 준비 중...")
        item = {"code": code, "price_status": "failed", "dart_status": batch["dart_state"], "reused": False}
        did_request = False
        try:
            cached = find_cached(data_root, code, start, end)
            if cached:
                snapshot, bars = cached
                item["reused"] = True
                progress(f"  검증된 기존 자료 재사용: {len(bars)}거래일")
            else:
                if source is None:
                    source = source_factory()
                did_request = True
                result = collect(source, code, start, end,
                                 progress=lambda page, rows: progress(f"  {page}페이지 · {rows}거래일"))
                snapshot, bars = save_collection(result, data_root), result.bars
            item.update(price_status="collected", row_count=len(bars), source_snapshot=str(snapshot.resolve()))
            filings, financials = None, None
            if batch["dart_state"] == "enabled":
                company = companies.get(code)
                if not company:
                    item["dart_status"] = "company_not_found"
                    item["error"] = "DART 기업번호를 찾지 못했습니다."
                else:
                    try:
                        filings = dart.disclosures(company["corp_code"], start, end)
                        item["dart_status"] = "disclosures_collected"
                        financials = dart.annual_accounts(company["corp_code"], end.year - 1)
                        item["dart_status"] = "collected" if financials["status"] == "collected" else "annual_data_unavailable"
                    except Exception as exc:
                        item["dart_status"] = "error"
                        item["error"] = safe_reason(exc)
                    dart_payloads[code] = {"company": company, "retrieved_at": now.isoformat(), "disclosures": filings,
                                           "annual_financials": financials, "historical_financial_features_enabled": False}
            stock_features, stock_outcomes = build_cases(bars, filings)
            features.extend(stock_features)
            outcomes.extend(stock_outcomes)
            item["feature_rows"] = len(stock_features)
            if not stock_features:
                item["error"] = "비교 사례 생성에 필요한 61개 이상의 일봉이 없습니다."
        except Exception as exc:
            item["error"] = safe_reason(exc)
        finally:
            batch["stocks"].append(item)
            if did_request:
                sleep(0.6)
    if all(item["price_status"] == "collected" and item.get("feature_rows", 0) > 0 and
           (not use_dart or item["dart_status"] == "collected") for item in batch["stocks"]):
        batch["status"] = "collected"
    batch["feature_rows"], batch["outcome_rows"] = len(features), len(outcomes)
    report_root.mkdir(parents=True, exist_ok=True)
    target = report_root / f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=report_root))
    try:
        write_table(staging / "features.csv", FEATURE_COLUMNS, features)
        write_table(staging / "future_outcomes.csv", OUTCOME_COLUMNS, outcomes)
        (staging / "batch.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
        if dart_payloads:
            (staging / "dart").mkdir()
            for code, payload in dart_payloads.items():
                (staging / "dart" / f"{code}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (staging / "report.html").write_text(render_report(batch, outcomes), encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return target, batch


def main(argv=None) -> int:
    start, end = default_period()
    root = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="여러 종목의 가격 사례와 DART 자료 준비")
    parser.add_argument("--codes", default=",".join(DEFAULT_CODES))
    parser.add_argument("--start", default=start.isoformat())
    parser.add_argument("--end", default=end.isoformat())
    parser.add_argument("--data-root", type=Path, default=root / "data" if sys.platform == "win32" else root)
    parser.add_argument("--report-root", type=Path, default=root / "research")
    parser.add_argument("--profile", default="kiwoom-real")
    parser.add_argument("--prices-only", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        codes = [part.strip() for part in args.codes.split(",")]
        if not 1 <= len(set(codes)) <= 20:
            raise CollectionError("한 번에 1~20개 종목을 지정해주세요.")
        start, end = parse_date(args.start), parse_date(args.end)
        for code in codes:
            validate_request(code, start, end)
        dart, dart_setup_error = None, None
        if not args.prices_only:
            try:
                key = stored_key()
                if key:
                    dart = DartClient(key)
            except Exception:
                dart_setup_error = "DART 보안 설정을 읽지 못했습니다. DART_SETUP.cmd로 확인해주세요."
        path, batch = run_batch(codes, start, end, args.data_root, args.report_root,
            source_factory=lambda: KiwoomSource(args.profile), dart=dart, use_dart=not args.prices_only,
            dart_setup_error=dart_setup_error)
        print(f"\n가격 수집 성공: {sum(s['price_status'] == 'collected' for s in batch['stocks'])}/{len(batch['stocks'])}종목")
        print(f"관찰 사례: {batch['feature_rows']}개 · 결과 보고서: {path / 'report.html'}")
        if batch["dart_state"] == "waiting_key":
            print("재무·공시 연결은 DART 인증키 등록 후 가능합니다. DART_SETUP.cmd를 사용해주세요.")
        if batch["status"] != "collected":
            print("대기·실패 항목이 있습니다. 보고서의 종목별 상태를 확인해주세요.")
        print("모델 학습과 자동매매는 아직 실행하지 않았습니다.")
        if sys.platform == "win32" and not args.no_open:
            webbrowser.open((path / "report.html").resolve().as_uri())
        return 0 if batch["status"] == "collected" else 2
    except CollectionError as exc:
        print(f"준비 실패: {exc}", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print("작업을 취소했습니다.", file=sys.stderr)
        return 130
    except Exception:
        print("준비 작업을 완료하지 못했습니다. 오류가 발생한 단계를 알려주세요.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
