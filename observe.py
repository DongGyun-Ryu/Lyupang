"""Forward-only prediction journal. No fitting, portfolio, or order API."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import sys
import tempfile
import time
import webbrowser
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, time as daytime, timedelta
from pathlib import Path

from dart_client import DartClient, stored_key
from financial_history import audit_reports, financial_as_of
from learning import MODEL_FEATURES, predict
from lyupang import CollectionError, KiwoomSource, SEOUL, collect, parse_date, save_collection
from research import safe_reason
from research_data import build_cases, load_snapshot
from review import aggregate, diagnose, latest_learning, load_source, validate_model


def local_now():
    return datetime.now(SEOUL)


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise CollectionError("예측 기록 시각에 시간대가 없습니다.")
    return result.astimezone(SEOUL)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    def invalid(_):
        raise CollectionError("기록에 유효하지 않은 수치가 있습니다.")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def append_record(path, payload):
    """An existing record is never overwritten; interrupted writes fail closed on reading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"payload": payload, "sha256": digest(payload)}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def read_record(path):
    wrapped = read_json(path)
    if digest(wrapped["payload"]) != wrapped["sha256"]:
        raise CollectionError("예측 또는 결과 기록의 해시가 다릅니다. 원본을 보존하고 확인해주세요.")
    return wrapped["payload"]


@contextmanager
def journal_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".running.lock"
    stream = lock.open("a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if not stream.tell():
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise CollectionError("예측 기록 프로그램이 이미 실행 중입니다. 다른 실행 창을 먼저 확인해주세요.") from None
    try:
        yield
    finally:
        # The OS releases the lock on close, process death, and PC restart. Keep the
        # file itself so another process can never lock a different inode by racing.
        stream.close()


def initialize(root, source, *, clock=local_now):
    experiment = root / "experiment"
    if experiment.exists():
        manifest = read_record(experiment / "experiment.json")
        models = read_json(experiment / "models.json")
        if (manifest.get("policy") != "forward_journal_v1" or manifest.get("features") != list(MODEL_FEATURES)
                or digest(models) != manifest["model_sha256"]):
            raise CollectionError("고정한 모델 파일이 변경됐습니다. 기존 예측과 섞지 않습니다.")
        for horizon in manifest["horizons"]:
            validate_model(models["models"][horizon])
        return experiment, manifest, models
    models, features, outcomes = load_source(source)
    diagnosis, _ = diagnose(models, features, outcomes)
    horizons = [h for h, m in diagnosis["models"].items() if m["disposition"] == "hold"]
    if not horizons:
        raise CollectionError("새 예측을 관찰할 보류 모델이 없습니다. 탈락 모델은 기록 대상으로 사용하지 않습니다.")
    created = clock()
    if timestamp(read_json(source / "summary.json")["created_at"]) > created:
        raise CollectionError("학습 결과의 생성 시각보다 PC 시각이 이전입니다. 시각을 확인해주세요.")
    manifest = {"schema_version": 1, "policy": "forward_journal_v1", "created_at": created.isoformat(), "source_learning": str(source.resolve()),
                "model_sha256": digest(models), "codes": sorted({f["code"] for f in features}), "horizons": horizons,
                "excluded_horizons": [h for h in diagnosis["models"] if h not in horizons],
                "features": list(MODEL_FEATURES), "target": "next_observed_session_open_to_horizon_close_before_costs",
                "source_test_already_consumed": True, "automatic_promotion": False, "live_trading_allowed": False,
                "trusted_timestamp": False, "exchange_calendar_verified": False}
    if len(manifest["codes"]) > 20 or timestamp(manifest["created_at"]) > created:
        raise CollectionError("관찰할 모델의 종목 수 또는 기록 시각이 올바르지 않습니다.")
    staging = Path(tempfile.mkdtemp(prefix=".initializing-", dir=root))
    try:
        (staging / "models.json").write_bytes(canonical(models))
        append_record(staging / "experiment.json", manifest)
        os.replace(staging, experiment)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return experiment, manifest, models


def prediction_window(now):
    # Existing model target is the next observed day's open, not a later substituted entry.
    return now.astimezone(SEOUL).time() < daytime(9)


def record_predictions(experiment, manifest, models, feature, bars, evidence, *, clock=local_now):
    recorded = clock().astimezone(SEOUL)
    as_of = parse_date(feature["as_of_date"])
    if not prediction_window(recorded):
        return 0  # Fetching crossed 09:00: never backdate a prediction.
    if (as_of >= recorded.date() or timestamp(feature["available_from"]) > recorded
            or timestamp(manifest["created_at"]) > recorded):
        raise CollectionError("예측 입력이나 고정 모델이 기록 시점에 사용 가능하지 않습니다.")
    if (feature.get("quality_issue") or feature.get("financial_available") is not True
            or feature.get("dart_available") is not True):
        raise CollectionError("가격·재무·공시 입력을 확인하지 못해 새 예측을 기록하지 않습니다.")
    if (feature["financial_receipt_date"] >= feature["as_of_date"]
            or timestamp(feature["financial_available_from"]) > recorded):
        raise CollectionError("예측에 사용할 재무 공개 시점이 올바르지 않습니다.")
    if bars[-1].date != feature["as_of_date"] or bars[-1].code != feature["code"]:
        raise CollectionError("최신 가격과 예측 입력의 종목·날짜가 다릅니다.")
    count = 0
    for horizon in manifest["horizons"]:
        model = models["models"][horizon]
        if model["training_label_end"] >= feature["as_of_date"]:
            raise CollectionError("학습에 사용한 결과 이후의 새 관찰일만 예측할 수 있습니다.")
        path = experiment / "predictions" / f"{feature['as_of_date']}_{feature['code']}_{horizon}.json"
        if path.exists():
            read_record(path)
            continue
        predicted = float(predict(model, [(feature, {})])[0])
        if not math.isfinite(predicted):
            raise CollectionError("예측값이 유효하지 않습니다.")
        recorded = clock().astimezone(SEOUL)
        if not prediction_window(recorded):
            break
        payload = {"schema_version": 1, "recorded_at": recorded.isoformat(), "code": feature["code"],
                   "as_of_date": feature["as_of_date"], "horizon": int(horizon),
                   "model_sha256": manifest["model_sha256"], "predicted_change_pct": predicted,
                   "baseline_change_pct": model["intercept"], "feature": feature, "evidence": evidence,
                   "anchor_bars": [asdict(b) for b in bars[-61:]], "live_trading_allowed": False}
        append_record(path, payload)
        # Receipt is captured after prediction data has been flushed to disk. A crash
        # before this receipt or a write spanning the cutoff cannot be rescued later.
        committed = clock().astimezone(SEOUL)
        append_record(experiment / "commits" / path.name,
                      {"prediction_sha256": digest(payload), "committed_at": committed.isoformat(),
                       "before_cutoff": prediction_window(committed)})
        count += 1
    return count


def settle(prediction, bars, commit):
    """None means pending. Never alter the original forecast or evaluate a late forecast."""
    if commit is None:
        return {"status": "excluded", "reason": "commit_missing"}
    if commit["prediction_sha256"] != digest(prediction):
        raise CollectionError("예측 저장 완료 기록의 해시가 다릅니다.")
    if not commit["before_cutoff"]:
        return {"status": "excluded", "reason": "commit_after_cutoff"}
    if timestamp(commit["committed_at"]) < timestamp(prediction["recorded_at"]):
        raise CollectionError("예측 저장 중 PC 시각이 이전으로 바뀌었습니다.")
    if ([b.date for b in bars] != sorted({b.date for b in bars})
            or any(b.code != prediction["code"] for b in bars)):
        raise CollectionError("결과 대조에는 같은 종목의 정렬된 일봉이 필요합니다.")
    indexed = {b.date: b for b in bars}
    for anchor in prediction["anchor_bars"]:
        current = indexed.get(anchor["date"])
        if current is None:
            return {"status": "excluded", "reason": "anchor_history_missing"}
        if asdict(current) != anchor:
            return {"status": "excluded", "reason": "price_history_revised"}
    future = [b for b in bars if b.date > prediction["as_of_date"]]
    if not future:
        return None
    entry_at = datetime.combine(parse_date(future[0].date), daytime(9), SEOUL)
    if max(timestamp(prediction["recorded_at"]), timestamp(commit["committed_at"])) >= entry_at:
        return {"status": "excluded", "reason": "recorded_after_entry"}
    if len(future) < prediction["horizon"]:
        return None
    selected = future[:prediction["horizon"]]
    if any(b.open <= 0 or b.close <= 0 or b.zero_volume or b.volume_capped for b in selected):
        return {"status": "excluded", "reason": "future_price_quality"}
    first, last = selected[0], selected[-1]
    actual = (last.close / first.open - 1) * 100
    return {"status": "observed", "code": prediction["code"], "as_of_date": prediction["as_of_date"],
            "horizon": prediction["horizon"], "entry_date": first.date, "entry_open": first.open,
            "exit_date": last.date, "exit_close": last.close, "actual_change_pct": actual,
            "predicted_change_pct": prediction["predicted_change_pct"],
            "baseline_change_pct": prediction["baseline_change_pct"],
            "absolute_error_pp": abs(actual - prediction["predicted_change_pct"]),
            "baseline_absolute_error_pp": abs(actual - prediction["baseline_change_pct"])}


def records(experiment, kind):
    return {p.stem: read_record(p) for p in sorted((experiment / kind).glob("*.json"))}


def summarize(experiment, manifest, *, now, new=0, errors=None, notices=None):
    predictions, results = records(experiment, "predictions"), records(experiment, "results")
    if set(results) - set(predictions):
        raise CollectionError("대응하는 예측이 없는 결과 기록이 있습니다.")
    for key, prediction in predictions.items():
        if prediction["model_sha256"] != manifest["model_sha256"]:
            raise CollectionError("서로 다른 모델의 예측 기록이 섞여 있습니다.")
        if key in results and results[key]["prediction_sha256"] != digest(prediction):
            raise CollectionError("결과와 원래 예측의 해시가 일치하지 않습니다.")
        if timestamp(prediction["recorded_at"]) > now:
            raise CollectionError("PC 시각이 저장된 예측보다 이전입니다. 시각을 확인해주세요.")
    metrics = {}
    for horizon in manifest["horizons"]:
        cases = [r for r in results.values() if r["status"] == "observed" and str(r["horizon"]) == horizon]
        if cases:
            metrics[horizon] = aggregate(cases)
    return {"updated_at": now.isoformat(), "experiment_created_at": manifest["created_at"],
            "horizons": manifest["horizons"], "excluded_horizons": manifest["excluded_horizons"],
            "new_predictions": new, "total_predictions": len(predictions), "completed": sum(r["status"] == "observed" for r in results.values()),
            "excluded": sum(r["status"] == "excluded" for r in results.values()), "pending": len(predictions) - len(results),
            "exclusion_reasons": {reason: sum(r.get("reason") == reason for r in results.values())
                                  for reason in sorted({r["reason"] for r in results.values() if r["status"] == "excluded"})},
            "metrics": metrics, "errors": errors or [], "notices": notices or [], "automatic_promotion": False,
            "live_trading_allowed": False, "exchange_calendar_verified": False, "trusted_timestamp": False,
            "statistical_significance_established": False}


def render_report(summary):
    esc = lambda v: html.escape(str(v))
    rows = []
    for horizon in summary["horizons"]:
        stats = summary["metrics"].get(horizon)
        cells = (f"<td>{stats['rows']} / {stats['unique_dates']}</td><td>{stats['mae_pp']:.3f}</td>"
                 f"<td>{stats['baseline_mae_pp']:.3f}</td><td>{stats['direction_accuracy']*100:.1f}%</td><td>{stats['actual_up_fraction']*100:.1f}%</td>"
                 if stats else "<td colspan='5'>결과 대기</td>")
        rows.append(f"<tr><td>{esc(horizon)}일봉</td>{cells}</tr>")
    messages = "".join(f"<li>{esc(message)}</li>" for message in summary["notices"] + summary["errors"])
    reasons = {"commit_missing": "예측 저장 완료 기록 없음", "commit_after_cutoff": "오전 9시 이후 저장 완료",
               "anchor_history_missing": "기존 가격 이력 누락", "price_history_revised": "기존 가격 이력 변경",
               "recorded_after_entry": "관찰 시작 뒤 저장된 예측", "future_price_quality": "이후 가격·거래량 품질 문제"}
    excluded = " · ".join(f"{reasons.get(k, k)}: {v}개" for k, v in summary["exclusion_reasons"].items()) or "없음"
    return f"""<!doctype html><html lang='ko'><meta charset='utf-8'><meta http-equiv='refresh' content='60'>
<meta name='viewport' content='width=device-width'><title>Lyupang 새 예측 관찰</title>
<style>body{{max-width:1000px;margin:35px auto;padding:0 20px;font-family:system-ui,sans-serif;line-height:1.8;color:#17263c}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:10px;text-align:left}}.note{{background:#fff2d5;padding:18px;border-radius:10px}}</style>
<h1>새 예측 관찰</h1><p class='note'>실제 주문과 모델 재학습 없이, 미리 저장한 예측을 이후 가격과 비교합니다. 자동 채택한 모델은 없습니다.</p>
<p>모델 고정: {esc(summary['experiment_created_at'])}<br>마지막 확인: {esc(summary['updated_at'])}</p>
<h2>기록 상태</h2><table><tr><th>이번 새 예측</th><th>누적 예측</th><th>결과 대기</th><th>결과 확인</th><th>검증 제외</th></tr>
<tr><td>{summary['new_predictions']}</td><td>{summary['total_predictions']}</td><td>{summary['pending']}</td><td>{summary['completed']}</td><td>{summary['excluded']}</td></tr></table><ul>{messages}</ul><p>검증 제외 사유: {esc(excluded)}</p>
<h2>새 기간의 예측 오차</h2><p>오차는 작을수록 좋습니다. 비교 기준은 고정 모델의 학습 당시 평균 변화입니다. 가격 변화는 전략 수익률이 아닙니다.</p>
<table><tr><th>관찰 구간</th><th>사례 / 서로 다른 날짜</th><th>모델 오차(%p)</th><th>기준 오차(%p)</th><th>상승 여부 일치</th><th>실제 상승 비율</th></tr>{''.join(rows)}</table>
<p>실제 상승 비율은 항상 상승이라고 답했을 때의 일치율입니다.</p>
<h2>자동 확인 시간</h2><p>창을 켜둔 동안 한국 시간 매일 오전 8시 30분에 확인합니다. 처음 실행할 때도 한 번 확인합니다. 오전 9시 이후에는 새 예측을 만들지 않고 기존 결과만 대조합니다. 컴퓨터가 꺼져 있거나 잠들어 놓친 날은 나중에 채워 넣지 않습니다.</p>
<p>다시 실행하면 같은 고정 모델과 기록을 이어갑니다. 재학습한 모델로 몰래 바꾸지 않습니다. 탈락해 제외한 관찰 구간: {esc(', '.join(summary['excluded_horizons']) or '없음')}.</p>
<h2>검증의 범위</h2><p>저장 시각 이후에 시작한 다음 일봉 시가부터 5·20개 등 해당 수의 일봉 마지막 종가까지 비교합니다. 보유 기간을 선택하거나 매수·매도를 실행하는 기능이 아닙니다. 수수료·세금·슬리피지와 체결은 반영하지 않았습니다.</p>
<p>원래 가격 이력이 바뀌거나 거래량 등 품질 문제가 생기면 검증에서 제외합니다. 거래소 달력과 대조하지 않아 실제 거래일 누락·특별 개장 시각·거래정지를 완전히 검증하지 못합니다. 비교에는 이후 조회 시점의 수정주가가 쓰입니다.</p>
<p>기록과 해시는 사용자 PC에 저장하며 외부에서 인증한 시각이나 위변조 방지 증명이 아닙니다. PC 시각이 정확해야 합니다. 겹치는 관찰 구간은 독립 표본이 아니며, 이 기록만으로 통계적 유의성·시장 전체 성능을 입증하지 않습니다. 사업·뉴스 본문 의미 분석과 모의매매는 아직 없습니다.</p></html>"""


def write_report(experiment, summary):
    # Only the derived dashboard is replaced; source predictions and results are append-only.
    for name, content in (("summary.json", canonical(summary)), ("report.html", render_report(summary).encode("utf-8"))):
        temporary = experiment / ("." + name + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, experiment / name)


def run_cycle(root, source_learning=None, *, source_factory=KiwoomSource, dart_factory=None,
              clock=local_now, progress=print, sleep=time.sleep):
    forward = root / "forward"
    with journal_lock(forward):
        source_learning = source_learning or (None if (forward / "experiment").exists() else latest_learning(root / "learning"))
        experiment, manifest, models = initialize(forward, source_learning, clock=clock)
        prior, results = records(experiment, "predictions"), records(experiment, "results")
        now = clock().astimezone(SEOUL)
        errors, notices, new = [], [], 0
        if not prediction_window(now):
            notices.append("새 예측 대기 — 한국 시간 오전 9시 전 실행해야 기록할 수 있습니다. 자동 확인은 다음 오전 8시 30분입니다.")
        pending = {k: p for k, p in prior.items() if k not in results}
        wants_new = prediction_window(now)
        source, dart = None, None
        companies = {}
        if wants_new:
            try:
                key = None if dart_factory else stored_key()
                if not dart_factory and not key:
                    raise CollectionError("기존 DART 키를 찾지 못했습니다. DART_SETUP.cmd로 확인해주세요.")
                dart = dart_factory() if dart_factory else DartClient(key)
                companies = dart.companies(manifest["codes"])
            except Exception as exc:
                errors.append("재무·공시 연결 확인 필요: " + safe_reason(exc))
                wants_new = False
        for code in manifest["codes"]:
            items = {k: p for k, p in pending.items() if p["code"] == code}
            if not wants_new and not items:
                continue
            try:
                progress(f"종목 {code} · 새 예측과 이전 기록 확인 중...")
                end = now.date() - timedelta(days=1)
                start = end - timedelta(days=400)
                if items:
                    start = min(start, min(parse_date(p["anchor_bars"][0]["date"]) for p in items.values()))
                if source is None:
                    source = source_factory()
                collection = collect(source, code, start, end, sleep=sleep)
                snapshot = save_collection(collection, root / "data", now=clock())
                metadata, bars = load_snapshot(snapshot)
                evidence = {"price_snapshot": str(snapshot.resolve()), "price_sha256": metadata["csv_sha256"]}
                for key, prediction in items.items():
                    commit_path = experiment / "commits" / f"{key}.json"
                    commit = read_record(commit_path) if commit_path.exists() else None
                    outcome = settle(prediction, bars, commit)
                    if outcome is not None:
                        outcome.update(prediction_sha256=digest(prediction), checked_at=clock().isoformat(), evidence=evidence)
                        append_record(experiment / "results" / f"{key}.json", outcome)
                if wants_new and prediction_window(clock()):
                    latest = bars[-1].date
                    missing = any(not (experiment / "predictions" / f"{latest}_{code}_{h}.json").exists() for h in manifest["horizons"])
                    if missing:
                        company = companies.get(code)
                        if not company:
                            raise CollectionError("DART 기업번호를 확인하지 못했습니다.")
                        as_of = parse_date(latest)
                        filings = dart.disclosures(company["corp_code"], as_of - timedelta(days=7), as_of)
                        events = audit_reports(code, company["corp_code"], as_of, end, dart, root / "financial_documents", progress=progress)
                        features, _ = build_cases(bars, filings)
                        if not features:
                            raise CollectionError("새 예측에 필요한 61개 일봉이 없습니다.")
                        feature = {**features[-1], **financial_as_of(events, as_of)}
                        input_evidence = {**evidence, "filings": filings, "financial_audit": events}
                        count = record_predictions(experiment, manifest, models, feature, bars, input_evidence, clock=clock)
                        new += count
                        progress(f"  새 예측 {count}개 기록")
            except Exception as exc:
                errors.append(f"{code}: " + safe_reason(exc))
            finally:
                sleep(0.6)
        summary = summarize(experiment, manifest, now=clock(), new=new, errors=errors, notices=notices)
        write_report(experiment, summary)
        return experiment, summary


def next_check(now, failed=False):
    now = now.astimezone(SEOUL)
    retry = now + timedelta(minutes=5)
    if failed and daytime(8, 30) <= now.time() < daytime(8, 55) and retry.time() < daytime(9):
        return retry
    today = datetime.combine(now.date(), daytime(8, 30), SEOUL)
    return today if now < today else today + timedelta(days=1)


def main(argv=None):
    root = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="고정 모델의 새 예측 기록과 이후 결과 자동 비교")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--input", type=Path, help="최초 고정할 기존 학습 결과; 이후에는 고정 모델을 계속 사용")
    parser.add_argument("--profile", default="kiwoom-real")
    parser.add_argument("--watch", action="store_true", help="창을 켜둔 동안 매일 한국 시간 오전 8시 30분 확인")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    opened = False
    try:
        while True:
            target, summary = run_cycle(args.root, args.input, source_factory=lambda: KiwoomSource(args.profile))
            print(f"\n새 예측 {summary['new_predictions']}개 · 결과 대기 {summary['pending']}개 · 결과 확인 {summary['completed']}개 · 검증 제외 {summary['excluded']}개")
            for message in summary["notices"] + summary["errors"]:
                print(message)
            print("자동 채택 모델: 0개 · 실제 주문은 실행하지 않습니다.")
            print(f"관찰 보고서: {target / 'report.html'}")
            if not opened and sys.platform == "win32" and not args.no_open:
                webbrowser.open((target / "report.html").resolve().as_uri())
                opened = True
            if not args.watch:
                return 2 if summary["errors"] else 0
            scheduled = next_check(local_now(), bool(summary["errors"]))
            print(f"다음 자동 확인: {scheduled:%Y-%m-%d %H:%M} 한국 시간 · 이 창과 컴퓨터를 켜두세요. 종료: Ctrl+C", flush=True)
            while local_now() < scheduled:
                time.sleep(min(30, max(0, (scheduled - local_now()).total_seconds())))
    except (KeyboardInterrupt, EOFError):
        print("관찰을 종료했습니다. 저장된 기록은 다음 실행에서 이어갑니다.")
        return 130
    except CollectionError as exc:
        print(f"관찰 확인 필요: {exc}", file=sys.stderr)
    except Exception:
        print("관찰을 완료하지 못했습니다. 실행 창의 마지막 메시지를 알려주세요. 기존 기록은 덮어쓰지 않습니다.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
