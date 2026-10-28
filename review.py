"""Read-only diagnosis of existing models. Never fits, promotes, or sends an order."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np

from financial_history import FINANCIAL_COLUMNS
from learning import MODEL_FEATURES, _matrix, index_outcomes, metrics, partition_examples, predict
from lyupang import CollectionError, SEOUL, parse_date
from research_data import FEATURE_COLUMNS, HORIZONS, OUTCOME_COLUMNS, write_table

CASE_COLUMNS = ("horizon", "code", "as_of_date", "exit_date", "actual_change_pct", "predicted_change_pct",
                "baseline_change_pct", "absolute_error_pp", "baseline_absolute_error_pp")


def _json(path):
    def invalid(_):
        raise CollectionError("JSON에 유효하지 않은 수치가 있습니다.")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def _table(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(columns):
            raise CollectionError("학습 결과의 CSV 열 구성이 지원 형식과 다릅니다.")
        return list(reader)


def load_source(folder: Path):
    summary = _json(folder / "summary.json")
    for name, key in (("features_with_financials.csv", "feature_sha256"), ("future_outcomes.csv", "outcome_sha256")):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != summary.get(key):
            raise CollectionError("저장된 학습 자료의 해시가 일치하지 않습니다. 원본을 확인해주세요.")
    features = _table(folder / "features_with_financials.csv", FEATURE_COLUMNS + FINANCIAL_COLUMNS)
    outcomes = _table(folder / "future_outcomes.csv", OUTCOME_COLUMNS)
    for row in features:
        if not re.fullmatch(r"[0-9]{6}", row["code"]):
            raise CollectionError("학습 자료의 종목코드가 올바르지 않습니다.")
        parse_date(row["as_of_date"])
        for key in ("quality_issue", "dart_available", "financial_available"):
            if row[key] not in ("True", "False"):
                raise CollectionError("학습 자료의 상태 표시가 올바르지 않습니다.")
            row[key] = row[key] == "True"
    for row in outcomes:
        row["observed_sessions"] = int(row["observed_sessions"])
        if row["observed_sessions"] not in HORIZONS or row["status"] not in ("observed", "pending", "quality_issue"):
            raise CollectionError("미래 결과의 상태 또는 관찰 구간이 올바르지 않습니다.")
        row["price_return_pct"] = float(row["price_return_pct"]) if row["price_return_pct"] else None
    return _json(folder / "models.json"), features, outcomes


def validate_model(model):
    if model.get("features") != list(MODEL_FEATURES):
        raise CollectionError("지원하지 않는 모델 입력 구성입니다.")
    for key in ("median", "mean", "scale", "coefficients"):
        values = model.get(key)
        if not isinstance(values, list) or len(values) != len(MODEL_FEATURES):
            raise CollectionError("모델 수치의 크기가 올바르지 않습니다.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise CollectionError("모델에 유효하지 않은 수치가 있습니다.")
    intercept = model.get("intercept")
    if (any(v <= 0 for v in model["scale"]) or isinstance(intercept, bool)
            or not isinstance(intercept, (int, float)) or not math.isfinite(intercept)):
        raise CollectionError("모델 보정값이 올바르지 않습니다.")


def aggregate(cases):
    actual = np.array([c["actual_change_pct"] for c in cases])
    predicted = np.array([c["predicted_change_pct"] for c in cases])
    mae = float(np.mean([c["absolute_error_pp"] for c in cases]))
    baseline = float(np.mean([c["baseline_absolute_error_pp"] for c in cases]))
    return {"rows": len(cases), "unique_dates": len({c["as_of_date"] for c in cases}),
            "mae_pp": mae, "baseline_mae_pp": baseline,
            "relative_error_reduction_pct": (baseline - mae) / baseline * 100 if baseline > 0 else None,
            "direction_accuracy": float(np.mean((predicted > 0) == (actual > 0))),
            "actual_up_fraction": float(np.mean(actual > 0)),
            "always_up_direction_accuracy": float(np.mean(actual > 0))}


def diagnose(models, features, outcomes):
    result = {"schema_version": 1, "status": "reviewed", "models": {}, "live_trading_allowed": False,
              "automatic_promotion": False, "fresh_unseen_evidence": False, "test_period_consumed": True,
              "statistical_significance_established": False, "overlapping_labels": True}
    cases = []
    trained = {h: m for h, m in models.get("models", {}).items() if m.get("status") == "trained"}
    result["untrained_horizons"] = [h for h in map(str, HORIZONS) if h not in trained]
    if not trained:
        result["status"] = "no_trained_models"
        return result, cases
    days = sorted({f["as_of_date"] for f in features})
    if len(days) < 30 or len(features) != len({(f["code"], f["as_of_date"]) for f in features}):
        raise CollectionError("모델과 비교할 관찰 날짜가 부족하거나 중복됐습니다.")
    split = models.get("split", {})
    if (split.get("policy") != "global_date_60_20_20_exit_date_purge" or
        split.get("validation_start") != days[int(len(days) * 0.6)] or split.get("test_start") != days[int(len(days) * 0.8)]):
        raise CollectionError("모델과 자료의 기간 분리 기록이 일치하지 않습니다.")
    indexed = index_outcomes(outcomes)
    for horizon, model in trained.items():
        if int(horizon) not in HORIZONS:
            raise CollectionError("지원하지 않는 모델 관찰 구간입니다.")
        validate_model(model)
        partitions, _ = partition_examples(features, indexed, int(horizon), split["validation_start"], split["test_start"])
        test = partitions["test"]
        counts = {name: len(rows) for name, rows in partitions.items()}
        if not test or counts != model.get("counts") or model.get("training_rows") != counts["train"] + counts["validation"]:
            raise CollectionError("모델과 평가 자료의 사례 수가 일치하지 않습니다.")
        fitting = partitions["train"] + partitions["validation"]
        if max(r[1]["exit_date"] for r in fitting) != model.get("training_label_end") or model["training_label_end"] >= split["test_start"]:
            raise CollectionError("모델의 학습 기간 기록이 평가 구간과 맞지 않습니다.")
        computed = metrics(model, test)
        for key, value in computed.items():
            if key not in model["test_metrics"] or not math.isclose(float(model["test_metrics"][key]), float(value), rel_tol=1e-8, abs_tol=1e-8):
                raise CollectionError("저장된 평가 결과와 재계산한 결과가 다릅니다. 모델·자료 조합을 확인해주세요.")
        predicted = predict(model, test)
        if not np.isfinite(predicted).all():
            raise CollectionError("모델 예측에 유효하지 않은 수치가 있습니다.")
        selected = []
        for (feature, outcome), prediction in zip(test, predicted):
            actual = float(outcome["price_return_pct"])
            selected.append({"horizon": int(horizon), "code": feature["code"], "as_of_date": feature["as_of_date"],
                "exit_date": outcome["exit_date"], "actual_change_pct": actual, "predicted_change_pct": float(prediction),
                "baseline_change_pct": model["intercept"], "absolute_error_pp": abs(actual - prediction),
                "baseline_absolute_error_pp": abs(actual - model["intercept"])})
        overall = aggregate(selected)
        groups = {"symbols": {}, "quarters": {}}
        for case in selected:
            day = parse_date(case["as_of_date"])
            groups["symbols"].setdefault(case["code"], []).append(case)
            groups["quarters"].setdefault(f"{day.year}-Q{(day.month-1)//3+1}", []).append(case)
        grouped = {kind: {key: aggregate(rows) for key, rows in sorted(values.items())} for kind, values in groups.items()}
        test_x, fit_x = _matrix(test), _matrix(fitting)
        filled = np.where(np.isfinite(test_x), test_x, model["median"])
        fitted = np.where(np.isfinite(fit_x), fit_x, model["median"])
        normalized = (filled - model["mean"]) / model["scale"]
        contributions = normalized * model["coefficients"]
        inputs = []
        for i, key in enumerate(MODEL_FEATURES):
            constant = bool(fitted[:, i].std() < 1e-12)
            inputs.append({"feature": key, "observed_test_rows": int(np.isfinite(test_x[:, i]).sum()),
                "constant_during_fit": constant,
                "test_mean_shift_in_fit_std": None if constant else float(normalized[:, i].mean()),
                "mean_prediction_contribution_pp": float(contributions[:, i].mean())})
        result["models"][horizon] = {"disposition": "rejected" if overall["mae_pp"] >= overall["baseline_mae_pp"] else "hold",
            "reason": "worse_or_equal_to_baseline" if overall["mae_pp"] >= overall["baseline_mae_pp"] else "new_validation_required",
            "overall": overall, **grouped, "inputs": inputs,
            "observed_start": min(c["as_of_date"] for c in selected), "label_end": max(c["exit_date"] for c in selected)}
        cases.extend(selected)
    return result, cases


def render_report(result):
    esc = lambda value: html.escape(str(value))
    fmt = lambda value: "—" if value is None else f"{value:.3f}"
    table, sections = [], []
    for horizon, model in result["models"].items():
        stats = model["overall"]
        disposition = "탈락 — 기준보다 오차가 크거나 같음" if model["disposition"] == "rejected" else "보류 — 새 검증 필요"
        table.append(f"<tr><td>{esc(horizon)}</td><td>{stats['rows']} / {stats['unique_dates']}</td><td>{fmt(stats['mae_pp'])}</td>"
                     f"<td>{fmt(stats['baseline_mae_pp'])}</td><td>{fmt(stats['relative_error_reduction_pct'])}%</td><td>{esc(disposition)}</td></tr>")
        detail = []
        for kind, label in (("symbols", "종목별"), ("quarters", "분기별")):
            rows = "".join(f"<tr><td>{esc(key)}</td><td>{s['rows']}</td><td>{fmt(s['mae_pp'])}</td><td>{fmt(s['baseline_mae_pp'])}</td>"
                           f"<td>{s['direction_accuracy']*100:.1f}%</td><td>{s['actual_up_fraction']*100:.1f}%</td></tr>" for key, s in model[kind].items())
            detail.append(f"<h3>{label}</h3><table><tr><th>구분</th><th>사례</th><th>모델 오차</th><th>기준 오차</th><th>상승 여부 일치</th><th>실제 상승 비율</th></tr>{rows}</table>")
        names = {"return_5_pct": "과거 5일 변화", "return_20_pct": "과거 20일 변화", "return_60_pct": "과거 60일 변화",
                 "volume_ratio_previous_20": "거래량 비율", "disclosures_previous_7_days": "이전 공시 건수",
                 "debt_ratio_pct": "부채/자산 비율", "net_margin_pct": "순이익률", "revenue_yoy_pct": "매출 증가율"}
        inputs = "".join(f"<tr><td>{esc(names[r['feature']])}</td><td>{r['observed_test_rows']}</td>"
                         f"<td>{'학습 때 변화 없음' if r['constant_during_fit'] else fmt(r['test_mean_shift_in_fit_std'])}</td>"
                         f"<td>{fmt(r['mean_prediction_contribution_pp'])}</td></tr>" for r in model["inputs"])
        sections.append(f"<section><h2>{esc(horizon)}거래일 모델</h2>{''.join(detail)}<h3>학습 때와 달라진 입력</h3>"
                        f"<table><tr><th>입력</th><th>평가 때 값 있음</th><th>평균 이동/학습 표준편차</th><th>모델 계산상 평균 영향(%p)</th></tr>{inputs}</table></section>")
    empty = "학습이 완료된 모델이 없어 성능 재계산을 하지 않았습니다. 기존 학습 대기 이유를 먼저 확인해주세요." if result["status"] == "no_trained_models" else ""
    return f"""<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Lyupang 모델 성능 진단</title><style>body{{max-width:1100px;margin:35px auto;padding:0 20px;font-family:system-ui,sans-serif;line-height:1.8;color:#17263c}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:10px;text-align:left;font-size:14px}}section{{margin:35px 0}}.note{{background:#fff2d5;border-radius:10px;padding:18px}}</style>
<h1>모델 성능 진단</h1><p class='note'>현재 자동 채택한 모델은 없습니다. 학습·매매는 실행하지 않고 기존 결과만 재계산했습니다. 기준보다 못하면 탈락, 개선돼도 새 검증 전에는 보류합니다.</p>
<h2>전체 결과</h2><p>{esc(empty)}</p><p>오차는 작을수록 좋습니다. 개선율이 음수이면 기준보다 나쁩니다. 이것은 예측 오차이며 실제 거래 수익률이 아닙니다.</p>
<table><tr><th>관찰 구간</th><th>사례/서로 다른 날짜</th><th>모델 오차(%p)</th><th>기준 오차(%p)</th><th>오차 개선율</th><th>자동 판정</th></tr>{''.join(table)}</table>
<p>같은 날짜의 여러 종목과 겹치는 관찰 구간은 독립 표본이 아닙니다. 이 진단은 개선의 통계적 유의성을 입증하지 않습니다.</p>
{''.join(sections)}<section><h2>표를 읽는 방법</h2><p>실제 상승 비율은 평가 자료에서 상승한 사례의 비율이며, 항상 상승이라고 답했을 때의 일치율과 같습니다. 이 기준도 함께 확인해야 합니다. 유리한 종목·분기만 고르면 전체 성능이 검증되는 것은 아닙니다.</p>
<p>입력의 평균 이동이 크면 학습 때와 다른 값에서 예측했을 가능성이 있습니다. 계산상 영향은 모델 식의 기여도이며 실제 상승 원인이나 인과관계를 뜻하지 않습니다.</p></section>
<section><h2>평가 기간 사용 기록</h2><p>이 결과를 보고 모델을 수정했다면 같은 기간을 새로운 미사용 평가 자료로 주장하지 않습니다. 새 기간의 검증이나 앞으로 기록할 모의매매가 필요합니다. 이번 열람은 평가 사용 기록에 보관합니다.</p>
<p>이전 기록과 기간이 겹침: {'예' if result.get('overlaps_previous_review') else '기록상 확인되지 않음'}. 기록상 처음이어도 과거에 보지 않았다는 뜻은 아닙니다.</p>
<p>사업·뉴스 의미 분석, 시장 전체·상장폐지 종목, 과거 수정주가·거래일 달력, 비용과 체결·모의매매 검증은 남아 있습니다. 실제 주문은 연결하지 않았습니다.</p></section></html>"""


def latest_learning(root):
    candidates = []
    if root.exists():
        for folder in root.iterdir():
            if not folder.is_dir() or folder.is_symlink() or folder.name.startswith("."):
                continue
            try:
                captured = datetime.fromisoformat(_json(folder / "summary.json")["created_at"])
                if captured.tzinfo is not None:
                    candidates.append((captured, folder))
            except (OSError, ValueError, KeyError, TypeError, CollectionError):
                continue
    if not candidates:
        raise CollectionError("저장된 학습 결과가 없습니다. 먼저 TRAIN.cmd를 실행해주세요.")
    return max(candidates, key=lambda item: item[0])[1]


def run(input_folder: Path, output_root: Path, registry_root: Path):
    models, features, outcomes = load_source(input_folder)
    result, cases = diagnose(models, features, outcomes)
    result["created_at"] = datetime.now(SEOUL).isoformat()
    result["source_learning"] = str(input_folder.resolve())
    result["source_model_sha256"] = hashlib.sha256((input_folder / "models.json").read_bytes()).hexdigest()
    result["source_feature_sha256"] = hashlib.sha256((input_folder / "features_with_financials.csv").read_bytes()).hexdigest()
    result["source_outcome_sha256"] = hashlib.sha256((input_folder / "future_outcomes.csv").read_bytes()).hexdigest()
    symbols = sorted({case["code"] for case in cases})
    start = min((case["as_of_date"] for case in cases), default=None)
    end = max((case["exit_date"] for case in cases), default=None)
    overlap = False
    if registry_root.exists() and cases:
        for file in registry_root.glob("*.json"):
            try:
                prior = _json(file)
                if set(symbols).intersection(prior["symbols"]) and start <= prior["label_information_end"] and prior["observation_start"] <= end:
                    overlap = True
            except (OSError, ValueError, KeyError, TypeError, CollectionError):
                # Never turn unreadable evidence into permission to promote a model.
                result["registry_record_unreadable"] = True
    result["overlaps_previous_review"] = overlap
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=output_root))
    target = output_root / f"{datetime.now(SEOUL):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    try:
        write_table(staging / "test_predictions.csv", CASE_COLUMNS, cases)
        (staging / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (staging / "report.html").write_text(render_report(result), encoding="utf-8")
        if cases:
            registry_root.mkdir(parents=True, exist_ok=True)
            event = {"created_at": result["created_at"], "symbols": symbols, "observation_start": start,
                     "label_information_end": end, "source_model_sha256": result["source_model_sha256"],
                     "usage": "diagnostic_consumed", "fresh_unseen_evidence": False}
            with (registry_root / (uuid.uuid4().hex + ".json")).open("x", encoding="utf-8") as stream:
                json.dump(event, stream, ensure_ascii=False, indent=2)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return target, result


def main(argv=None):
    root = Path(os.environ["LOCALAPPDATA"]) / "Lyupang" if sys.platform == "win32" else Path("data")
    parser = argparse.ArgumentParser(description="저장된 모델의 성능 진단")
    parser.add_argument("--input", type=Path, help="학습 결과 폴더; 생략하면 최신")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = args.input or latest_learning(args.root / "learning")
        target, result = run(source, args.root / "diagnostics", args.root / "evaluation_registry")
        for horizon, model in result["models"].items():
            status = "탈락" if model["disposition"] == "rejected" else "보류"
            print(f"{horizon}거래일 모델: {status} · 모델 오차 {model['overall']['mae_pp']:.3f} / 기준 {model['overall']['baseline_mae_pp']:.3f}")
        print("자동 채택 모델: 0개 · 실제 주문은 연결하지 않았습니다.")
        print(f"진단 보고서: {target / 'report.html'}")
        if sys.platform == "win32" and not args.no_open:
            webbrowser.open((target / "report.html").resolve().as_uri())
        return 0
    except CollectionError as exc:
        print(f"진단 실패: {exc}", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print("진단을 취소했습니다.", file=sys.stderr)
        return 130
    except Exception:
        print("저장된 모델·자료를 진단하지 못했습니다. 실행 창의 마지막 메시지를 알려주세요.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
