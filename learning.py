"""Small research-only ridge models with purged chronological evaluation."""
from __future__ import annotations

import math
from datetime import datetime, time

import numpy as np

from lyupang import CollectionError, SEOUL, parse_date
from research_data import HORIZONS

MODEL_FEATURES = ("return_5_pct", "return_20_pct", "return_60_pct", "volume_ratio_previous_20",
                  "disclosures_previous_7_days", "debt_ratio_pct", "net_margin_pct", "revenue_yoy_pct")


def _number(value):
    if value is None or value == "":
        return np.nan
    if isinstance(value, bool):
        raise CollectionError("학습용 수치 형식이 올바르지 않습니다.")
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise CollectionError("학습용 수치를 읽지 못했습니다.") from None
    return result if math.isfinite(result) else np.nan


def _matrix(rows):
    return np.array([[_number(r[0].get(key)) for key in MODEL_FEATURES] for r in rows], dtype=float)


def fit(rows, alpha):
    x = _matrix(rows)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in x.T])
    filled = np.where(np.isfinite(x), x, median)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (filled - mean) / scale
    y = np.array([r[1]["price_return_pct"] for r in rows], dtype=float)
    intercept = float(y.mean())
    coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(len(MODEL_FEATURES)), z.T @ (y - intercept))
    return {"features": list(MODEL_FEATURES), "median": median.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": coefficients.tolist(), "intercept": intercept, "alpha": alpha,
            "input_observed_counts": {key: int(np.isfinite(x[:, i]).sum()) for i, key in enumerate(MODEL_FEATURES)},
            "training_rows": len(rows), "training_label_end": max(r[1]["exit_date"] for r in rows)}


def predict(model, rows):
    x = _matrix(rows)
    x = np.where(np.isfinite(x), x, np.array(model["median"]))
    return (x - model["mean"]) / model["scale"] @ np.array(model["coefficients"]) + model["intercept"]


def metrics(model, rows):
    y = np.array([r[1]["price_return_pct"] for r in rows], dtype=float)
    predicted = predict(model, rows)
    baseline = np.full(len(rows), model["intercept"])
    return {"rows": len(rows), "mae_percentage_points": float(np.mean(np.abs(y - predicted))),
            "mse_percentage_points_squared": float(np.mean((y - predicted) ** 2)),
            "baseline_mae_percentage_points": float(np.mean(np.abs(y - baseline))),
            "baseline_mse_percentage_points_squared": float(np.mean((y - baseline) ** 2)),
            "direction_accuracy": float(np.mean((predicted > 0) == (y > 0)))}


def index_outcomes(outcomes):
    indexed = {}
    for row in outcomes:
        key = (row["code"], row["as_of_date"], row["observed_sessions"])
        if key in indexed:
            raise CollectionError("같은 종목·날짜·관찰 구간의 미래 결과가 중복됐습니다.")
        indexed[key] = row
    return indexed


def partition_examples(features, indexed, horizon, validation_start, test_start):
    """Shared by fitting and read-only review so the original test cohort is reproduced."""
    partitions = {"train": [], "validation": [], "test": []}
    excluded = {"financial_or_quality": 0, "future_unobserved": 0, "cross_boundary": 0}
    for feature in features:
        as_of = feature["as_of_date"]
        if feature.get("financial_available") is not True or feature.get("quality_issue"):
            excluded["financial_or_quality"] += 1
            continue
        if not feature.get("financial_receipt_date") or feature["financial_receipt_date"] >= as_of:
            raise CollectionError("학습 입력에 관찰일 이후의 재무 공시가 포함되어 있습니다.")
        outcome = indexed.get((feature["code"], as_of, horizon))
        if not outcome or outcome["status"] != "observed" or not math.isfinite(_number(outcome["price_return_pct"])):
            excluded["future_unobserved"] += 1
            continue
        if not as_of < outcome["entry_date"] <= outcome["exit_date"]:
            raise CollectionError("미래 결과의 매수·관찰 종료일이 올바르지 않습니다.")
        available = datetime.fromisoformat(feature["available_from"])
        if available.tzinfo is None or available > datetime.combine(parse_date(outcome["entry_date"]), time(9), SEOUL):
            raise CollectionError("입력 정보가 사용 가능해지기 전에 매수한 사례가 있습니다.")
        financial_available = datetime.fromisoformat(feature["financial_available_from"])
        if financial_available.tzinfo is None or financial_available > datetime.combine(parse_date(as_of), time.min, SEOUL):
            raise CollectionError("관찰일 시작 이후에 공개된 재무 정보가 학습에 포함됐습니다.")
        partition = "train" if as_of < validation_start else "validation" if as_of < test_start else "test"
        boundary = validation_start if partition == "train" else test_start if partition == "validation" else None
        if boundary and outcome["exit_date"] >= boundary:
            excluded["cross_boundary"] += 1
            continue
        partitions[partition].append((feature, outcome))
    return partitions, excluded


def train_models(features: list[dict], outcomes: list[dict], *, minimums=(100, 30, 30)) -> dict:
    if len(minimums) != 3 or any(not isinstance(n, int) or n < 2 for n in minimums):
        raise CollectionError("기간별 최소 표본 수 설정이 올바르지 않습니다.")
    days = sorted({f["as_of_date"] for f in features})
    result = {"status": "blocked", "research_only": True, "live_trading_allowed": False,
              "point_in_time_verified": False, "models": {}, "excluded": {},
              "target": "future_price_change_before_costs", "absolute_price_level_excluded": True,
              "evaluation_scope": "later_dates_of_selected_symbols",
              "feature_whitelist": list(MODEL_FEATURES)}
    if len(days) < 30:
        result["reason"] = "학습·검증·평가를 나눌 관찰 날짜가 부족합니다."
        return result
    validation_start, test_start = days[int(len(days) * 0.6)], days[int(len(days) * 0.8)]
    result["split"] = {"validation_start": validation_start, "test_start": test_start,
                       "policy": "global_date_60_20_20_exit_date_purge"}
    indexed = index_outcomes(outcomes)
    keys = [(f["code"], f["as_of_date"]) for f in features]
    if len(keys) != len(set(keys)):
        raise CollectionError("관찰 사례에 중복된 종목·날짜가 있습니다.")
    for horizon in HORIZONS:
        partitions, excluded = partition_examples(features, indexed, horizon, validation_start, test_start)
        result["excluded"][str(horizon)] = excluded
        counts = {name: len(rows) for name, rows in partitions.items()}
        if any(counts[name] < minimum for name, minimum in zip(partitions, minimums)):
            result["models"][str(horizon)] = {"status": "blocked", "reason": "검증된 재무 사례 또는 기간별 표본 수 부족", "counts": counts}
            continue
        candidates = []
        for alpha in (0.1, 1.0, 10.0, 100.0):
            model = fit(partitions["train"], alpha)
            validation = metrics(model, partitions["validation"])
            candidates.append((validation["mse_percentage_points_squared"], alpha, validation))
        _, alpha, validation = min(candidates, key=lambda c: (c[0], c[1]))
        # Test data never participates in imputation, scaling, tuning or final fitting.
        model = fit(partitions["train"] + partitions["validation"], alpha)
        if model["training_label_end"] >= test_start:
            raise CollectionError("최종 학습 구간의 미래 결과가 평가 기간과 겹칩니다.")
        model.update(status="trained", counts=counts, validation_metrics=validation, test_metrics=metrics(model, partitions["test"]))
        result["models"][str(horizon)] = model
    statuses = [model["status"] for model in result["models"].values()]
    result["status"] = "trained" if statuses and all(s == "trained" for s in statuses) else "partial" if "trained" in statuses else "blocked"
    return result
