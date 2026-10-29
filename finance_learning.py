"""Offline, receipt-specific financial inputs and paired development experiments.

Direct XBRL facts never certify a narrative claim or replace its rounded amount.
Selection uses publication metadata only, before any future outcome is read.
"""
from __future__ import annotations

from collections import Counter
from decimal import localcontext
import math

import numpy as np

from check_facts import number, require
from evidence_table import METRICS, SCOPES, snapshot_as_of
from learning import MODEL_FEATURES, _number, index_outcomes, partition_examples
from lyupang import parse_date

POLICY = "same_receipt_annual_operating_margin_research_v1"
EXTRA_FEATURES = ("source_operating_margin_pct", "source_operating_margin_change_pp")
FEATURE_NAMES = {EXTRA_FEATURES[0]: "영업이익률",
                 EXTRA_FEATURES[1]: "영업이익률의 전년 대비 변화"}
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def financial_records(documents, coverage, narrative_records, comparisons):
    """One record per metric/scope/year/receipt, including explicit missing states."""
    linked = {(r["code"], r["receipt_no"], r["scope"], r["period_end"], r["metric"]): r["id"]
              for r in narrative_records}
    conflicts = {(r["code"], r["receipt_no"], r["context_resolution"]["scope"],
                  str(r["context_resolution"]["year"])+"-12-31", r["financial_facts"][0]["metric"])
                 for r in comparisons if r["status"] == "value_mismatch" and r["financial_facts"]}
    result = []
    for doc in coverage:
        key = doc["code"], doc["receipt_no"]
        financial = documents[key]
        year = parse_date(doc["period_end"]).year
        for current_year in (year, year-1):
            start, end = f"{current_year}-01-01", f"{current_year}-12-31"
            for scope in SCOPES:
                for metric in METRICS:
                    facts = [f for f in financial["facts"] if (f["metric"], f["scope"], f["period_start"], f["period_end"], f["currency"])
                             == (metric, scope, start, end, "KRW")] if financial["status"] == "read" else []
                    values = {number(f["value"]) for f in facts}
                    quanta = {f["quantum"] for f in facts}
                    group = (*key, scope, end, metric)
                    state = ("source_unavailable" if financial["status"] != "read" else
                             "fact_missing" if not facts else "fact_conflict" if len(values) != 1 else
                             "precision_unresolved" if None in quanta or len(quanta) != 1 else
                             "statement_value_mismatch" if group in conflicts else "available")
                    result.append({"code": key[0], "company_name": doc["company_name"], "receipt_no": key[1],
                        "receipt_date": doc["receipt_date"], "available_from": doc["available_from"],
                        "report_period_end": doc["period_end"], "period_start": start, "period_end": end,
                        "scope": scope, "metric": metric, "currency": "KRW", "status": state,
                        "value_krw": format(next(iter(values)), "f") if state == "available" else None,
                        "quantum": next(iter(quanta)) if state == "available" else None,
                        "facts": facts, "source_sha256": financial.get("source_sha256"),
                        "source_kind": "financial_original", "narrative_record_id": linked.get(group),
                        "narrative_verified": False, "production_input_approved": False})
    return result


def margin(revenue, operating):
    if not revenue or not operating or any(r["status"] != "available" for r in (revenue, operating)):
        return None
    with localcontext() as ctx:
        ctx.prec = 60
        denominator = number(revenue["value_krw"])
        # Avoid an undefined or sign-uncertain denominator at source precision.
        if denominator - number(revenue["quantum"])/2 <= 0:
            return None
        value = float(number(operating["value_krw"]) / denominator * 100)
        return value if math.isfinite(value) else None


def attach_financials(features, records, coverage, completeness):
    attached, excluded = [], Counter()
    by_key = {(r["code"], r["receipt_no"], r["scope"], r["period_end"], r["metric"]): r for r in records}
    require(len(by_key) == len(records), "재무 근거 항목이 중복됐습니다.")
    cache = {}
    for feature in features:
        code, day = feature["code"], feature["as_of_date"]
        key = code, day
        if key not in cache:
            cache[key] = snapshot_as_of([], coverage, {code}, day, completeness=completeness)
        selected = cache[key][0]
        receipt = selected["selected_receipt_no"]
        scope = feature.get("financial_scope")
        if receipt is None:
            excluded[selected["state"]] += 1; continue
        if feature.get("financial_available") is not True or feature.get("quality_issue"):
            excluded["original_financial_or_quality_unavailable"] += 1; continue
        if scope not in SCOPES:
            excluded["original_scope_not_explicit"] += 1; continue
        if (receipt != feature.get("financial_receipt_no") or selected["period_end"] != feature.get("financial_period_end")
                or selected["available_from"] != feature.get("financial_available_from")):
            excluded["original_and_selected_receipt_differ"] += 1; continue
        end = selected["period_end"]
        if not end.endswith("-12-31"):
            excluded["non_calendar_annual_period"] += 1; continue
        previous = f"{parse_date(end).year-1}-12-31"
        def get(period, metric): return by_key.get((code, receipt, scope, period, metric))
        current_margin = margin(get(end, "revenue"), get(end, "operating_income"))
        previous_margin = margin(get(previous, "revenue"), get(previous, "operating_income"))
        attached.append({**feature, "source_financial_document_id": code+"_"+receipt,
            "source_financial_scope": scope, "source_financial_available_from": selected["available_from"],
            "source_financial_sha256": get(end, "revenue")["source_sha256"],
            EXTRA_FEATURES[0]: current_margin,
            EXTRA_FEATURES[1]: current_margin-previous_margin if current_margin is not None and previous_margin is not None else None,
            "source_current_period_end": end, "source_comparative_period_end": previous,
            "source_margin_missing": current_margin is None, "source_margin_change_missing": current_margin is None or previous_margin is None})
    return attached, dict(excluded)


def matrix(rows, columns):
    return np.array([[_number(f.get(k)) for k in columns] for f, _ in rows], dtype=float)


def fit_model(rows, alpha, columns):
    x = matrix(rows, columns)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in x.T])
    filled = np.where(np.isfinite(x), x, median)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (filled-mean)/scale
    y = np.array([o["price_return_pct"] for _, o in rows], dtype=float)
    intercept = float(y.mean())
    coefficients = np.linalg.solve(z.T@z + alpha*np.eye(len(columns)), z.T@(y-intercept))
    return {"features": list(columns), "alpha": alpha, "median": median.tolist(), "mean": mean.tolist(),
        "scale": scale.tolist(), "coefficients": coefficients.tolist(), "intercept": intercept,
        "input_observed_counts": {k: int(np.isfinite(x[:, i]).sum()) for i, k in enumerate(columns)},
        "training_rows": len(rows), "training_label_end": max(o["exit_date"] for _, o in rows)}


def predict_model(model, rows):
    x = matrix(rows, model["features"])
    predicted = (np.where(np.isfinite(x), x, model["median"])-model["mean"])/model["scale"] @ model["coefficients"] + model["intercept"]
    require(np.isfinite(predicted).all(), "재무 추가 모델의 예측 수치를 확인해야 합니다.")
    return predicted


def score_model(model, rows):
    y = np.array([o["price_return_pct"] for _, o in rows], dtype=float)
    p = predict_model(model, rows)
    return {"rows": len(rows), "unique_dates": len({f["as_of_date"] for f, _ in rows}),
        "mae_pp": float(np.abs(y-p).mean()), "mse_pp_squared": float(((y-p)**2).mean()),
        "constant_mae_pp": float(np.abs(y-model["intercept"]).mean()),
        "direction_accuracy": float(((p>0) == (y>0)).mean()), "actual_up_fraction": float((y>0).mean())}


def train_comparison(features, outcomes, split, horizons, *, minimums=(100, 30, 30), minimum_documents=6):
    require(set(horizons) <= {"5", "20"} and len(horizons) == len(set(horizons)))
    require(len(minimums) == 3 and all(type(n) is int and n >= 2 for n in minimums) and minimum_documents >= 1)
    require(len(features) == len({(f["code"], f["as_of_date"]) for f in features}))
    for f in features:
        require(f["source_financial_document_id"] == f["code"]+"_"+f["financial_receipt_no"]
            and f["source_financial_scope"] == f["financial_scope"]
            and f["source_financial_available_from"] == f["financial_available_from"],
            "추가 재무 입력과 기존 사례의 접수번호·범위·공개 시점이 다릅니다.")
    indexed = index_outcomes(outcomes)
    result = {"status": "blocked", "policy": POLICY, "models": {}, "split": split,
        "research_only": True, "fresh_unseen_evidence": False, "source_test_already_consumed": True,
        "automatic_promotion": False, "live_trading_allowed": False, "statistical_significance_established": False}
    for horizon in horizons:
        parts, excluded = partition_examples(features, indexed, int(horizon), split["validation_start"], split["test_start"])
        counts = {k: len(v) for k, v in parts.items()}
        train_ids = {f["source_financial_document_id"] for f, _ in parts["train"]}
        reason = None
        if any(counts[k] < n for k, n in zip(parts, minimums)) or len(train_ids) < minimum_documents:
            reason = "기간별 사례 또는 학습 구간의 서로 다른 재무 원문 수가 부족합니다."
        elif not np.isfinite(matrix(parts["train"], EXTRA_FEATURES)).any():
            reason = "학습 구간에서 추가 재무 입력을 확인하지 못했습니다."
        if reason:
            result["models"][horizon] = {"status": "blocked", "reason": reason, "counts": counts,
                "training_documents": len(train_ids), "excluded": excluded}; continue
        compared = {}
        for kind, columns in (("numeric", MODEL_FEATURES), ("numeric_and_finance", MODEL_FEATURES+EXTRA_FEATURES)):
            choices = []
            for alpha in ALPHAS:
                candidate = fit_model(parts["train"], alpha, columns)
                choices.append((score_model(candidate, parts["validation"])["mse_pp_squared"], alpha))
            _, alpha = min(choices)
            model = fit_model(parts["train"]+parts["validation"], alpha, columns)
            require(model["training_label_end"] < split["test_start"], "이후 평가 결과가 학습에 섞였습니다.")
            model["test_metrics"] = score_model(model, parts["test"])
            compared[kind] = model
        fit_ids = {f["source_financial_document_id"] for f, _ in parts["train"]+parts["validation"]}
        test_ids = {f["source_financial_document_id"] for f, _ in parts["test"]}
        base = compared["numeric"]["test_metrics"]["mae_pp"]
        extra = compared["numeric_and_finance"]["test_metrics"]["mae_pp"]
        constant = compared["numeric"]["test_metrics"]["constant_mae_pp"]
        result["models"][horizon] = {"status": "trained", "counts": counts, "excluded": excluded, **compared,
            "training_documents": len(fit_ids), "test_documents": len(test_ids), "unseen_test_documents": len(test_ids-fit_ids),
            "rows_using_seen_documents": sum(f["source_financial_document_id"] in fit_ids for f, _ in parts["test"]),
            "test_observed_counts": {k: int(np.isfinite(matrix(parts["test"], (k,))).sum()) for k in EXTRA_FEATURES},
            "relative_error_reduction_pct": (base-extra)/base*100 if base else None,
            "disposition": "rejected" if extra >= base or extra >= constant else "hold_for_new_validation"}
    states = [m["status"] for m in result["models"].values()]
    result["status"] = "trained" if states and all(s == "trained" for s in states) else "partial" if "trained" in states else "blocked"
    if not horizons: result["reason"] = "기존 진단에서 보류 중인 5·20일봉 모델이 없습니다."
    return result
