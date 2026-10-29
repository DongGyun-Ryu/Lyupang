"""Paired numeric/text research models; never changes the forward predictor.

Text features are statistical character fragments, not validated semantic facts.
Vocabulary, IDF, imputation, scaling and tuning use fitting periods only.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, time, timedelta

import numpy as np

from business_text import evidence_as_of
from learning import MODEL_FEATURES, _number, index_outcomes, partition_examples
from lyupang import CollectionError, SEOUL, parse_date

TEXT_POLICY = "unique_receipt_hangul_latin_2_3gram_tfidf_v1"
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def fragments(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    found = Counter()
    for word in re.findall(r"[가-힣a-z]{2,}", normalized):
        if len(word) > 40:
            continue
        for width in (2, 3):
            for start in range(len(word) - width + 1):
                found[word[start:start+width]] += 1
    return found


def attach_text(features, histories, documents):
    attached, excluded = [], Counter()
    for original in features:
        as_of = parse_date(original["as_of_date"])
        selected = evidence_as_of(histories.get(original["code"], []), as_of)
        if selected["status"] != "available":
            excluded[selected["reason"]] += 1
            continue
        event = selected["event"]
        key = original["code"] + "_" + event["receipt_no"]
        if key not in documents:
            excluded["document_text_unavailable"] += 1
            continue
        available = datetime.fromisoformat(event["available_from"])
        expected = datetime.combine(parse_date(event["receipt_date"]), time.min, SEOUL)
        if (available.tzinfo is None or available != expected + timedelta(days=1)
                or available > datetime.combine(as_of, time.min, SEOUL)):
            raise CollectionError("회사 설명이 공개되기 전의 관찰 사례에 연결됐습니다.")
        attached.append({**original, "text_document_id": key, "text_receipt_date": event["receipt_date"],
                         "text_available_from": event["available_from"], "text_sha256": documents[key]["sha256"]})
    return attached, dict(excluded)


def fit_vocabulary(document_ids, documents, *, max_terms=128):
    ids = sorted(set(document_ids))
    frequencies = Counter()
    for key in ids:
        frequencies.update(documents[key]["terms"].keys())
    # Repeated daily cases never count as separate documents in IDF.
    choices = [term for term, count in frequencies.items() if count >= 2 and count < len(ids) * 0.95]
    vocabulary = sorted(choices, key=lambda term: (-frequencies[term], term))[:max_terms]
    return {"policy": TEXT_POLICY, "vocabulary": vocabulary,
            "idf": [math.log((len(ids)+1)/(frequencies[term]+1))+1 for term in vocabulary],
            "training_document_ids": ids}


def text_vector(key, vocabulary, documents):
    terms = documents[key]["terms"]
    value = np.array([math.log1p(terms.get(term, 0)) * weight
                      for term, weight in zip(vocabulary["vocabulary"], vocabulary["idf"])], dtype=float)
    length = np.linalg.norm(value)
    return value / length if length else value


def matrix(rows, vocabulary, documents):
    numeric = np.array([[_number(feature.get(key)) for key in MODEL_FEATURES] for feature, _ in rows], dtype=float)
    if vocabulary is None:
        return numeric
    by_document = {key: text_vector(key, vocabulary, documents) for key in {f["text_document_id"] for f, _ in rows}}
    text = np.array([by_document[feature["text_document_id"]] for feature, _ in rows])
    return np.concatenate((numeric, text), axis=1)


def fit_candidate(rows, alpha, vocabulary, documents):
    x = matrix(rows, vocabulary, documents)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in x.T])
    filled = np.where(np.isfinite(x), x, median)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (filled - mean) / scale
    y = np.array([outcome["price_return_pct"] for _, outcome in rows])
    intercept = float(y.mean())
    coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (y - intercept))
    return {"numeric_features": list(MODEL_FEATURES), "text_vocabulary": vocabulary, "alpha": alpha,
            "median": median.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": coefficients.tolist(), "intercept": intercept,
            "training_rows": len(rows), "training_label_end": max(outcome["exit_date"] for _, outcome in rows)}


def predict_candidate(model, rows, documents):
    x = matrix(rows, model["text_vocabulary"], documents)
    filled = np.where(np.isfinite(x), x, model["median"])
    result = (filled - model["mean"]) / model["scale"] @ model["coefficients"] + model["intercept"]
    if not np.isfinite(result).all():
        raise CollectionError("회사 설명 모델의 예측에 유효하지 않은 값이 있습니다.")
    return result


def score_candidate(model, rows, documents):
    predicted = predict_candidate(model, rows, documents)
    y = np.array([outcome["price_return_pct"] for _, outcome in rows])
    return {"rows": len(rows), "unique_dates": len({feature["as_of_date"] for feature, _ in rows}),
            "mae_pp": float(np.abs(y-predicted).mean()), "mse_pp_squared": float(((y-predicted)**2).mean()),
            "constant_baseline_mae_pp": float(np.abs(y-model["intercept"]).mean()),
            "direction_accuracy": float(((predicted > 0) == (y > 0)).mean()),
            "actual_up_fraction": float((y > 0).mean())}


def train_comparison(features, outcomes, documents, split, horizons, *, minimums=(100, 30, 30), minimum_documents=6):
    for feature in features:
        key = feature["text_document_id"]
        available = datetime.fromisoformat(feature["text_available_from"])
        expected = datetime.combine(parse_date(feature["text_receipt_date"]), time.min, SEOUL) + timedelta(days=1)
        if (key not in documents or feature["text_receipt_date"] != documents[key]["receipt_date"]
                or feature["text_sha256"] != documents[key]["sha256"] or available.tzinfo is None or available != expected
                or feature["text_receipt_date"] >= feature["as_of_date"]
                or available > datetime.combine(parse_date(feature["as_of_date"]), time.min, SEOUL)):
            raise CollectionError("학습할 회사 설명의 원문 또는 공개 시점이 올바르지 않습니다.")
    indexed = index_outcomes(outcomes)
    result = {"status": "blocked", "models": {}, "split": split, "text_policy": TEXT_POLICY,
              "research_only": True, "fresh_unseen_evidence": False, "automatic_promotion": False,
              "live_trading_allowed": False, "source_test_already_consumed": True,
              "semantic_accuracy_established": False, "causal_claim_verified": False,
              "point_in_time_verified": False, "statistical_significance_established": False,
              "costs_included": False, "evaluation_scope": "historical_development_same_symbols"}
    if not horizons:
        result["reason"] = "기존 진단에서 보류 중인 5·20일봉 모델이 없습니다. 탈락 모델을 자동으로 되살리지 않습니다."
    for horizon in horizons:
        parts, excluded = partition_examples(features, indexed, int(horizon), split["validation_start"], split["test_start"])
        counts = {kind: len(rows) for kind, rows in parts.items()}
        train_ids = {feature["text_document_id"] for feature, _ in parts["train"]}
        if any(counts[kind] < minimum for kind, minimum in zip(parts, minimums)) or len(train_ids) < minimum_documents:
            result["models"][str(horizon)] = {"status": "blocked", "reason": "기간별 사례 또는 학습 원문 수가 부족합니다.",
                "counts": counts, "training_documents": len(train_ids), "minimum_training_documents": minimum_documents, "excluded": excluded}
            continue
        vocabulary = fit_vocabulary(train_ids, documents)
        if not vocabulary["vocabulary"]:
            result["models"][str(horizon)] = {"status": "blocked", "reason": "학습 원문에서 비교 가능한 글자 조각을 찾지 못했습니다.", "counts": counts}
            continue
        compared = {}
        for name in ("numeric", "numeric_and_text"):
            initial_vocabulary = vocabulary if name == "numeric_and_text" else None
            choices = []
            for alpha in ALPHAS:
                model = fit_candidate(parts["train"], alpha, initial_vocabulary, documents)
                metrics = score_candidate(model, parts["validation"], documents)
                choices.append((metrics["mse_pp_squared"], alpha))
            _, alpha = min(choices)
            fitting = parts["train"] + parts["validation"]
            final_vocabulary = fit_vocabulary((f["text_document_id"] for f, _ in fitting), documents) if name == "numeric_and_text" else None
            model = fit_candidate(fitting, alpha, final_vocabulary, documents)
            if model["training_label_end"] >= split["test_start"]:
                raise CollectionError("설명 모델 학습에 이후 비교 기간의 결과가 들어갔습니다.")
            model["test_metrics"] = score_candidate(model, parts["test"], documents)
            compared[name] = model
        fit_ids = set(compared["numeric_and_text"]["text_vocabulary"]["training_document_ids"])
        test_ids = {feature["text_document_id"] for feature, _ in parts["test"]}
        base, text = (compared[name]["test_metrics"]["mae_pp"] for name in ("numeric", "numeric_and_text"))
        constant = compared["numeric"]["test_metrics"]["constant_baseline_mae_pp"]
        result["models"][str(horizon)] = {"status": "trained", "counts": counts, "excluded": excluded, **compared,
            "training_documents": len(fit_ids), "test_documents": len(test_ids), "unseen_test_documents": len(test_ids-fit_ids),
            "rows_using_seen_documents": sum(feature["text_document_id"] in fit_ids for feature, _ in parts["test"]),
            "relative_error_reduction_pct": (base-text)/base*100 if base else None,
            "disposition": "rejected" if text >= base or text >= constant else "hold_for_new_validation"}
    statuses = [model["status"] for model in result["models"].values()]
    result["status"] = "trained" if statuses and all(s == "trained" for s in statuses) else "partial" if "trained" in statuses else "blocked"
    return result
