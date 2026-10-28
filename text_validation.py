"""Frozen, price-blind review samples and agreement with a reviewer's answers.

Local hashes link artifacts; they do not authenticate a human or a clock.
Neither sampling nor evaluation fits or changes any model.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from lyupang import CollectionError, SEOUL

SUBJECTS = ("company", "industry", "mixed", "unclear")
TIMES = ("reported", "outlook", "mixed", "unclear")
SUBJECT_NAMES = dict(zip(SUBJECTS, ("회사", "산업 전체", "둘 다", "주체 불명확")))
TIME_NAMES = dict(zip(TIMES, ("과거·현재의 상태나 활동", "미래 전망·계획", "둘 다", "시점 불명확")))
GROUP_NAMES = {"general": "일반 추출", "challenge": "오류 점검 추출"}
RUBRIC = "text_subject_time_v1"
SUPPORTED_POLICIES = {"sentence_context_rules_v2", "sentence_context_rules_v3"}
MAX_BYTES = 64 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path, limit=MAX_BYTES):
    raw = path.read_bytes()
    if len(raw) > limit:
        raise CollectionError("검토 자료의 파일 크기가 지원 범위를 넘었습니다.")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise CollectionError("검토 자료에 중복된 항목 이름이 있습니다.")
            result[key] = value
        return result
    def invalid(_):
        raise CollectionError("검토 자료에 유효하지 않은 수치가 있습니다.")
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)


def latest_context(root):
    candidates = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and not path.is_symlink() and not path.name.startswith("."):
                try:
                    stamp = datetime.fromisoformat(read_json(path / "summary.json")["created_at"])
                    if stamp.tzinfo is not None:
                        candidates.append((stamp, path))
                except (OSError, ValueError, KeyError, TypeError, CollectionError):
                    continue
    if not candidates:
        raise CollectionError("먼저 CONTEXT.cmd의 문장 분류를 완료해주세요.")
    return max(candidates, key=lambda item: item[0])[1]


def source_rows(source):
    summary = read_json(source / "summary.json")
    if (summary.get("schema_version") != 1 or summary.get("status") != "linked"
            or summary.get("classification_policy") not in SUPPORTED_POLICIES):
        raise CollectionError("0.7.1 이상의 CONTEXT.cmd에서 처리가 완료된 결과가 필요합니다.")
    raw = (source / "statements.jsonl").read_bytes()
    if len(raw) > MAX_BYTES or digest(raw) != summary.get("statement_sha256"):
        raise CollectionError("문장 분류 파일의 크기 또는 해시가 저장 기록과 다릅니다.")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(rows) != summary["counts"]["sentences"] or not 1 <= len(rows) <= 100_000:
        raise CollectionError("검토 대상 문장 수가 저장 기록과 다릅니다.")
    docs, blocks, companies = {}, {}, {}
    for stock in summary["stocks"]:
        code = stock["code"]
        if not re.fullmatch(r"[0-9]{6}", code) or code in companies:
            raise CollectionError("검토 대상 종목 기록이 올바르지 않습니다.")
        companies[code] = stock["company_name"]
        for doc in stock["documents"]:
            receipt = doc["receipt_no"]
            release = f"{code}_{receipt}"
            if not re.fullmatch(r"[0-9]{14}", receipt) or release in docs or doc["release_id"] != release:
                raise CollectionError("검토 대상 접수번호 기록이 올바르지 않습니다.")
            path = source / "sources" / (release + ".json")
            payload = path.read_bytes()
            if len(payload) > MAX_BYTES or digest(payload) != doc["source_blocks_sha256"]:
                raise CollectionError("검토 문장의 원문 문단 해시가 일치하지 않습니다.")
            documents = json.loads(payload)
            if not isinstance(documents, list) or len(documents) > 100_000:
                raise CollectionError("검토 원문의 문단 형식이 올바르지 않습니다.")
            mapped = {}
            for block in documents:
                if (not re.fullmatch(r"b[0-9]{6}", block["id"]) or block["id"] in mapped
                        or not isinstance(block["text"], str) or digest(block["text"].encode()) != block["text_sha256"]):
                    raise CollectionError("검토 원문의 문단 식별번호 또는 해시가 다릅니다.")
                mapped[block["id"]] = block
            docs[release], blocks[release] = doc, mapped
    seen = set()
    for row in rows:
        release = f"{row['code']}_{row['receipt_no']}"
        if (row["release_id"] != release or release not in docs or row["id"] in seen
                or not re.fullmatch(re.escape(release + "_" + row["block_id"] + "_") + r"[1-9][0-9]*", row["id"])):
            raise CollectionError("검토 문장의 출처 또는 식별번호가 올바르지 않습니다.")
        block = blocks[release][row["block_id"]]
        start, end = row["start"], row["end"]
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(block["text"])
                or block["kind"] != "paragraph" or block["text"][start:end] != row["text"]
                or block["text_sha256"] != row["block_sha256"] or block["headings"] != row["headings"]
                or digest(row["text"].encode()) != row["text_sha256"]):
            raise CollectionError("검토 문장이 원문의 정확한 문자 범위와 일치하지 않습니다.")
        cls = row["classification"]
        if (cls["method"] != summary["classification_policy"] or cls["subject"] not in SUBJECTS
                or cls["time_orientation"] not in TIMES or cls["status"] not in {"needs_review", "rule_based_candidate"}):
            raise CollectionError("검토 대상 분류의 정책 또는 답 형식이 올바르지 않습니다.")
        seen.add(row["id"])
    return summary, rows, blocks, companies


def sample_rows(rows, seed):
    """12 hash-ranked unique texts + up to 12 explicitly selected diagnostic texts.

    Exact whitespace-normalized repeats share a group. Similar paraphrases and
    same-company/year correlations are not claimed independent.
    """
    unique = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        key = digest(re.sub(r"\s+", "", row["text"]).encode())
        unique.setdefault(key, row)
    ranked = sorted(unique, key=lambda key: digest((seed + ":general:" + key).encode()))
    selected = [(key, "general") for key in ranked[:12]]
    used = {key for key, _ in selected}
    checks = (
        lambda row: bool(row["classification"]["excluded_expressions"]),
        lambda row: row["classification"]["status"] == "rule_based_candidate",
        lambda row: row["classification"]["status"] == "needs_review",
    )
    for number, predicate in enumerate(checks):
        candidates = [key for key in ranked if key not in used and predicate(unique[key])]
        candidates.sort(key=lambda key: digest((seed + f":challenge{number}:" + key).encode()))
        for key in candidates[:4]:
            selected.append((key, "challenge"))
            used.add(key)
    # Small corpora or sparse diagnostic strata: use remaining unique texts.
    for key in ranked:
        if len(selected) >= min(24, len(unique)):
            break
        if key not in used:
            selected.append((key, "challenge"))
            used.add(key)
    # Hide sampling groups in the annotation screen; keep their audit mapping.
    selected.sort(key=lambda pair: digest((seed + ":display:" + pair[0]).encode()))
    return [(unique[key], group, key) for key, group in selected], len(unique)


def make_pack(source, target):
    summary, rows, blocks, companies = source_rows(source)
    seed = summary["statement_sha256"] + ":" + RUBRIC
    selected, population = sample_rows(rows, seed)
    items, predictions = [], []
    for row, group, key in selected:
        block = blocks[row["release_id"]][row["block_id"]]
        # Bound the surrounding text for the review screen, keeping real offsets.
        left, right = max(0, row["start"] - 700), min(len(block["text"]), row["end"] + 700)
        items.append({k: row[k] for k in ("id", "code", "receipt_no", "text", "text_sha256", "block_id", "block_sha256", "start", "end", "headings")})
        items[-1].update(company_name=companies[row["code"]], context=block["text"][left:right],
                         context_start=left, context_end=right)
        cls = row["classification"]
        predictions.append({"id": row["id"], "text_sha256": row["text_sha256"], "dedup_group": key,
                            "sample_group": group, "subject": cls["subject"], "time_orientation": cls["time_orientation"],
                            "routing": cls["status"]})
    item_bytes, prediction_bytes = canonical(items), canonical(predictions)
    pack_id = digest(seed.encode())[:24]
    manifest = {"schema_version": 1, "pack_id": pack_id, "created_at": datetime.now(SEOUL).isoformat(),
        "rubric": RUBRIC, "classification_policy": summary["classification_policy"], "sample_size": len(items),
        "source_context": str(source.resolve()), "source_summary_sha256": digest((source / "summary.json").read_bytes()),
        "source_statements_sha256": summary["statement_sha256"], "population_sentences": len(rows),
        "population_unique_texts": population, "items_sha256": digest(item_bytes), "predictions_sha256": digest(prediction_bytes),
        "sampling": "12_hash_ranked_unique_texts_plus_12_diagnostic_strata_v1", "selection_seed": seed,
        "fresh_unseen_evidence": False, "reviewer_identity_verified": False, "training_executed": False,
        "live_trading_allowed": False}
    manifest["pack_hash"] = digest(canonical(manifest))
    target.mkdir(parents=True, exist_ok=False)
    (target / "items.json").write_bytes(item_bytes)
    (target / "predictions.json").write_bytes(prediction_bytes)
    (target / "manifest.json").write_bytes(canonical(manifest))
    (target / "review.html").write_text(review_html(manifest, items), encoding="utf-8")
    return manifest


def review_html(manifest, items, saved_answers=None):
    template = Path(__file__).with_name("text_review.html").read_text(encoding="utf-8")
    data = {"pack_id": manifest["pack_id"], "pack_hash": manifest["pack_hash"], "rubric": RUBRIC, "items": items}
    if saved_answers is not None:
        # Only previously validated user-entered fields belong in the editor;
        # never carry arbitrary extra JSON fields (e.g. program guesses) over.
        seed = {key: saved_answers[key] for key in ("schema_version", "pack_id", "pack_hash", "rubric", "annotation_source", "reviewer_attestation")}
        seed["answers"] = [{**{key: row[key] for key in ("id", "text_sha256", "state", "subject", "time_orientation")},
                            "note": row.get("note", "")} for row in saved_answers["answers"]]
        data["saved_answers"] = seed
        data["seed_hash"] = digest(canonical(seed))
    embedded = canonical(data).decode().replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return template.replace("__REVIEW_DATA__", embedded)


def load_pack(path):
    manifest = read_json(path / "manifest.json", 256_000)
    if (manifest.get("schema_version") != 1 or manifest.get("rubric") != RUBRIC
            or manifest.get("classification_policy") not in SUPPORTED_POLICIES
            or not re.fullmatch(r"[0-9a-f]{24}", manifest.get("pack_id", ""))):
        raise CollectionError("지원하는 문장 검토 묶음이 아닙니다.")
    unsigned = {k: v for k, v in manifest.items() if k != "pack_hash"}
    if digest(canonical(unsigned)) != manifest.get("pack_hash"):
        raise CollectionError("검토 묶음의 출처 기록 해시가 다릅니다.")
    values = []
    for name, key in (("items.json", "items_sha256"), ("predictions.json", "predictions_sha256")):
        raw = (path / name).read_bytes()
        if len(raw) > MAX_BYTES or digest(raw) != manifest[key]:
            raise CollectionError("검토 문장 또는 고정 분류의 해시가 다릅니다.")
        values.append(read_json(path / name))
    items, predictions = values
    if not 1 <= len(items) == len(predictions) == manifest["sample_size"] <= 24:
        raise CollectionError("검토 문장 수가 올바르지 않습니다.")
    seen = set()
    for item, prediction in zip(items, predictions):
        if (item["id"] in seen or item["id"] != prediction["id"] or digest(item["text"].encode()) != item["text_sha256"]
                or item["text_sha256"] != prediction["text_sha256"] or prediction["subject"] not in SUBJECTS
                or prediction["time_orientation"] not in TIMES or prediction["sample_group"] not in GROUP_NAMES
                or prediction["routing"] not in {"needs_review", "rule_based_candidate"}):
            raise CollectionError("검토 문장과 고정 분류의 식별번호 또는 답 형식이 다릅니다.")
        seen.add(item["id"])
    return manifest, items, predictions


def evaluate(pack, label_path):
    manifest, items, predictions = load_pack(pack)
    answers = read_json(label_path, 2_000_000)
    if (answers.get("schema_version") != 1 or answers.get("pack_id") != manifest["pack_id"]
            or answers.get("pack_hash") != manifest["pack_hash"] or answers.get("rubric") != RUBRIC
            or answers.get("annotation_source") != "human_entered" or answers.get("reviewer_attestation") is not True):
        raise CollectionError("이 검토 묶음에서 직접 답하고 저장한 답안 파일이 필요합니다.")
    records = answers.get("answers")
    if not isinstance(records, list) or not 1 <= len(records) <= len(items):
        raise CollectionError("검토 답안 수가 올바르지 않습니다.")
    expected = {item["id"]: item for item in items}
    labels = {}
    for row in records:
        if (not isinstance(row, dict) or row.get("id") not in expected or row["id"] in labels
                or row.get("text_sha256") != expected[row["id"]]["text_sha256"]
                or not isinstance(row.get("note", ""), str) or len(row.get("note", "")) > 500):
            raise CollectionError("검토 답안의 문장·해시·중복·메모를 확인해주세요.")
        if row.get("state") == "answered":
            if row.get("subject") not in SUBJECTS or row.get("time_orientation") not in TIMES:
                raise CollectionError("검토 답안의 선택 항목이 올바르지 않습니다.")
        elif row.get("state") == "abstain":
            if row.get("subject") is not None or row.get("time_orientation") is not None:
                raise CollectionError("판단을 유보한 문장에는 정답을 채우지 않습니다.")
        else:
            raise CollectionError("답하지 않은 문장을 검토 완료로 처리할 수 없습니다.")
        labels[row["id"]] = row
    comparisons = []
    for prediction in predictions:
        label = labels.get(prediction["id"])
        comparisons.append({**prediction, "review": label, "state": label["state"] if label else "unanswered"})
    groups = []
    for group in GROUP_NAMES:
        selected = [row for row in comparisons if row["sample_group"] == group]
        scored = [row for row in selected if row["state"] == "answered"]
        metrics = {"group": group, "sampled": len(selected), "answered": len(scored),
                   "abstained": sum(row["state"] == "abstain" for row in selected),
                   "unanswered": sum(row["state"] == "unanswered" for row in selected)}
        for axis in ("subject", "time_orientation"):
            metrics[axis + "_agreement_pct"] = (100 * sum(row[axis] == row["review"][axis] for row in scored) / len(scored)) if scored else None
            metrics[axis + "_confusion"] = {truth: {guess: sum(row["review"][axis] == truth and row[axis] == guess for row in scored)
                                                   for guess in (SUBJECTS if axis == "subject" else TIMES)}
                                            for truth in (SUBJECTS if axis == "subject" else TIMES)}
        metrics["joint_agreement_pct"] = (100 * sum(all(row[axis] == row["review"][axis] for axis in ("subject", "time_orientation")) for row in scored) / len(scored)) if scored else None
        metrics["candidate_answered"] = sum(row["routing"] == "rule_based_candidate" for row in scored)
        metrics["candidate_joint_disagreements"] = sum(row["routing"] == "rule_based_candidate" and any(row[axis] != row["review"][axis] for axis in ("subject", "time_orientation")) for row in scored)
        groups.append(metrics)
    summary = {"schema_version": 1, "created_at": datetime.now(SEOUL).isoformat(), "pack_id": manifest["pack_id"],
        "pack_hash": manifest["pack_hash"], "label_sha256": digest(label_path.read_bytes()), "groups": groups,
        "sampled": len(items), "answered": sum(row["state"] == "answered" for row in comparisons),
        "abstained": sum(row["state"] == "abstain" for row in comparisons),
        "unanswered": sum(row["state"] == "unanswered" for row in comparisons),
        "status": "reviewer_comparison_only", "human_review_attested": True, "reviewer_identity_verified": False,
        "ground_truth_independently_verified": False, "fresh_unseen_evidence": False,
        "population_accuracy_established": False, "model_approved": False, "training_executed": False,
        "live_trading_allowed": False}
    return summary, comparisons


def evaluation_html(summary, comparisons, items):
    esc = lambda value: html.escape(str(value))
    pct = lambda value: "—" if value is None else f"{value:.1f}%"
    rows = []
    for group in summary["groups"]:
        rows.append(f"<tr><td>{GROUP_NAMES[group['group']]}</td><td>{group['sampled']}</td><td>{group['answered']}</td>"
                    f"<td>{group['abstained']}</td><td>{group['unanswered']}</td><td>{pct(group['subject_agreement_pct'])}</td>"
                    f"<td>{pct(group['time_orientation_agreement_pct'])}</td><td>{pct(group['joint_agreement_pct'])}</td></tr>")
    lookup = {item["id"]: item for item in items}
    details = []
    for row in comparisons:
        if row["state"] != "answered":
            continue
        item, label = lookup[row["id"]], row["review"]
        same = all(row[axis] == label[axis] for axis in ("subject", "time_orientation"))
        preview = row.get("development_recheck")
        preview_line = (f"<p>보완 규칙으로 다시 읽은 답 (개발용): {SUBJECT_NAMES[preview['subject']]} / {TIME_NAMES[preview['time_orientation']]}</p>"
                        if preview else "")
        details.append(f"<details open><summary>{'일치' if same else '불일치'} · {esc(item['company_name'])}</summary>"
            f"<blockquote>{esc(item['text'])}</blockquote><p>검토자 답: {SUBJECT_NAMES[label['subject']]} / {TIME_NAMES[label['time_orientation']]}</p>"
            f"<p>고정한 프로그램 답: {SUBJECT_NAMES[row['subject']]} / {TIME_NAMES[row['time_orientation']]}</p>"
            f"{preview_line}<p>메모: {esc(label.get('note', ''))}</p>"
            f"<p><a href='review.html#{esc(quote(item['id'], safe=''))}'>이 문장의 답 수정하기</a></p></details>")
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>문장 검토 결과</title><style>body{{font-family:system-ui;max-width:1100px;margin:30px auto;padding:16px;line-height:1.8}}td,th{{border-bottom:1px solid #ccc;padding:10px}}table{{border-collapse:collapse}}blockquote{{white-space:pre-wrap;background:#f0f4f8;padding:15px}}details{{padding:15px;border:1px solid #ddd;margin:12px 0}}</style>
<h1>문장 검토 결과</h1><p>전체 {summary['sampled']}문장 · 답변 {summary['answered']} · 판단 유보 {summary['abstained']} · 미답변 {summary['unanswered']}</p>
<p>검토자가 직접 입력했다고 확인한 답과 프로그램의 고정 분류를 비교했습니다. 검토자 신원이나 답의 옳음을 외부에서 확인한 결과는 아닙니다. 유보·미답변은 정답으로 채우거나 일치율의 분모에 넣지 않습니다.</p>
<p><a href='review.html'>저장한 답을 이어서 확인하기</a> · 아래 각 문장의 '이 문장의 답 수정하기'를 누르면 해당 문장으로 바로 이동합니다. 저장한 기존 답은 유지되며 검토자 답을 자동으로 바꾸지 않습니다.</p>
<table><tr><th>추출 구분</th><th>문장</th><th>답변</th><th>유보</th><th>미답변</th><th>주체 일치</th><th>시점 일치</th><th>둘 다 일치</th></tr>{''.join(rows)}</table>
<p>일반 추출은 공백을 제외한 동일 문장을 하나로 모은 뒤 해시 순서로 골랐습니다. 오류 점검 추출은 특정 유형을 더 골랐습니다. 두 구분을 합쳐 전체 정확도로 주장하지 않습니다. 비슷한 문장과 같은 회사의 설명은 독립 표본이 아닐 수 있습니다. 이 소수 사례와 1인 검토만으로 전체 문장·다른 회사의 분류 정확도나 통계적 유의성을 입증하지 않습니다.</p>
<p>이미 확인한 과거 자료에서 고른 개발용 검토입니다. 새 미사용 평가가 아니며, 이 답을 보고 규칙을 고쳤다면 같은 검토 묶음을 새 검증으로 재사용할 수 없습니다. 재학습·모델 승인·주문은 실행하지 않았습니다.</p>
<p>보완 규칙으로 다시 읽은 답은 오류를 수정한 개발 결과입니다. 위 일치율은 처음 고정한 프로그램 답으로 계산하며, 보완 규칙의 답으로 바꿔 계산하지 않습니다. 보완 결과는 새로운 미사용 검증이 아닙니다.</p>
<h2>문장별 비교</h2>{''.join(details)}</html>"""
