"""Transparent text candidates and receipt-level price observations, never trading signals."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, time, timedelta

from lyupang import CollectionError, SEOUL, parse_date
from research_data import HORIZONS

POLICY = "sentence_context_rules_v3"
COMPANY = re.compile(r"당사|우리\s*회사|연결\s*회사|연결\s*대상\s*회사|연결\s*기준|별도\s*기준")
# A geographical sales destination ('미국시장에서는') does not establish
# that the sentence describes the industry rather than the reporting company.
INDUSTRY = re.compile(r"시장(?:은|이)(?![가-힣])|시장\s*(?:규모|수요|성장)|산업(?:은|이)(?![가-힣])|업계(?:는|가)(?![가-힣])")
OUTLOOK = re.compile(r"전망|예상|예측|계획|목표|예정|기대|추정|추진|추구|확대할|것으로\s*보")
TECHNICAL_TERMS = (
    ("accounting_term", re.compile(r"기대\s*신용\s*손실|회계\s*추정|추정\s*(?:내용\s*연수|내용연수|잔존\s*가치|잔존가치|공정\s*가치|공정가치)|예상\s*신용\s*손실")),
    ("management_term", re.compile(r"(?:자금\s*수지|현금\s*흐름|현금\s*수지)\s*예측|계획\s*표준\s*생산|(?:확정\s*급여|퇴직\s*급여)\s*계획")),
    ("organization_term", re.compile(r"추진(?:체계|체|위원회|단|조직|부서|본부|력)")),
)
REPORTED = re.compile(r"하였습니다|했습니다|되었습니다|됐습니다|하였으며|했으며|되었으며|기록했|기록하였|달성했|달성하였|집계되었")
ONGOING = re.compile(r"(?:창출|운영|생산|판매|제조|공급|제공|관리|보유|수행|사용|모니터링)(?:하고|해\s*오고)\s*있습니다")
ACCOUNTING_AMOUNT = re.compile(r"^(?:당기|전기)의\s*계정과목별\s*손상차손(?:\s*\(환입\))?\s*인식액(?:은|이)")
FINANCIAL_HEADINGS = re.compile(r"경영진단및분석의견|재무상태및영업실적|재무제표주석")
PERFORMANCE = re.compile(r"매출|영업이익|순이익|영업손실|판매량|수익성|영업실적|거래액|실적")
RISK = re.compile(r"위험|리스크|불확실|우려|손실\s*가능|경쟁\s*심화|환율\s*변동")
COMPANY_HEADINGS = re.compile(r"매출및수주|주요제품|제품및서비스|재무상태및영업실적|영업실적")
SENTENCE_END = re.compile(r'''(?:[다요음함][.!?。]|[!?。])["'”’)\]]*''')


def sentence_spans(text):
    """Conservative Korean endings; preserve exact offsets and decimal numbers."""
    start = 0
    ends = [match.end() for match in SENTENCE_END.finditer(text)] + [len(text)]
    for end in ends:
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            yield left, right
        start = end


def classify(text, headings, company_name):
    """No prices or future outcomes are accepted by the classifier."""
    matches = {}
    for name, pattern in (("company", COMPANY), ("industry", INDUSTRY), ("outlook", OUTLOOK),
                          ("reported", REPORTED), ("performance", PERFORMANCE), ("risk", RISK)):
        found = [{"start": m.start(), "end": m.end(), "text": m.group()} for m in pattern.finditer(text)]
        if found:
            matches[name] = found
    for match in ONGOING.finditer(text):
        matches.setdefault("reported", []).append({"start": match.start(), "end": match.end(), "text": match.group()})
    # Ignore only the cue inside a specifically identified compound term. Keep
    # independent future clauses and the exact source positions for inspection.
    excluded = []
    terms = [(category, m) for category, pattern in TECHNICAL_TERMS for m in pattern.finditer(text)]
    retained = []
    for cue in matches.get("outlook", []):
        term = next(((category, m) for category, m in terms
                     if m.start() <= cue["start"] and cue["end"] <= m.end()), None)
        if term:
            category, match = term
            excluded.append({**cue, "reason": category, "context_start": match.start(),
                             "context_end": match.end(), "context_text": match.group()})
        else:
            retained.append(cue)
    if retained:
        matches["outlook"] = retained
    else:
        matches.pop("outlook", None)
    aliases = [company_name]
    if company_name in {"NAVER", "네이버"}:
        aliases = ["NAVER", "네이버"]
    for alias in aliases:
        if len(alias) >= 2:
            for match in re.finditer(re.escape(alias), text, re.I):
                matches.setdefault("company", []).append({"start": match.start(), "end": match.end(), "text": match.group()})
    company, industry = bool(matches.get("company")), bool(matches.get("industry"))
    heading_fallback = False
    headings_text = re.sub(r"\s+", "", " / ".join(headings))
    amount_match = ACCOUNTING_AMOUNT.search(text)
    financial_context = bool(amount_match and FINANCIAL_HEADINGS.search(headings_text)
                             and re.search(r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:억원|백만원|천원|원)", text)
                             and re.search(r"(?:등)?입니다[.!?。]?\s*$", text))
    financial_fallback = bool(financial_context and not company and not industry)
    if financial_fallback:
        company = True
    if financial_context:
        matches["financial_amount"] = [{"start": amount_match.start(), "end": amount_match.end(), "text": amount_match.group()}]
    if not company and not industry and matches.get("performance"):
        heading_fallback = bool(COMPANY_HEADINGS.search(headings_text))
        company = heading_fallback
    subject = "mixed" if company and industry else "company" if company else "industry" if industry else "unclear"
    outlook = bool(matches.get("outlook"))
    reported = bool(matches.get("reported"))
    # Static numeric performance is a reported-value candidate only without a
    # forward-looking cue. A year or '전년 대비' alone never proves realization.
    numeric_present = bool(matches.get("performance") and re.search(r"[0-9]", text) and re.search(r"입니다|기록입니다", text))
    if (numeric_present or financial_context) and not outlook:
        reported = True
    orientation = "mixed" if outlook and reported else "outlook" if outlook else "reported" if reported else "unclear"
    topics = [key for key in ("performance", "risk", "financial_amount") if matches.get(key)] or ["business"]
    review_reasons = []
    if subject in {"mixed", "unclear"}:
        review_reasons.append("subject_" + subject)
    if orientation in {"mixed", "unclear"}:
        review_reasons.append("time_" + orientation)
    if excluded:
        # Removing a false future cue is not proof that the whole sentence is
        # a realized result, even when numeric amounts or past verbs appear.
        review_reasons.append("technical_term_requires_context")
    if len(text) > 2000:
        review_reasons.append("long_sentence")
    return {"subject": subject, "time_orientation": orientation, "topics": topics,
            "status": "needs_review" if review_reasons else "rule_based_candidate", "matched_expressions": matches,
            "excluded_expressions": excluded, "review_reasons": review_reasons,
            "company_heading_fallback": heading_fallback, "method": POLICY,
            "financial_heading_fallback": financial_fallback,
            "externally_verified": False, "causal_claim_verified": False}


def annotate(blocks, event, code, company_name):
    received = parse_date(event["receipt_date"])
    expected = (received + timedelta(days=1)).isoformat() + "T00:00:00+09:00"
    if event["available_from"] != expected:
        raise CollectionError("문서의 접수일과 사용 가능 시각이 일치하지 않습니다.")
    statements, skipped_tables = [], 0
    for block in blocks:
        path = re.sub(r"\s+", "", " / ".join(block["headings"]))
        if not any(word in path for word in ("사업의내용", "사업개요", "사업의개요", "경영진단", "영업실적", "경영성과")):
            continue
        if block["kind"] == "table":
            skipped_tables += 1
            continue
        if block["kind"] != "paragraph":
            continue
        for index, (start, end) in enumerate(sentence_spans(block["text"]), 1):
            text = block["text"][start:end]
            if len(text) < 8:
                continue
            statement = {"id": f"{code}_{event['receipt_no']}_{block['id']}_{index}",
                "release_id": f"{code}_{event['receipt_no']}", "code": code, "receipt_no": event["receipt_no"],
                "receipt_date": event["receipt_date"], "available_from": expected, "period_end": event["period_end"],
                "block_id": block["id"], "block_sha256": block["text_sha256"], "start": start, "end": end,
                "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "headings": block["headings"],
                "classification": classify(text, block["headings"], company_name)}
            statements.append(statement)
            if len(statements) > 50_000:
                raise CollectionError("문서당 문장 수가 지원 범위를 넘었습니다.")
    return statements, skipped_tables


def release_outcomes(event, code, metadata, bars, reference_date):
    """One row per receipt/horizon, not per sentence. No same-day receipt prices."""
    received = parse_date(event["receipt_date"])
    available_day = received + timedelta(days=1)
    if event["available_from"] != available_day.isoformat() + "T00:00:00+09:00":
        raise CollectionError("가격 연결 대상의 공개 시점이 올바르지 않습니다.")
    if (metadata.get("code") != code or metadata.get("adjusted") is not True
            or metadata.get("adjustment_base_date") != metadata.get("requested_end")):
        raise CollectionError("가격 자료의 종목 또는 수정주가 기준이 일치하지 않습니다.")
    if ([b.date for b in bars] != sorted({b.date for b in bars}) or any(b.code != code for b in bars)):
        raise CollectionError("가격 연결에는 같은 종목의 정렬된 일봉이 필요합니다.")
    coverage_start, coverage_end = parse_date(metadata["requested_start"]), parse_date(metadata["requested_end"])
    reference = parse_date(reference_date)
    eligible = [b for b in bars if available_day.isoformat() <= b.date <= min(coverage_end, reference).isoformat()]
    outcomes = []
    for horizon in HORIZONS:
        row = {"release_id": f"{code}_{event['receipt_no']}", "code": code, "receipt_no": event["receipt_no"],
               "receipt_date": event["receipt_date"], "available_from": event["available_from"],
               "period_end": event["period_end"], "horizon": horizon, "status": "pending", "reason": "later_bars_needed",
               "entry_date": None, "exit_date": None, "entry_open": None, "exit_close": None, "price_change_pct": None,
               "calendar_verified": False, "point_in_time_prices_verified": False, "strategy_return": False}
        if available_day < coverage_start or not any(b.date < available_day.isoformat() for b in bars):
            row.update(status="outside_coverage", reason="release_before_price_coverage")
        elif available_day > reference or available_day > coverage_end:
            row["reason"] = "release_not_available_within_reference"
        elif len(eligible) >= horizon:
            selected = eligible[:horizon]
            if any(b.open <= 0 or b.close <= 0 or b.zero_volume or b.volume_capped for b in selected):
                row.update(status="quality_issue", reason="price_or_volume_quality")
            else:
                first, last = selected[0], selected[-1]
                entry_time = datetime.combine(parse_date(first.date), time(9), SEOUL)
                if entry_time < datetime.fromisoformat(event["available_from"]):
                    raise CollectionError("문서가 사용 가능해지기 전의 가격에 연결됐습니다.")
                row.update(status="observed", reason="first_observed_session_after_availability",
                           entry_date=first.date, exit_date=last.date, entry_open=first.open, exit_close=last.close,
                           price_change_pct=(last.close / first.open - 1) * 100,
                           days_until_first_observed_session=(parse_date(first.date) - available_day).days)
        outcomes.append(row)
    return outcomes


def mark_overlaps(outcomes):
    for row in outcomes:
        row["same_day_receipts"] = sorted({other["receipt_no"] for other in outcomes
            if other["code"] == row["code"] and other["available_from"] == row["available_from"]})
        row["overlaps_other_receipt"] = any(other["code"] == row["code"] and other["horizon"] == row["horizon"]
            and other["receipt_no"] != row["receipt_no"] and other["status"] == row["status"] == "observed"
            and other["entry_date"] <= row["exit_date"] and row["entry_date"] <= other["exit_date"] for other in outcomes)


def counts(statements):
    classes = [s["classification"] for s in statements]
    return {"sentences": len(classes),
            "company_reported_performance": sum(c["subject"] == "company" and c["time_orientation"] == "reported"
                                                and "performance" in c["topics"] and c["status"] == "rule_based_candidate" for c in classes),
            "industry": sum(c["subject"] == "industry" for c in classes),
            "outlook_included": sum(c["time_orientation"] in {"outlook", "mixed"} for c in classes),
            "technical_term_review": sum(bool(c["excluded_expressions"]) for c in classes),
            "needs_review": sum(c["status"] == "needs_review" for c in classes)}
