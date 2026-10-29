"""Conservative, source-linked statement candidates and literal measurements.

These rules do not verify company claims, understand arbitrary language, or
establish causes of price changes. Ambiguous numbers are never guessed.
"""
from __future__ import annotations

import hashlib
import re
from datetime import timedelta
from decimal import Decimal, localcontext

from evidence_context import classify, sentence_spans
from lyupang import CollectionError, parse_date

POLICY = "source_statement_measurements_v2"
TOPIC_NAMES = {"performance": "매출·이익", "orders": "수주·계약", "investment": "투자", "cost": "비용", "risk": "위험요인"}
METRICS = {
    "매출총이익률": ("gross_margin", "performance"), "매출총이익": ("gross_profit", "performance"),
    "영업수익": ("operating_revenue", "performance"),
    "매출원가": ("cost", "cost"), "영업이익률": ("operating_margin", "performance"),
    "순이익률": ("net_margin", "performance"), "원가율": ("cost_ratio", "cost"),
    "매출비중": ("revenue_share", "performance"), "매출액": ("revenue", "performance"), "매출": ("revenue", "performance"),
    "영업이익": ("operating_profit", "performance"), "영업손실": ("operating_loss", "performance"),
    "당기순이익": ("net_profit", "performance"), "순이익": ("net_profit", "performance"),
    "순손실": ("net_loss", "performance"), "손상차손": ("impairment", "cost"),
    "판매대수": ("sales_volume", "performance"), "판매량": ("sales_volume", "performance"), "판매실적": ("sales_volume", "performance"),
    "수주잔고": ("order_backlog", "orders"), "수주금액": ("orders", "orders"), "수주액": ("orders", "orders"),
    "수주": ("orders", "orders"), "공급계약": ("contract", "orders"), "계약체결": ("contract", "orders"),
    "설비투자": ("investment", "investment"), "시설투자": ("investment", "investment"), "투자금액": ("investment", "investment"),
    "신규투자": ("investment", "investment"), "투자유치": ("investment", "investment"),
    "투자": ("investment", "investment"),
    "연구개발비": ("research_cost", "cost"), "영업비용": ("operating_cost", "cost"),
    "판매관리비": ("selling_cost", "cost"), "판매비와관리비": ("selling_cost", "cost"), "원재료비": ("material_cost", "cost"),
    "인건비": ("labor_cost", "cost"), "비용": ("cost", "cost"), "원가": ("cost", "cost"),
}
METRIC = re.compile("|".join(re.escape(word) for word in sorted(METRICS, key=len, reverse=True)))
RISK = re.compile(r"위험|리스크|불확실|환율\s*변동|원자재\s*가격|손실\s*가능")
CHANGE = re.compile(r"증가|감소|확대|축소|개선|악화|성장|체결|수주|투자|지출|부담|상승|하락")
RELATED = re.compile(r"자회사|종속\s*(?:회사|기업)|관계회사|계열사|계열회사|[가-힣A-Za-z]{1,16}\s*(?:부문|사업부)|현대로템|현대캐피탈|현대카드|삼성디스플레이|Harman|하만", re.I)
OTHER_NAMED_SUBJECT = re.compile(r"(?<![가-힣A-Za-z])(?:[A-Z][A-Za-z0-9&.]{1,20}|[가-힣A-Za-z]{2,20}(?:회사|그룹|전자|자동차|은행|캐피탈|카드|디스플레이|클라우드))(?=은|는|이|가|에서는)")
SCALES = {"조": 10**12, "천억": 10**11, "백억": 10**10, "십억": 10**9, "억": 10**8,
          "천만": 10**7, "백만": 10**6, "십만": 10**5, "만": 10**4, "천": 10**3, "백": 10**2}
NUMBER = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
SCALE = "(?:" + "|".join(sorted(SCALES, key=len, reverse=True)) + ")"
PARTS = rf"(?:{NUMBER}\s*{SCALE}\s*)+(?:{NUMBER}\s*)?|{NUMBER}\s*"
AFTER_UNIT = r"(?=$|[\s,.，;:()\[\]·]|[은는이가을를의에와과도으]|입니다|임|규모|수준|상당|가량|대|씩)"
QUANTITY = re.compile(rf"(?<![0-9.,A-Za-z])(?P<sign>[+\-−△▲])?\s*(?P<parts>{PARTS})(?P<unit>원|달러|유로|엔|대|명|건|개|주|톤){AFTER_UNIT}")
PERCENT = re.compile(rf"(?<![0-9.,A-Za-z])(?P<sign>[+\-−△▲])?\s*(?P<number>{NUMBER})\s*(?P<unit>%포인트|%[pP]|퍼센트포인트|퍼센트|%)")
BARE_SCALE = re.compile(rf"(?<![0-9.,A-Za-z])(?P<parts>{NUMBER}\s*{SCALE})(?![가-힣0-9])")
COMPONENT = re.compile(rf"(?P<number>{NUMBER})\s*(?P<scale>{SCALE})?")
CURRENCIES = {"원": "KRW", "달러": "USD", "유로": "EUR", "엔": "JPY"}
YEARS = re.compile(r"(?<![0-9])(?P<year>(?:19|20)[0-9]{2})년(?:\s*[1-4]분기|\s*상반기|\s*하반기)?")
RELATIVE_PERIOD = re.compile(r"전년\s*동기|전년|전기|당기|당해\s*연도|올해|내년")
RATIO_METRICS = {"operating_margin", "net_margin", "gross_margin", "cost_ratio", "revenue_share"}
MONEY_METRICS = {metric for metric, _ in METRICS.values()} - RATIO_METRICS - {"sales_volume"}
# These nouns introduce quantities with a different role. Do not carry a
# nearby investment/cost/etc. label across them, even if both values are money.
BINDING_BLOCKER = re.compile(r"유형자산|무형자산|유동자산|단기금융상품|특허|영업권|투자주식|임직원|직원|인원")
TOPIC_TITLE = re.compile(r"^(?:[가-하][.．]\s*|\([0-9]+\)\s*)?(?:매출\s*(?:및\s*수주)?\s*(?:실적|현황|상황)|수주\s*(?:현황|상황)|(?:주요\s*)?투자\s*현황|위험\s*요인)[.。]?$" )


def span(match):
    start, end = match.start(), match.end()
    raw = match.group()
    left = len(raw)-len(raw.lstrip())
    right = len(raw.rstrip())
    return {"start": start+left, "end": start+right, "text": raw[left:right]}


def normalize_parts(parts):
    total, prior = Decimal(0), float("inf")
    cursor = 0
    with localcontext() as context:
        context.prec = 80
        for match in COMPONENT.finditer(parts):
            if parts[cursor:match.start()].strip() or len(re.sub(r"\D", "", match["number"])) > 24:
                raise ValueError("unsupported_number")
            scale = SCALES.get(match["scale"], 1)
            if scale >= prior:
                raise ValueError("ambiguous_unit_order")
            total += Decimal(match["number"].replace(",", "")) * scale
            prior, cursor = scale, match.end()
        if not cursor or parts[cursor:].strip():
            raise ValueError("unsupported_number")
        return total


def measurements(text):
    values = []
    for kind, pattern in (("quantity", QUANTITY), ("percent", PERCENT), ("untyped", BARE_SCALE)):
        for match in pattern.finditer(text):
            where = span(match)
            if any(where["start"] < row["end"] and row["start"] < where["end"] for row in values):
                continue
            raw_unit = match["unit"] if kind != "untyped" else None
            unit = CURRENCIES.get(raw_unit, raw_unit)
            if kind == "percent":
                unit = "percentage_points" if raw_unit.lower() in {"%p", "%포인트", "퍼센트포인트"} else "percent"
            row = {**where, "unit": unit, "raw_unit": raw_unit, "normalized_value": None,
                   "status": "needs_review", "reason": None, "metric": None, "metric_binding": "unbound"}
            try:
                if kind == "untyped":
                    raise ValueError("currency_or_count_unit_missing")
                if kind == "percent" and len(re.sub(r"\D", "", match["number"])) > 24:
                    raise ValueError("unsupported_number")
                value = normalize_parts(match["parts"]) if kind == "quantity" else Decimal(match["number"].replace(",", ""))
                if match["sign"] in {"△", "▲"}:
                    raise ValueError("ambiguous_sign_symbol")
                if match["sign"] in {"-", "−"}:
                    value = value.copy_negate()
                row.update(normalized_value=format(value, "f"), status="supported")
            except (ValueError, ArithmeticError) as exc:
                row["reason"] = str(exc)
            prefix, suffix = text[max(0, row["start"]-20):row["start"]], text[row["end"]:row["end"]+8]
            qualifiers = []
            if re.search(r"약\s*$|최대\s*$|최소\s*$", prefix) or re.match(r"\s*(?:가량|정도|이상|이하|내외|대(?:의|에|를|\s|$))", suffix):
                qualifiers.append("approximate_or_bound")
            ranged = re.search(rf"{NUMBER}\s*[~∼–−\-]\s*$", prefix)
            if ranged:
                row["start"] -= len(prefix)-ranged.start()
                row["text"] = text[row["start"]:row["end"]]
                row.update(normalized_value=None, status="needs_review", reason="range_not_single_value")
            if row["unit"] in CURRENCIES.values() and prefix.endswith("(") and suffix.startswith(")"):
                qualifiers.append("parentheses_no_negative_inference")
            row["qualifiers"] = qualifiers
            values.append(row)
    values.sort(key=lambda row: row["start"])
    for left, right in zip(values, values[1:]):
        if (re.fullmatch(r"\s*[~∼–−\-]\s*", text[left["end"]:right["start"]])
                or (re.fullmatch(r"\s*", text[left["end"]:right["start"]]) and right["text"].startswith(("-", "−")))):
            for row in (left, right):
                row.update(normalized_value=None, status="needs_review", reason="range_not_single_value")
    return values


def metric_spans(text):
    result = []
    for match in METRIC.finditer(text):
        # A sales destination is not the company's reported revenue amount.
        if match.group() == "매출" and text[match.end():].startswith("처"):
            continue
        if match.group() == "투자" and re.match(r"자|주식|손익|수익|부동산|등급", text[match.end():]):
            continue
        result.append({**span(match), "metric": METRICS[match.group()][0], "topic": METRICS[match.group()][1]})
    return result


def bind_measurements(text, values, metrics):
    previous = None
    for row in values:
        barriers = [m.end() for m in re.finditer(r"[,;，]", text[:row["start"]])]
        lower = max([row["start"]-100, previous["end"] if previous else 0, *barriers])
        anchors = [m for m in metrics if lower <= m["start"] and m["end"] <= row["start"]]
        candidates = {m["metric"] for m in anchors}
        if len(candidates) == 1:
            row.update(metric=anchors[-1]["metric"], metric_binding="nearby_explicit_metric")
            blockers = list(BINDING_BLOCKER.finditer(text, anchors[-1]["end"], row["start"]))
            if blockers:
                row.update(metric=None, metric_binding="different_quantity_role", binding_blockers=[span(m) for m in blockers])
            causal = list(re.finditer(r"영향(?:\s*등)?(?:으)?로|기인(?:하여|해)|때문에", text[anchors[-1]["end"]:row["start"]]))
            if causal:
                offset = anchors[-1]["end"]
                row.update(metric=None, metric_binding="metric_in_causal_context",
                    binding_blockers=[{"start": offset+m.start(), "end": offset+m.end(), "text": m.group()} for m in causal])
        elif len(candidates) > 1:
            row["metric_binding"] = "multiple_metrics_in_clause"
        elif (previous and previous["metric"] and row["unit"] in {"percent", "percentage_points"}
              and len({m["metric"] for m in metrics}) == 1
              and re.search(r"대비|전년비|증가|감소|상승|하락", text[max(0, previous["end"]):row["end"]+12])):
            row.update(metric=previous["metric"], metric_binding="explicit_change_after_amount")
        if row["metric"] in RATIO_METRICS:
            if row["unit"] not in {"percent", "percentage_points"}:
                row.update(metric=None, metric_binding="metric_unit_mismatch")
        elif row["metric"] in MONEY_METRICS and row["unit"] not in {*CURRENCIES.values(), "percent", "percentage_points"}:
            row.update(metric=None, metric_binding="metric_unit_mismatch")
        elif row["metric"] == "sales_volume" and row["unit"] not in {"대", "개", "건", "톤", "percent", "percentage_points"}:
            row.update(metric=None, metric_binding="metric_unit_mismatch")
        if row["unit"] in {"percent", "percentage_points"} and row["metric"] not in RATIO_METRICS:
            window = text[max(0, row["start"]-25):row["end"]+15]
            if not re.search(r"대비|전년비|증가|감소|상승|하락", window):
                row.update(metric=None, metric_binding="percent_relation_unclear")
        changed = bool(re.search(r"대비|전년비|증가|감소|상승|하락", text[max(0,row["start"]-25):row["end"]+15]))
        row["relation"] = ("relative_change_candidate" if changed else "ratio_or_rate_unspecified") if row["unit"] in {"percent", "percentage_points"} else "reported_amount_or_count"
        if changed and row["unit"] == "percent" and row["metric"] in RATIO_METRICS:
            row["qualifiers"].append("rate_change_basis_requires_review")
        previous = row


def structure_sentence(text, headings, company_name):
    if TOPIC_TITLE.fullmatch(text.strip()):
        return None
    metrics = metric_spans(text)
    values = measurements(text)
    topics = sorted({m["topic"] for m in metrics} | ({"risk"} if RISK.search(text) else set()))
    numeric_expression = bool(re.search(r"[0-9][^.!?。]{0,25}(?:원|달러|%|대|명|건)", text))
    if not topics or not (values or numeric_expression or CHANGE.search(text) or "risk" in topics):
        return None
    base = classify(text, headings, company_name)
    related = [{**span(match)} for match in RELATED.finditer(text)]
    aliases = {company_name.casefold(), "네이버" if company_name in {"NAVER", "네이버"} else company_name.casefold()}
    for match in OTHER_NAMED_SUBJECT.finditer(text):
        if match.group().casefold() not in aliases and not re.search(r"당사|연결|우리", match.group()):
            related.append(span(match))
    scope = ("mixed" if base["subject"] in {"industry", "mixed"} else "related_or_segment") if related else base["subject"]
    bind_measurements(text, values, metrics)
    orientation = base["time_orientation"]
    additional_plans = [span(m) for m in re.finditer(r"방침|향후|추후|할\s*것|될\s*것", text)]
    if additional_plans:
        orientation = "mixed" if base["matched_expressions"].get("reported") else "outlook"
    if orientation == "unclear" and values and not base["matched_expressions"].get("outlook") and re.search(r"입니다[.!?。]?\s*$", text):
        orientation = "reported"
    periods = [{**span(match), "kind": "explicit_year", "year": int(match["year"])} for match in YEARS.finditer(text)]
    periods += [{**span(match), "kind": "relative_period", "year": None} for match in RELATIVE_PERIOD.finditer(text)]
    years = sorted({p["year"] for p in periods if p["year"] is not None})
    basis = [span(m) for m in re.finditer(r"연결\s*기준|별도\s*기준", text)]
    adjustments = [span(m) for m in re.finditer(r"차감|자산화|제외하고|총액|순액", text)]
    conditions = [span(m) for m in re.finditer(r"가정|전제|만약|경우", text)]
    attributions = [span(m) for m in re.finditer(r"힘입어|인하여|인해|기인|때문|영향|요인", text)]
    reasons = []
    if scope != "company":
        reasons.append("scope_requires_review")
    if orientation != "reported":
        reasons.append("not_unambiguous_reported_statement")
    if len(years) != 1 or len({re.sub(r"\s+", "", p["text"]) for p in periods if p["year"] is not None}) != 1:
        reasons.append("period_not_single_explicit_year")
    if not values or any(q["status"] != "supported" or q["metric"] is None or q["qualifiers"] for q in values):
        reasons.append("measurements_require_review")
    if numeric_expression and not values:
        reasons.append("numeric_expression_not_supported")
    if base["status"] == "needs_review":
        reasons.append("base_context_requires_review")
    if len({re.sub(r"\s+", "", b["text"]) for b in basis}) > 1:
        reasons.append("multiple_reporting_bases")
    scalar_metrics = [q["metric"] for q in values if q["unit"] not in {"percent", "percentage_points"} and q["metric"]]
    if any(scalar_metrics.count(metric) > 1 for metric in scalar_metrics):
        reasons.append("multiple_amounts_for_same_metric")
    if adjustments and values:
        reasons.append("adjustment_or_netting_context")
    if conditions:
        reasons.append("conditional_context")
    directions = [{**span(m), "direction": "increase" if m.group() in {"증가", "상승", "확대", "성장"}
                   else "decrease" if m.group() in {"감소", "하락", "축소"} else "other"} for m in CHANGE.finditer(text)]
    return {"topics": topics, "scope": scope, "time_orientation": orientation, "period_expressions": periods,
            "explicit_years": years, "measurements": values, "metrics": metrics, "change_expressions": directions,
            "basis_expressions": basis,
            "adjustment_expressions": adjustments,
            "conditional_expressions": conditions, "attribution_expressions": attributions,
            "additional_plan_expressions": additional_plans,
            "related_scope_expressions": related, "status": "needs_review" if reasons else "reported_candidate",
            "review_reasons": reasons, "classification_policy": POLICY,
            "externally_verified": False, "causal_claim_verified": False, "model_input_approved": False}


def extract_claims(blocks, event, code, company_name):
    expected = (parse_date(event["receipt_date"])+timedelta(days=1)).isoformat()+"T00:00:00+09:00"
    if event["available_from"] != expected:
        raise CollectionError("구조화 원문의 접수일과 사용 가능 시각이 일치하지 않습니다.")
    claims, skipped_tables, long_sentences = [], 0, 0
    for block in blocks:
        heading = re.sub(r"\s+", "", "/".join(block["headings"]))
        if not any(word in heading for word in ("사업의내용", "사업개요", "사업의개요", "경영진단", "영업실적", "경영성과")):
            continue
        if block["kind"] == "table":
            skipped_tables += 1
            continue
        if block["kind"] != "paragraph":
            continue
        for i, (start, end) in enumerate(sentence_spans(block["text"]), 1):
            text = block["text"][start:end]
            if len(text) > 4000:
                long_sentences += 1
                continue
            structured = structure_sentence(text, block["headings"], company_name)
            if structured is None:
                continue
            claims.append({"id": f"{code}_{event['receipt_no']}_{block['id']}_{i}", "code": code,
                "receipt_no": event["receipt_no"], "receipt_date": event["receipt_date"], "available_from": event["available_from"],
                "report_period_end": event["period_end"], "block_id": block["id"], "block_sha256": block["text_sha256"],
                "start": start, "end": end, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "headings": block["headings"], **structured})
            if len(claims) > 10_000:
                raise CollectionError("원문당 구조화 후보 수가 지원 한도를 넘었습니다.")
    return claims, {"tables_not_interpreted": skipped_tables, "long_sentences_not_structured": long_sentences}


def mark_possible_repeats(claims):
    groups = {}
    for claim in claims:
        bound = sorted((q["metric"], q["unit"], q["normalized_value"]) for q in claim["measurements"]
                       if q["metric"] and q["status"] == "supported")
        if not bound:
            continue
        # Only within one receipt; matching values do not prove identical meaning.
        payload = repr((claim["code"], claim["receipt_no"], claim["explicit_years"], claim["scope"],
                        claim["time_orientation"], bound, [p["text"] for p in claim["related_scope_expressions"]]))
        key = hashlib.sha256(payload.encode()).hexdigest()
        groups.setdefault(key, []).append(claim)
    for key, values in groups.items():
        for claim in values:
            claim["possible_repeat_group"] = key
            claim["possible_repeat_count"] = len(values)
