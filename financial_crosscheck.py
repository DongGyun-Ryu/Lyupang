"""Receipt-specific, offline financial fact comparison. Never approves model inputs."""
from __future__ import annotations

import hashlib
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal, localcontext
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from check_facts import content, decoded, number, require
from financial_history import XBRLI, XBRLDI, _context, local, parse_original
from lyupang import CollectionError, parse_date
from report_claims import COMPONENT, SCALES

TEXT_METRICS = {"revenue": "revenue", "operating_revenue": "revenue", "net_profit": "net_income",
                "net_loss": "net_income", "operating_profit": "operating_income", "operating_loss": "operating_income"}
FACT_NAMES = {"Revenue": "revenue", "ProfitLoss": "net_income", "ProfitLossFromOperatingActivities": "operating_income"}
STATES = {"exact_match": "조건 내 표기값 같음", "rounding_overlap": "표시 정밀도 범위 겹침", "value_mismatch": "조건 내 금액 다름", "not_comparable": "조건 확인 필요"}
REASONS = {
    "financial_source_unavailable": "같은 접수번호의 재무 원문 미확인", "unsupported_amount": "대조 대상은 명확한 원화 금액에 한정",
    "statement_requires_review": "문장의 범위·시점·수치 의미 미확정", "period_unclear": "단일 연간 회계연도 미확정",
    "period_mismatch": "문장 연도와 재무 원문 회계기간 다름", "basis_unclear": "문장의 연결·별도 기준 미확정",
    "fact_missing": "같은 항목·기간·범위의 표준 항목 없음", "fact_conflict": "같은 항목·기간·범위의 값 충돌",
    "loss_sign_unclear": "손실 금액의 부호 미확정", "rounding_unknown": "서로 다른 금액의 표시 정밀도 확인 필요",
    "amount_change_context": "금액 수준이 아닌 증감액 표현 확인 필요",
    "submetric_context": "부분 금액·귀속·세전 등 다른 항목의 범위 확인 필요"}


def cached_xbrl(cache, code, receipt, expected):
    require(re.fullmatch(r"[0-9]{6}", code) and re.fullmatch(r"[0-9]{14}", receipt))
    require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected))
    folders = sorted(cache.glob(code+"_"+receipt+"_*")) if cache.exists() else []
    require(len(folders) <= 100, "같은 재무 원문의 저장 수가 처리 한도를 초과했습니다.")
    for folder in folders:
        if folder.is_symlink() or not folder.is_dir():
            continue
        try:
            metadata = decoded(content(folder / "source.json", 64 * 1024))
            if (metadata.get("provider"), metadata.get("endpoint"), metadata.get("code"), metadata.get("receipt_no"), metadata.get("sha256")) != ("OpenDART", "fnlttXbrl.xml", code, receipt, expected):
                continue
            raw = content(folder / "original.zip")
            if hashlib.sha256(raw).hexdigest() == expected:
                return raw, metadata
        except (CollectionError, OSError, ValueError):
            continue
    raise CollectionError("과거 검증 기록과 같은 해시의 재무 원문이 없습니다. 다른 접수번호로 대신하지 않습니다.")


def _host(namespace):
    return urlsplit(namespace).hostname or ""


def _standard(namespace):
    host = _host(namespace)
    return host == "ifrs.org" or host.endswith(".ifrs.org")


def parse_facts(payload, period_end, audited_metrics, corp, code):
    """Reproduce the old issuer-verified audit before reading extra traceable facts.

    Legacy caches did not retain issuer identifier sets. Require a single issuer
    and reproduction of every old amount/scope/period from the identical ZIP.
    This reuses a prior local identity check, not a new external identity proof.
    """
    facts, issuers, instance_count = [], set(), 0
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        entries = bundle.infolist()
        require(len(entries) <= 100 and sum(e.file_size for e in entries) <= 128 * 1024 * 1024)
        names = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in entry.filename
                    and ":" not in entry.filename and entry.filename not in names and not entry.flag_bits & 1
                    and (entry.external_attr >> 16) & 0o170000 != 0o120000)
            names.add(entry.filename)
            if not entry.filename.lower().endswith((".xml", ".xbrl")):
                continue
            require(entry.file_size <= 64 * 1024 * 1024)
            raw = bundle.read(entry)
            require(not re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", raw, re.I))
            root = ET.fromstring(raw)
            if root.tag != f"{{{XBRLI}}}xbrl":
                continue
            instance_count += 1
            namespaces = {}
            for _, (prefix, uri) in ET.iterparse(io.BytesIO(raw), events=("start-ns",)):
                require(prefix not in namespaces or namespaces[prefix] == uri)
                namespaces[prefix] = uri
            nodes = root.findall(f"{{{XBRLI}}}context")
            contexts, units = {}, {}
            for node in nodes:
                ident = node.find(f"{{{XBRLI}}}entity/{{{XBRLI}}}identifier")
                require(ident is not None and isinstance(ident.text, str))
                issuer = ident.text.strip()
                if re.fullmatch(r"[0-9-]+", issuer):
                    issuer = re.sub(r"[^0-9]", "", issuer)
                require(bool(issuer))
                issuers.add(issuer)
                require(node.get("id") is not None and node.get("id") not in contexts)
                contexts[node.get("id")] = _context(node, {issuer}, namespaces)
                # Member namespaces as well as axis namespaces must be known.
                for member in node.iter(f"{{{XBRLDI}}}explicitMember"):
                    prefix = (member.text or "").strip().split(":")[0]
                    ns = namespaces.get(prefix, "")
                    if not (_standard(ns) or _host(ns) == "dart.fss.or.kr"):
                        contexts[node.get("id")] = None
            for unit in root.findall(f"{{{XBRLI}}}unit"):
                require(unit.get("id") is not None and unit.get("id") not in units)
                measure = unit[0] if len(unit) == 1 else None
                text = (measure.text or "").strip() if measure is not None else ""
                units[unit.get("id")] = bool(measure is not None and measure.tag == f"{{{XBRLI}}}measure"
                    and text.endswith(":KRW") and namespaces.get(text.split(":")[0]) == "http://www.xbrl.org/2003/iso4217")
            for node in root:
                namespace = node.tag[1:].split("}", 1)[0] if node.tag.startswith("{") else ""
                concept = local(node.tag)
                metric = FACT_NAMES.get(concept) if _standard(namespace) else (
                    "operating_income" if _host(namespace) == "dart.fss.or.kr" and concept == "OperatingIncomeLoss" else None)
                context = contexts.get(node.get("contextRef"))
                if not metric or context is None or not units.get(node.get("unitRef")) or node.get("{http://www.w3.org/2001/XMLSchema-instance}nil") in ("true", "1"):
                    continue
                scope, start, end = context
                if start is None:
                    continue
                value = number((node.text or "").strip())
                require(abs(value) <= Decimal("1e30"))
                decimals = node.get("decimals")
                quantum = Decimal(0) if decimals == "INF" else (
                    Decimal(10) ** -int(decimals) if decimals is not None and re.fullmatch(r"-?[0-9]{1,2}", decimals) and abs(int(decimals)) <= 30 else None)
                facts.append({"id": f"f{len(facts)+1:06d}", "metric": metric, "value": format(value, "f"), "currency": "KRW",
                    "scope": scope, "period_start": start.isoformat(), "period_end": end.isoformat(), "member": entry.filename,
                    "concept": node.tag, "context_ref": node.get("contextRef"), "decimals": decimals,
                    "quantum": str(quantum) if quantum is not None else None})
                require(len(facts) <= 20_000, "대조용 재무 항목 수가 처리 한도를 넘었습니다.")
    require(instance_count > 0 and len(issuers) == 1, "단일 기업의 표준 XBRL 원문을 확인하지 못했습니다.")
    reproduced = parse_original(payload, issuers, period_end)
    for key in ("scope", "period_start", "period_end"):
        require(reproduced.get(key) == audited_metrics.get(key), "재무 원문이 과거 기업 검증 기록과 일치하지 않습니다.")
    for key in ("assets_krw", "liabilities_krw", "revenue_krw", "net_income_krw"):
        a, b = reproduced.get(key), audited_metrics.get(key)
        require((a is None and b is None) or (a is not None and b is not None and number(a) == number(b)),
                "재무 원문 재계산 값이 저장된 검증 기록과 다릅니다.")
    issuer = next(iter(issuers))
    return facts, {"issuer_identifier": issuer, "identity_method": "stored_verified_audit_and_same_zip_reproduction",
                   "issuer_matches_recorded_code": issuer in {corp, code}, "new_external_identity_check": False}


def text_quantum(quantity):
    raw = quantity["text"].strip()
    if quantity["raw_unit"] != "원" or not raw.endswith("원"):
        return None
    body = raw[:-1].strip().lstrip("+-−").strip()
    cursor, quantum = 0, None
    for match in COMPONENT.finditer(body):
        if body[cursor:match.start()].strip():
            return None
        fraction = match["number"].partition(".")[2]
        quantum = Decimal(SCALES.get(match["scale"], 1)) / Decimal(10) ** len(fraction)
        cursor = match.end()
    return quantum if cursor and not body[cursor:].strip() else None


def compare_amount(claim, quantity, document, *, context=None):
    result = {"claim_id": claim["id"], "code": claim["code"], "receipt_no": claim["receipt_no"], "block_id": claim["block_id"],
        "text": claim["text"], "quantity": quantity, "status": "not_comparable", "reasons": [], "financial_facts": [],
        "difference_krw": None, "rounding_half_width_krw": None, "model_input_approved": False,
        "context_resolution": context}
    reasons = result["reasons"]
    if document.get("status") != "read": reasons.append("financial_source_unavailable")
    if quantity["status"] != "supported" or quantity["unit"] != "KRW" or quantity["normalized_value"] is None or quantity["qualifiers"]:
        reasons.append("unsupported_amount")
    permitted = {"period_not_single_explicit_year", "multiple_reporting_bases", "multiple_amounts_for_same_metric"}
    grouped = set(claim.get("review_reasons", [])) & {"multiple_reporting_bases", "multiple_amounts_for_same_metric"}
    context_review = bool(context and claim["status"] == "needs_review" and context["year"] is not None and context["scope"] is not None
        and context["same_metric_scope_amounts"] == 1 and set(claim.get("review_reasons", [])) <= permitted
        and (not grouped or context["basis_method"] == "clause_explicit"))
    if (claim["status"] != "reported_candidate" and not context_review) or claim["scope"] != "company" or claim["time_orientation"] != "reported":
        reasons.append("statement_requires_review")
    explicit = [p for p in claim["period_expressions"] if p["kind"] == "explicit_year"]
    year = explicit[0]["year"] if len(explicit) == 1 and re.fullmatch(r"[0-9]{4}년", explicit[0]["text"]) else None
    if context is not None: year = context["year"]
    if re.search(r"분기|상반기|하반기|[0-9]+\s*개월|[0-9]+\s*월|말\s*현재", claim["text"]): year = None
    if year is None: reasons.append("period_unclear")
    basis = {re.sub(r"\s+", "", b["text"]) for b in claim["basis_expressions"]}
    scope = {"연결기준": "CFS", "별도기준": "OFS"}.get(next(iter(basis))) if len(basis) == 1 else None
    if context is not None: scope = context["scope"]
    if scope is None: reasons.append("basis_unclear")
    if year is not None and claim["report_period_end"] != f"{year}-12-31": reasons.append("period_mismatch")
    if quantity["metric"] in {"net_loss", "operating_loss"}: reasons.append("loss_sign_unclear")
    before = claim["text"][max(0, quantity["start"]-35):quantity["start"]]
    after = claim["text"][quantity["end"]:quantity["end"]+25]
    if (re.search(r"증감액|증가액|감소액|증가분|감소분|증대액|순증|순감", before)
            or re.match(r"\s*(?:이|가)?\s*(?:증가|감소|증감|늘|줄|증대|개선|악화)", after)):
        reasons.append("amount_change_context")
    if (re.search(r"지배주주|비지배|소유주|주당|계속영업|중단영업|세전|차감전|매출\s*채권|국내\s*매출|해외\s*매출|내수\s*매출|수출\s*매출|부문|사업부|공장|사업장|자회사", before)):
        reasons.append("submetric_context")
    if reasons:
        return result
    period_start, period_end = f"{year}-01-01", f"{year}-12-31"
    found = [f for f in document["facts"] if (f["metric"], f["scope"], f["period_start"], f["period_end"]) ==
             (TEXT_METRICS[quantity["metric"]], scope, period_start, period_end)]
    result["financial_facts"] = found
    if not found:
        reasons.append("fact_missing"); return result
    values = {number(f["value"]) for f in found}
    if len(values) != 1:
        reasons.append("fact_conflict"); return result
    value, reported = next(iter(values)), number(quantity["normalized_value"])
    with localcontext() as context:
        context.prec = 80
        difference = reported-value
        result["difference_krw"] = format(difference, "f")
        if difference == 0:
            result["status"] = "exact_match"; return result
        quantum = text_quantum(quantity)
        # Conflicting precision declarations are not silently selected.
        quanta = {f["quantum"] for f in found}
        if quantum is None or None in quanta or len(quanta) != 1:
            reasons.append("rounding_unknown"); return result
        half = (quantum + number(next(iter(quanta))))/2
        result["rounding_half_width_krw"] = format(half, "f")
        result["status"] = "rounding_overlap" if abs(difference) <= half else "value_mismatch"
    return result
