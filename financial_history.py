"""Receipt-specific annual XBRL facts and conservative publication-date joins."""
from __future__ import annotations

import calendar
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit

from lyupang import CollectionError, SEOUL, parse_date
from dart_client import DartError, is_annual_report

XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"
METRICS = {"Assets": "assets", "Liabilities": "liabilities", "Revenue": "revenue", "ProfitLoss": "net_income"}
FINANCIAL_COLUMNS = ("financial_available", "financial_reason", "financial_receipt_no", "financial_receipt_date",
                     "financial_available_from", "financial_period_end", "financial_scope", "debt_ratio_pct",
                     "net_margin_pct", "revenue_yoy_pct")


class FinancialError(CollectionError):
    pass


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def prior_year(day: date) -> date:
    return day.replace(year=day.year - 1, day=min(day.day, calendar.monthrange(day.year - 1, day.month)[1]))


def report_period(name: str) -> date:
    match = re.fullmatch(r"(?:\[[^\]]+\]\s*)*사업보고서\s*\(([0-9]{4})\.([0-9]{2})\)", name.strip())
    if not match:
        raise FinancialError("사업보고서의 회계기간을 명확히 읽지 못했습니다.")
    year, month = map(int, match.groups())
    try:
        return date(year, month, calendar.monthrange(year, month)[1])
    except (ValueError, calendar.IllegalMonthError):
        raise FinancialError("사업보고서의 회계기간이 올바르지 않습니다.") from None


def _context(element, identifiers: set[str], namespaces: dict):
    identifier = element.find(f"{{{XBRLI}}}entity/{{{XBRLI}}}identifier")
    text = (identifier.text or "").strip() if identifier is not None else ""
    identifier_value = re.sub(r"[^0-9]", "", text) if re.fullmatch(r"[0-9-]+", text) else text
    if identifier is None or not identifier_value or identifier_value not in identifiers:
        return None
    scope = "undimensioned"
    for node in element.iter():
        if node.tag == f"{{{XBRLDI}}}typedMember":
            return None
        if node.tag == f"{{{XBRLDI}}}explicitMember":
            if scope != "undimensioned":
                return None
            prefix = node.attrib.get("dimension", "").split(":")[0]
            axis_host = urlsplit(namespaces.get(prefix, "")).hostname or ""
            if not (axis_host == "dart.fss.or.kr" or axis_host.endswith(".ifrs.org")):
                return None
            if local(node.attrib.get("dimension", "")) != "ConsolidatedAndSeparateFinancialStatementsAxis":
                return None  # segment/member facts cannot silently stand in for whole-company figures.
            member = local((node.text or "").strip())
            if member in ("ConsolidatedMember", "ConsolidatedFinancialStatementsMember"):
                scope = "CFS"
            elif member in ("SeparateMember", "SeparateFinancialStatementsMember"):
                scope = "OFS"
            else:
                return None
    period = element.find(f"{{{XBRLI}}}period")
    if period is None:
        return None
    try:
        instant = period.findtext(f"{{{XBRLI}}}instant")
        if instant:
            return scope, None, parse_date(instant)
        return scope, parse_date(period.findtext(f"{{{XBRLI}}}startDate") or ""), parse_date(period.findtext(f"{{{XBRLI}}}endDate") or "")
    except CollectionError:
        return None


def parse_original(payload: bytes, identifiers: set[str], period_end: date) -> dict:
    """No taxonomy network resolution, no guessed segment/scope/unit conversions."""
    facts, parsed = {}, 0
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            members = [m for m in bundle.infolist() if m.filename.lower().endswith((".xbrl", ".xml"))]
            if len(members) > 100 or sum(m.file_size for m in members) > 128 * 1024 * 1024:
                raise FinancialError("원문 압축 파일이 처리 한도를 초과했습니다.")
            for member in members:
                if member.file_size > 64 * 1024 * 1024:
                    raise FinancialError("개별 재무 원문 파일이 너무 큽니다.")
                xml = bundle.read(member)
                if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
                    raise FinancialError("원문 XML에 허용되지 않은 선언이 있습니다.")
                root = ET.fromstring(xml)
                if root.tag != f"{{{XBRLI}}}xbrl":
                    continue
                parsed += 1
                namespaces = {}
                for _, (prefix, uri) in ET.iterparse(io.BytesIO(xml), events=("start-ns",)):
                    if prefix in namespaces and namespaces[prefix] != uri:
                        raise FinancialError("원문의 이름공간이 중복 정의되어 자동 해석하지 않았습니다.")
                    namespaces[prefix] = uri
                context_nodes, unit_nodes = root.findall(f"{{{XBRLI}}}context"), root.findall(f"{{{XBRLI}}}unit")
                for nodes in (context_nodes, unit_nodes):
                    ids = [node.attrib.get("id") for node in nodes]
                    if None in ids or len(ids) != len(set(ids)):
                        raise FinancialError("원문의 문맥 또는 단위 식별번호가 중복됐습니다.")
                contexts = {node.attrib.get("id"): _context(node, identifiers, namespaces) for node in context_nodes}
                units = {node.attrib.get("id") for node in unit_nodes
                         if len(node) == 1 and node[0].tag == f"{{{XBRLI}}}measure" and local(node[0].text or "") == "KRW"
                         and namespaces.get((node[0].text or "").split(":")[0]) == "http://www.xbrl.org/2003/iso4217"}
                for node in root:
                    metric = METRICS.get(local(node.tag))
                    namespace = node.tag[1:].split("}", 1)[0] if node.tag.startswith("{") else ""
                    host = urlsplit(namespace).hostname or ""
                    if not metric or not (host == "ifrs.org" or host.endswith(".ifrs.org")):
                        continue
                    context = contexts.get(node.attrib.get("contextRef"))
                    if context is None or node.attrib.get("unitRef") not in units or node.attrib.get("{http://www.w3.org/2001/XMLSchema-instance}nil") in ("true", "1"):
                        continue
                    scope, start, end = context
                    if metric in ("assets", "liabilities") and start is not None:
                        continue
                    if metric in ("revenue", "net_income") and start is None:
                        continue
                    try:
                        number = Decimal((node.text or "").strip())
                        if not number.is_finite() or abs(number) > Decimal("1e30"):
                            raise InvalidOperation
                    except InvalidOperation:
                        raise FinancialError("원문 재무 수치를 명확하게 읽지 못했습니다.") from None
                    key = (metric, scope, start, end)
                    facts.setdefault(key, set()).add(number)
    except (zipfile.BadZipFile, ET.ParseError, RuntimeError, OSError):
        raise FinancialError("접수번호별 표준 XBRL 원문을 읽지 못했습니다.") from None
    if parsed == 0:
        raise FinancialError("지원하는 표준 XBRL 인스턴스가 없습니다. 현재 재무 API 수치로 대체하지 않았습니다.")

    def value(metric, scope, start, end):
        values = facts.get((metric, scope, start, end), set())
        if len(values) > 1:
            raise FinancialError("원문에 같은 항목·기간·범위의 상충하는 값이 있습니다.")
        return next(iter(values)) if values else None

    start = prior_year(period_end) + timedelta(days=1)
    for scope in ("CFS", "undimensioned", "OFS"):
        assets, liabilities = (value(m, scope, None, period_end) for m in ("assets", "liabilities"))
        if assets is None or liabilities is None:
            continue
        if assets <= 0 or liabilities < 0:
            raise FinancialError("원문의 자산·부채 값이 사용 조건에 맞지 않습니다.")
        revenue, income = (value(m, scope, start, period_end) for m in ("revenue", "net_income"))
        prior_revenue = value("revenue", scope, prior_year(start), prior_year(period_end))
        ratio = lambda a, b: round(float(a / b * 100), 6) if a is not None and b is not None and b > 0 else None
        return {"scope": scope, "period_end": period_end.isoformat(), "period_start": start.isoformat(),
                "assets_krw": str(assets), "liabilities_krw": str(liabilities),
                "revenue_krw": str(revenue) if revenue is not None else None,
                "net_income_krw": str(income) if income is not None else None,
                "debt_ratio_pct": ratio(liabilities, assets), "net_margin_pct": ratio(income, revenue),
                "revenue_yoy_pct": ratio(revenue - prior_revenue, prior_revenue) if revenue is not None and prior_revenue is not None else None}
    raise FinancialError("동일 기업·기간·재무 범위의 자산과 부채를 확인하지 못했습니다.")


def document_bytes(cache: Path, code: str, receipt: str, client) -> tuple[bytes, str]:
    if not re.fullmatch(r"[0-9]{6}", code) or not re.fullmatch(r"[0-9]{14}", receipt):
        raise FinancialError("원문 저장용 식별번호가 올바르지 않습니다.")
    cache.mkdir(parents=True, exist_ok=True)
    for folder in sorted(cache.glob(f"{code}_{receipt}_*"), reverse=True):
        if folder.is_symlink() or not folder.is_dir():
            continue
        try:
            metadata = json.loads((folder / "source.json").read_text(encoding="utf-8"))
            payload = (folder / "original.zip").read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if (metadata.get("provider") == "OpenDART" and metadata.get("endpoint") == "fnlttXbrl.xml"
                    and metadata.get("code") == code and metadata.get("receipt_no") == receipt and metadata.get("sha256") == digest):
                return payload, digest
        except (OSError, ValueError):
            continue
    payload = client.original_xbrl(receipt)
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise FinancialError("원문 압축 파일을 받지 못해 저장하지 않았습니다.")
    digest = hashlib.sha256(payload).hexdigest()
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=cache))
    try:
        (staging / "original.zip").write_bytes(payload)
        metadata = {"provider": "OpenDART", "endpoint": "fnlttXbrl.xml", "code": code, "receipt_no": receipt,
                    "sha256": digest, "retrieved_at": datetime.now(SEOUL).isoformat()}
        (staging / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(staging, cache / f"{code}_{receipt}_{uuid.uuid4().hex[:8]}")
    except BaseException:
        shutil.rmtree(staging)
        raise
    return payload, digest


def audit_reports(code: str, corp: str, start: date, end: date, client, cache: Path, *, progress=print) -> list[dict]:
    identifiers = client.company_identifiers(corp, code)
    records = client.disclosures(corp, start - timedelta(days=550), end, annual_only=True)
    annual_records = [record for record in records if is_annual_report(record.get("report_name"))]
    if len(records) != len(annual_records):
        progress(f"  {code} 분기·반기 등 연간 대상이 아닌 보고서 {len(records) - len(annual_records)}건 제외")
    records = annual_records
    if len(records) > 30:
        raise FinancialError("종목당 연간 원문 30건 한도를 넘었습니다. 기간을 줄여주세요.")
    audited = []
    for index, filing in enumerate(records, 1):
        title = "".join(c if c.isprintable() else " " for c in filing["report_name"])[:120]
        progress(f"  {code} 연간 원문 {index}/{len(records)} · 접수일 {filing['receipt_date']} · {title}")
        period = report_period(filing["report_name"])
        received = parse_date(filing["receipt_date"])
        if period >= received or not start - timedelta(days=550) <= received <= end:
            raise FinancialError("원문의 회계기간·접수일이 올바르지 않습니다.")
        event = {**filing, "period_end": period.isoformat(), "status": "unavailable"}
        try:
            payload, digest = document_bytes(cache, code, filing["receipt_no"], client)
            metrics = parse_original(payload, identifiers, period)
            event.update(status="verified", sha256=digest, metrics=metrics)
        except DartError as exc:
            if exc.code in {"010", "011", "012", "020", "800", "901", "network"} or str(exc.code).startswith("HTTP_"):
                raise  # A different receipt will not fix an authentication/transport/quota error.
            event["reason"] = str(exc)
        except CollectionError as exc:
            event["reason"] = str(exc)
        audited.append(event)
    return audited


def financial_as_of(events: list[dict], as_of: date) -> dict:
    result = dict.fromkeys(FINANCIAL_COLUMNS)
    result.update(financial_available=False, financial_reason="no_previous_report")
    eligible = [e for e in events if parse_date(e["receipt_date"]) < as_of and parse_date(e["period_end"]) < as_of]
    if not eligible:
        return result
    period = max(e["period_end"] for e in eligible)
    latest = max(e["receipt_date"] for e in eligible if e["period_end"] == period)
    selected = [e for e in eligible if e["period_end"] == period and e["receipt_date"] == latest]
    if len(selected) != 1:
        result["financial_reason"] = "ambiguous_same_day_reports"
        return result
    event = selected[0]
    result.update(financial_receipt_no=event["receipt_no"], financial_receipt_date=event["receipt_date"],
                  financial_available_from=(parse_date(event["receipt_date"]) + timedelta(days=1)).isoformat() + "T00:00:00+09:00",
                  financial_period_end=period)
    if event["status"] != "verified":
        result["financial_reason"] = "latest_report_unverified"
        return result  # Never fall back to an older original after an unreadable correction.
    metrics = event["metrics"]
    result.update(financial_available=True, financial_reason="receipt_original_verified", financial_scope=metrics["scope"])
    for key in ("debt_ratio_pct", "net_margin_pct", "revenue_yoy_pct"):
        result[key] = metrics[key]
    return result
